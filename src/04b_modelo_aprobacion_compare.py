import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
OUT_DIR = PROJECT_DIR / "data" / "out"
MODEL_DIR = PROJECT_DIR / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PERIODOS = ["202400", "202410", "202420", "202500"]
TEST_PERIODO = "202510"

RANDOM_SEED = 42

# Suavizado para “fallback” en contextos raros
PRIOR_P = 0.80
PRIOR_N = 30
P_FALLBACK = 0.80


def norm_str(s):
    return s.astype(str).str.strip()

def safe_int(x, default=0):
    return pd.to_numeric(x, errors="coerce").fillna(default).astype(int)

def safe_float(x, default=np.nan):
    return pd.to_numeric(x, errors="coerce").fillna(default)

def build_preprocessor():
    num_features = ["cred", "intentos"]
    cat_features = ["programa_id", "curso_id", "modalidad", "campus"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_features),
        ]
    )
    return preprocessor, num_features + cat_features


def eval_binary(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = float("nan")
    rep = classification_report(y_true, pred, digits=4)
    return acc, auc, rep


def predict_proba_safe(clf, X):
    proba_full = clf.predict_proba(X)
    model_classes = list(clf.named_steps["model"].classes_)
    if 1 in model_classes:
        idx1 = model_classes.index(1)
        return proba_full[:, idx1]
    return np.zeros(len(X), dtype=float)


def build_prob_ctx_from_predictions(df_test, proba_pred, prior_p=PRIOR_P, prior_n=PRIOR_N):
    tmp = df_test.copy()
    tmp["p_pred"] = proba_pred.astype(float)

    # Agregar por contexto: promedio de p(modelo)
    g = tmp.groupby(["curso_id", "programa_id", "modalidad", "campus"], as_index=False).agg(
        n=("p_pred", "size"),
        p_model_mean=("p_pred", "mean")
    )

    # Suavizado hacia prior_p para contextos con pocos registros
    g["p_aprobar_model_suav"] = (g["p_model_mean"] * g["n"] + prior_p * prior_n) / (g["n"] + prior_n)
    return g


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT
          periodo,
          estudiante_id,
          programa_id,
          curso_id,
          cred,
          intentos,
          modalidad,
          campus,
          aprobado
        FROM matricula_notas
        """,
        conn
    )

    df["periodo"] = norm_str(df["periodo"])
    df["estudiante_id"] = norm_str(df["estudiante_id"])
    df["curso_id"] = norm_str(df["curso_id"])
    df["modalidad"] = norm_str(df["modalidad"])
    df["campus"] = norm_str(df["campus"])

    df["programa_id"] = safe_int(df["programa_id"])
    df["cred"] = safe_float(df["cred"])
    df["intentos"] = safe_int(df["intentos"], default=0)
    df["aprobado"] = safe_int(df["aprobado"], default=0)

    train_df = df[df["periodo"].isin(TRAIN_PERIODOS)].copy()
    test_df = df[df["periodo"] == TEST_PERIODO].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"Sin data suficiente. Train={len(train_df)} Test={len(test_df)}")

    preprocessor, feat_cols = build_preprocessor()

    X_train = train_df[feat_cols]
    y_train = train_df["aprobado"].astype(int)
    X_test = test_df[feat_cols]
    y_test = test_df["aprobado"].astype(int)

    # ====== RF
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight="balanced_subsample",
    )
    rf_clf = Pipeline(steps=[("prep", preprocessor), ("model", rf)])
    rf_clf.fit(X_train, y_train)
    rf_proba = predict_proba_safe(rf_clf, X_test)
    rf_acc, rf_auc, rf_rep = eval_binary(y_test.values, rf_proba)

    joblib.dump(rf_clf, MODEL_DIR / "rf_aprobacion.joblib")

    # tabla prob por contexto desde predicción
    rf_ctx = build_prob_ctx_from_predictions(test_df[["curso_id","programa_id","modalidad","campus"]], rf_proba)
    rf_ctx.rename(columns={"p_aprobar_model_suav":"p_aprobar_suav"}, inplace=True)
    rf_ctx.to_sql("prob_aprobar_ctx_rf", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prob_ctx_rf ON prob_aprobar_ctx_rf(curso_id, programa_id, modalidad, campus)")
    conn.commit()

    # ====== XGB (si está instalado)
    xgb_acc = xgb_auc = np.nan
    xgb_rep = ""
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )
        xgb_clf = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])
        xgb_clf.fit(X_train, y_train)

        # ojo: XGB dentro de pipeline también da predict_proba
        xgb_proba = predict_proba_safe(xgb_clf, X_test)
        xgb_acc, xgb_auc, xgb_rep = eval_binary(y_test.values, xgb_proba)

        joblib.dump(xgb_clf, MODEL_DIR / "xgb_aprobacion.joblib")

        xgb_ctx = build_prob_ctx_from_predictions(test_df[["curso_id","programa_id","modalidad","campus"]], xgb_proba)
        xgb_ctx.rename(columns={"p_aprobar_model_suav":"p_aprobar_suav"}, inplace=True)
        xgb_ctx.to_sql("prob_aprobar_ctx_xgb", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prob_ctx_xgb ON prob_aprobar_ctx_xgb(curso_id, programa_id, modalidad, campus)")
        conn.commit()

    # ====== métricas comparativas
    out = pd.DataFrame([
        {"model":"RF",  "accuracy":rf_acc, "roc_auc":rf_auc, "train_periodos":",".join(TRAIN_PERIODOS), "test_periodo":TEST_PERIODO},
        {"model":"XGB", "accuracy":xgb_acc,"roc_auc":xgb_auc,"train_periodos":",".join(TRAIN_PERIODOS), "test_periodo":TEST_PERIODO, "xgb_installed":int(HAS_XGB)},
    ])
    out.to_csv(OUT_DIR / "modelo_aprobacion_comparativa.csv", index=False, encoding="utf-8")

    print("=== RF ===")
    print(f"ACC={rf_acc:.4f} | AUC={rf_auc:.4f}")
    print(rf_rep)

    if HAS_XGB:
        print("=== XGB ===")
        print(f"ACC={xgb_acc:.4f} | AUC={xgb_auc:.4f}")
        print(xgb_rep)
    else:
        print("XGBoost no está instalado. Ejecuta: pip install xgboost")

    conn.close()


if __name__ == "__main__":
    main()
