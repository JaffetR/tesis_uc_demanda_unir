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

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
OUT_DIR = PROJECT_DIR / "data" / "out"
MODEL_DIR = PROJECT_DIR / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PERIODOS = ["202400", "202410", "202420", "202500"]
TEST_PERIODO = "202510"

PRIOR_P = 0.80
PRIOR_N = 30

RANDOM_SEED = 42

def norm_str(s):
    return s.astype(str).str.strip()

def safe_int(x, default=0):
    return pd.to_numeric(x, errors="coerce").fillna(default).astype(int)

def safe_float(x, default=np.nan):
    return pd.to_numeric(x, errors="coerce").fillna(default)

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
        raise RuntimeError(
            f"No hay datos suficientes. Train({TRAIN_PERIODOS})={len(train_df)} | Test({TEST_PERIODO})={len(test_df)}"
        )

    # Diagnóstico: distribución de clases
    diag = (
        df[df["periodo"].isin(TRAIN_PERIODOS + [TEST_PERIODO])]
        .groupby(["periodo", "aprobado"], as_index=False)
        .size()
        .pivot(index="periodo", columns="aprobado", values="size")
        .fillna(0)
        .astype(int)
    )
    print("Distribución aprobado (0/1) por periodo:")
    print(diag.to_string())

    y_train = train_df["aprobado"].astype(int)
    clases_train = sorted(y_train.unique().tolist())

    print(f"Entrenando RF con train_periodos: {TRAIN_PERIODOS} | test_period: {TEST_PERIODO}")
    print("Clases en train:", clases_train)

    num_features = ["cred", "intentos"]
    cat_features = ["programa_id", "curso_id", "modalidad", "campus"]

    X_train = train_df[num_features + cat_features]
    X_test = test_df[num_features + cat_features]
    y_test = test_df["aprobado"].astype(int)

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

    # Caso 1: si train tiene 2 clases, entrenamos normal
    if len(clases_train) >= 2:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
        )

        clf = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

        clf.fit(X_train, y_train)

        proba_test_full = clf.predict_proba(X_test)

        # Mapear probabilidad de clase 1 de forma segura
        model_classes = list(clf.named_steps["model"].classes_)
        if 1 in model_classes:
            idx1 = model_classes.index(1)
            proba_test = proba_test_full[:, idx1]
        else:
            # raro, pero por seguridad
            proba_test = np.zeros(len(X_test), dtype=float)

        pred_test = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_test)
        try:
            auc = roc_auc_score(y_test, proba_test)
        except Exception:
            auc = float("nan")

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC_AUC: {auc:.4f}")
        print(classification_report(y_test, pred_test))

        model_path = MODEL_DIR / "rf_aprobacion.joblib"
        joblib.dump(clf, model_path)
        print(f"Modelo guardado: {model_path}")

    else:
        # Caso 2: train tiene solo una clase entonces no se puede entrenar un clasificador util
        # Fallback: usar tasa global de aprobación en train como prob constante
        only_class = clases_train[0]
        tasa = float(y_train.mean())  # será 0.0 o 1.0
        # suavizamos hacia PRIOR_P para evitar extremos
        tasa_suav = (tasa * len(train_df) + PRIOR_P * PRIOR_N) / (len(train_df) + PRIOR_N)

        print("Advertencia: train tiene una sola clase. No se entrena RF.")
        print(f"Clase única en train: {only_class} | tasa_train={tasa:.4f} | tasa_suav={tasa_suav:.4f}")

        clf = None
        proba_test = np.full(len(X_test), tasa_suav, dtype=float)
        pred_test = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_test)
        try:
            auc = roc_auc_score(y_test, proba_test)
        except Exception:
            auc = float("nan")

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC_AUC: {auc:.4f}")
        print(classification_report(y_test, pred_test))

    # Guardar métricas
    metricas_path = OUT_DIR / "modelo_aprobacion_metricas.csv"
    pd.DataFrame([{
        "train_periodos": ",".join(TRAIN_PERIODOS),
        "test_periodo": TEST_PERIODO,
        "accuracy": acc,
        "roc_auc": auc,
        "train_classes": ",".join(map(str, clases_train)),
        "fallback_used": int(len(clases_train) < 2),
    }]).to_csv(metricas_path, index=False, encoding="utf-8")
    print(f"Métricas guardadas: {metricas_path}")

    # Construir probabilidad por contexto usando proporción empírica suavizada (siempre se puede)
    g = (
        test_df
        .groupby(["curso_id", "programa_id", "modalidad", "campus"], as_index=False)
        .agg(n=("aprobado", "size"), aprobados=("aprobado", "sum"))
    )
    g["p_aprobar_emp"] = g["aprobados"] / g["n"]
    g["p_aprobar_suav"] = (g["aprobados"] + PRIOR_P * PRIOR_N) / (g["n"] + PRIOR_N)

    # Guardar en SQLite
    g.to_sql("prob_aprobar_ctx", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prob_ctx ON prob_aprobar_ctx(curso_id, programa_id, modalidad, campus)")
    conn.commit()
    print("Tabla SQLite creada: prob_aprobar_ctx")

    out_csv = OUT_DIR / "prob_aprobar_ctx_testperiod.csv"
    g.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"CSV generado: {out_csv}")

    conn.close()


if __name__ == "__main__":
    main()
