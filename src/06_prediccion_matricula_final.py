import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================
# CONFIG PRINCIPAL
# ==========================
MODEL_TAG = "xgb"  # "rf" o "xgb"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
OUT_DIR = PROJECT_DIR / "data" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_DIR = PROJECT_DIR / "data" / "config"
CURSOS_EXTERNO_CSV = CONFIG_DIR / "cursos_externo.csv"

PERIODO_OBJ = "202610"

# Ajuste “no todos se matriculan”
FACTOR_ASISTENCIA = 0.92

# Suavizado bayesiano Beta(a,b) para tasas
A_PRIOR = 2.0
B_PRIOR = 2.0

# ==========================
# TABLAS / SALIDAS (por modelo)
# ==========================
TABLA_BASE = f"demanda_simulada_{MODEL_TAG}_{PERIODO_OBJ}"
TABLA_OUT = f"matricula_esperada_{MODEL_TAG}_{PERIODO_OBJ}"
OUT_CSV = OUT_DIR / f"matricula_esperada_{MODEL_TAG}_{PERIODO_OBJ}.csv"


def beta_posterior_mean(k, n, a=A_PRIOR, b=B_PRIOR):
    return (a + k) / (a + b + n) if n > 0 else np.nan


def load_demanda_base(conn):
    # Lee la tabla generada por el 05 (según rf/xgb)
    df = pd.read_sql_query(f"SELECT * FROM {TABLA_BASE}", conn)
    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["CAMPUS"] = df["CAMPUS"].astype(str).str.strip()
    df["MODALIDAD"] = df["MODALIDAD"].astype(str).str.strip()
    return df


def load_cursos_externo():
    if not CURSOS_EXTERNO_CSV.exists():
        return set()
    df = pd.read_csv(CURSOS_EXTERNO_CSV, dtype=str)
    if "curso_id" not in df.columns:
        raise RuntimeError(f"El archivo {CURSOS_EXTERNO_CSV} debe tener columna 'curso_id'.")
    return set(df["curso_id"].astype(str).str.strip().tolist())


def calc_retiro_rates(conn):
    df = pd.read_sql_query(
        """
        SELECT
          CAMPUS,
          MODALIDAD,
          COD_ASIGNATURA AS curso_id,
          ESTADO_DESCRIPCION,
          CALIFICABLE,
          MATRICULADOS_CON_RETIRADOS AS m_con,
          MATRICULADOS_SIN_RETIRADOS AS m_sin,
          RETIRADOS AS retirados
        FROM oferta_secciones
        """,
        conn
    )

    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["CAMPUS"] = df["CAMPUS"].astype(str).str.strip()
    df["MODALIDAD"] = df["MODALIDAD"].astype(str).str.strip()

    df["ESTADO_DESCRIPCION"] = df["ESTADO_DESCRIPCION"].astype(str).str.strip().str.upper()
    df = df[df["ESTADO_DESCRIPCION"] == "ACTIVO"].copy()

    df["CALIFICABLE"] = df["CALIFICABLE"].astype(str).str.strip().str.upper()
    mask_pres = df["MODALIDAD"].eq("UC-PRESENCIAL") & df["CALIFICABLE"].eq("Y")
    mask_other = df["MODALIDAD"].isin(["UC-SEMIPRESENCIAL", "UC-A DISTANCIA"]) & df["CALIFICABLE"].eq("N")
    df = df[mask_pres | mask_other].copy()

    for c in ["m_con", "m_sin", "retirados"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["retirados_calc"] = (df["m_con"] - df["m_sin"]).clip(lower=0)
    df["retirados_use"] = np.where(df["retirados"] > 0, df["retirados"], df["retirados_calc"])

    g = df.groupby(["curso_id", "CAMPUS", "MODALIDAD"], as_index=False).agg(
        m_con_sum=("m_con", "sum"),
        retirados_sum=("retirados_use", "sum")
    )

    g["tasa_retiro"] = [
        beta_posterior_mean(k=r, n=n) for r, n in zip(g["retirados_sum"].values, g["m_con_sum"].values)
    ]

    return g[["curso_id", "CAMPUS", "MODALIDAD", "tasa_retiro", "m_con_sum", "retirados_sum"]]


def calc_desaprob_rates(conn):
    cols = pd.read_sql_query("PRAGMA table_info(matricula_notas);", conn)["name"].tolist()
    need = ["curso_id", "campus", "modalidad", "aprobado"]
    for n in need:
        if n not in cols:
            raise RuntimeError(f"Falta columna '{n}' en matricula_notas. Columnas: {cols}")

    df = pd.read_sql_query(
        """
        SELECT curso_id, campus, modalidad, aprobado
        FROM matricula_notas
        """,
        conn
    )

    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["CAMPUS"] = df["campus"].astype(str).str.strip()
    df["MODALIDAD"] = df["modalidad"].astype(str).str.strip()
    df["aprobado"] = pd.to_numeric(df["aprobado"], errors="coerce").fillna(1).astype(int)

    g = df.groupby(["curso_id", "CAMPUS", "MODALIDAD"], as_index=False).agg(
        n_total=("aprobado", "size"),
        n_noaprob=("aprobado", lambda s: int((s == 0).sum()))
    )

    g["tasa_desaprob"] = [
        beta_posterior_mean(k=k, n=n) for k, n in zip(g["n_noaprob"].values, g["n_total"].values)
    ]

    return g[["curso_id", "CAMPUS", "MODALIDAD", "tasa_desaprob", "n_total", "n_noaprob"]]


def calc_cap_hist_externo(conn):
    df = pd.read_sql_query(
        """
        SELECT
          PERIODO,
          CAMPUS,
          MODALIDAD,
          COD_ASIGNATURA AS curso_id,
          ESTADO_DESCRIPCION,
          CALIFICABLE,
          MATRICULADOS_CON_RETIRADOS AS m_con
        FROM oferta_secciones
        """,
        conn
    )

    df["PERIODO"] = df["PERIODO"].astype(str).str.strip()
    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["CAMPUS"] = df["CAMPUS"].astype(str).str.strip()
    df["MODALIDAD"] = df["MODALIDAD"].astype(str).str.strip()
    df["ESTADO_DESCRIPCION"] = df["ESTADO_DESCRIPCION"].astype(str).str.strip().str.upper()
    df["CALIFICABLE"] = df["CALIFICABLE"].astype(str).str.strip().str.upper()

    df = df[df["ESTADO_DESCRIPCION"] == "ACTIVO"].copy()

    mask_pres = df["MODALIDAD"].eq("UC-PRESENCIAL") & df["CALIFICABLE"].eq("Y")
    mask_other = df["MODALIDAD"].isin(["UC-SEMIPRESENCIAL", "UC-A DISTANCIA"]) & df["CALIFICABLE"].eq("N")
    df = df[mask_pres | mask_other].copy()

    df["m_con"] = pd.to_numeric(df["m_con"], errors="coerce").fillna(0.0)

    per = df.groupby(["curso_id", "CAMPUS", "MODALIDAD", "PERIODO"], as_index=False).agg(
        m_con_periodo=("m_con", "sum")
    )

    g = per.groupby(["curso_id", "CAMPUS", "MODALIDAD"], as_index=False).agg(
        cap_p90=("m_con_periodo", lambda s: float(np.percentile(s.values, 90)) if len(s) else 0.0),
        cap_max=("m_con_periodo", "max"),
        n_periodos=("m_con_periodo", "size")
    )

    g["cap_p90"] = g["cap_p90"].round().astype(int)
    g["cap_max"] = pd.to_numeric(g["cap_max"], errors="coerce").fillna(0).round().astype(int)
    g["n_periodos"] = pd.to_numeric(g["n_periodos"], errors="coerce").fillna(0).astype(int)

    return g


def main():
    conn = sqlite3.connect(DB_PATH)

    cursos_externo = load_cursos_externo()
    print(f"[{MODEL_TAG}] Cursos externo cargados: {len(cursos_externo)}")

    # 1) Base de demanda simulada (salida del 05)
    base = load_demanda_base(conn)

    # 2) Tasas históricas
    retiro = calc_retiro_rates(conn)
    desap = calc_desaprob_rates(conn)
    cap_hist = calc_cap_hist_externo(conn)

    # 3) Merge
    df = base.merge(retiro, on=["curso_id", "CAMPUS", "MODALIDAD"], how="left")
    df = df.merge(desap, on=["curso_id", "CAMPUS", "MODALIDAD"], how="left")
    df = df.merge(cap_hist, on=["curso_id", "CAMPUS", "MODALIDAD"], how="left")

    # 4) fallbacks
    df["tasa_retiro"] = df["tasa_retiro"].fillna(0.05)
    df["tasa_desaprob"] = df["tasa_desaprob"].fillna(0.15)
    df["cap_p90"] = df["cap_p90"].fillna(0).astype(int)
    df["cap_max"] = df["cap_max"].fillna(0).astype(int)
    df["n_periodos"] = df["n_periodos"].fillna(0).astype(int)

    # 5) Repetidores esperados
    df["p_repeat"] = (1.0 - (1.0 - df["tasa_retiro"]) * (1.0 - df["tasa_desaprob"])).clip(0, 0.60)

    df["demanda_base"] = pd.to_numeric(df["demanda_mean"], errors="coerce").fillna(0).astype(int)
    df["matricula_base_esperada"] = (df["demanda_base"] * FACTOR_ASISTENCIA).round().astype(int)
    df["repetidores_esperados"] = (df["demanda_base"] * df["p_repeat"]).round().astype(int)

    df["matricula_esperada_raw"] = (df["matricula_base_esperada"] + df["repetidores_esperados"]).astype(int)

    # Intervalos sobre p10/p90
    d10 = pd.to_numeric(df["demanda_p10"], errors="coerce").fillna(0)
    d90 = pd.to_numeric(df["demanda_p90"], errors="coerce").fillna(0)
    df["matricula_p10_raw"] = ((d10 * FACTOR_ASISTENCIA) + (d10 * df["p_repeat"])).round().astype(int)
    df["matricula_p90_raw"] = ((d90 * FACTOR_ASISTENCIA) + (d90 * df["p_repeat"])).round().astype(int)

    # Cap externo
    df["es_externo"] = df["curso_id"].isin(cursos_externo).astype(int)
    df["cap_hist_usado"] = np.where(df["cap_p90"] > 0, df["cap_p90"], df["cap_max"]).astype(int)

    df["matricula_esperada"] = np.where(
        (df["es_externo"] == 1) & (df["cap_hist_usado"] > 0),
        np.minimum(df["matricula_esperada_raw"], df["cap_hist_usado"]),
        df["matricula_esperada_raw"]
    ).astype(int)

    df["matricula_p10"] = np.where(
        (df["es_externo"] == 1) & (df["cap_hist_usado"] > 0),
        np.minimum(df["matricula_p10_raw"], df["cap_hist_usado"]),
        df["matricula_p10_raw"]
    ).astype(int)

    df["matricula_p90"] = np.where(
        (df["es_externo"] == 1) & (df["cap_hist_usado"] > 0),
        np.minimum(df["matricula_p90_raw"], df["cap_hist_usado"]),
        df["matricula_p90_raw"]
    ).astype(int)

    out = df[[
        "PERIODO", "curso_id", "CAMPUS", "MODALIDAD",
        "demanda_base",
        "tasa_retiro", "tasa_desaprob", "p_repeat",
        "matricula_base_esperada", "repetidores_esperados",
        "matricula_esperada_raw", "matricula_esperada",
        "matricula_p10_raw", "matricula_p90_raw",
        "matricula_p10", "matricula_p90",
        "es_externo", "cap_p90", "cap_max", "cap_hist_usado", "n_periodos",
        "K_SIM", "periodo_base"
    ]].copy()

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[{MODEL_TAG}] CSV generado: {OUT_CSV}")

    out.to_sql(TABLA_OUT, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLA_OUT} ON {TABLA_OUT}(curso_id, CAMPUS, MODALIDAD)")
    conn.commit()
    print(f"[{MODEL_TAG}] Tabla SQLite creada: {TABLA_OUT}")

    conn.close()


if __name__ == "__main__":
    main()
