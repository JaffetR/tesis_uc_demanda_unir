import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
RAW_DIR = PROJECT_DIR / "data" / "raw"

FILES = [
    RAW_DIR / "historico_notas_2024.xlsx",
    RAW_DIR / "historico_notas_2025.xlsx",
]

# Mapeo exacto segun las columnas actuales
COLMAP = {
    "SSBSECT_TERM_CODE": "periodo",
    "estudiante_id": "estudiante_id",
    "programa_id": "programa_id",
    "curso_id": "curso_id",
    "SCRSYLN_LONG_COURSE_TITLE": "curso_nombre",
    "CRED": "cred",
    "SHRTCKG_GRDE_CODE_FINAL": "nota_final",
    "intentos": "intentos",
    "resultado": "resultado",
    "modalidad": "modalidad",
    "campus": "campus",
    # opcionales utiles
    "ESTUDIANTE": "estudiante_nombre",
    "SZVMAJR_DESCRIPTION": "carrera",
    "CATALOGO": "catalogo",
    "SSBSECT_CRN": "crn",
    "PGA_PGA": "pga_pga",
    "MODULO": "modulo",
}

REQUIRED = [
    "SSBSECT_TERM_CODE",
    "estudiante_id",
    "programa_id",
    "curso_id",
    "CRED",
    "SHRTCKG_GRDE_CODE_FINAL",
    "resultado",
    "modalidad",
    "campus",
]


def _norm_text(x) -> str:
    x = "" if x is None else str(x)
    x = x.strip().lower()
    # quita acentos: "desaprobado" == "desaprobado" aunque venga con tildes raras
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    return x

def build_aprobado(resultado: pd.Series, nota_final: pd.Series) -> pd.Series:
    r = resultado.astype(str).map(_norm_text)

    # Primero NO APROBADO (prioridad)
    es_noaprob = (
        r.str.contains("desaprob") |
        r.str.contains("reprob") |
        r.str.contains("jal") |
        r.str.contains("retir") |
        r.str.contains("aband") |
        r.str.contains("fail")
    )

    # Luego APROBADO (pero excluyendo los no aprobados)
    es_aprob = r.str.contains("aprob") & (~es_noaprob)

    aprobado = np.where(es_noaprob, 0, np.where(es_aprob, 1, np.nan))

    # fallback por nota si el texto no ayudo
    nf = pd.to_numeric(nota_final, errors="coerce")
    aprobado = np.where(np.isnan(aprobado), np.where(nf >= 11, 1, 0), aprobado)

    return pd.Series(aprobado).astype(int)


def load_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    df = pd.read_excel(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Faltan columnas en {path.name}: {missing}\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    # renombrar
    ren = {k: v for k, v in COLMAP.items() if k in df.columns}
    df = df.rename(columns=ren)

    keep = list(ren.values())
    df = df[keep].copy()

    # normalizar string
    for c in ["periodo", "estudiante_id", "curso_id", "modalidad", "campus"]:
        df[c] = df[c].astype(str).str.strip()

    # tipos
    df["programa_id"] = pd.to_numeric(df["programa_id"], errors="coerce").fillna(0).astype(int)
    df["cred"] = pd.to_numeric(df["cred"], errors="coerce")
    df["nota_final"] = pd.to_numeric(df["nota_final"], errors="coerce")

    if "intentos" in df.columns:
        df["intentos"] = pd.to_numeric(df["intentos"], errors="coerce").fillna(0).astype(int)
    else:
        df["intentos"] = 0

    df["resultado"] = df["resultado"].astype(str).str.strip()

    # target
    df["aprobado"] = build_aprobado(df["resultado"], df["nota_final"])

    # limpieza básica
    df = df[(df["periodo"] != "") & (df["curso_id"] != "") & (df["estudiante_id"] != "")].copy()

    return df


def main():
    conn = sqlite3.connect(DB_PATH)

    all_df = []
    for f in FILES:
        print("Leyendo:", f.name)
        all_df.append(load_one(f))

    df = pd.concat(all_df, ignore_index=True)

    # verificación rápida antes de grabar
    print("\nValue counts resultado (top 20):")
    print(df["resultado"].astype(str).str.strip().value_counts().head(20).to_string())

    print("\nConteo aprobado (0/1):")
    print(df["aprobado"].value_counts().to_string())

    print("\nMin/Max nota_final:", df["nota_final"].min(), df["nota_final"].max())

    # guardar
    df.to_sql("matricula_notas", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_mn_periodo ON matricula_notas(periodo)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mn_ctx ON matricula_notas(curso_id, programa_id, modalidad, campus)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mn_est ON matricula_notas(estudiante_id)")
    conn.commit()

    # resumen por periodo
    resumen = pd.read_sql_query(
        """
        SELECT periodo,
               COUNT(*) n,
               COUNT(DISTINCT estudiante_id) n_est,
               SUM(aprobado) aprobados,
               (COUNT(*) - SUM(aprobado)) no_aprob
        FROM matricula_notas
        GROUP BY periodo
        ORDER BY periodo
        """,
        conn
    )
    print("\nResumen por periodo:")
    print(resumen.to_string(index=False))

    conn.close()
    print("\nTabla creada: matricula_notas")


if __name__ == "__main__":
    main()
