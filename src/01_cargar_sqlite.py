import re
import sqlite3
from pathlib import Path

import pandas as pd

# =======config
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"

FILES = {
    "estudiantes": RAW_DIR / "estudiantes.xlsx",
    "cursos": RAW_DIR / "cursos.xlsx",
    "prerequisitos": RAW_DIR / "prerequisitos.xlsx",
    "periodos": RAW_DIR / "periodos.xlsx",
    "avance_estudiante": RAW_DIR / "avance_estudiante.csv",  # o .csv
    "oferta_secciones": RAW_DIR / "oferta_secciones.csv",    # o .csv
}

# CSV grandes con separador ;
CSV_SEP = ";"

# Si el CSV de avance_estudiante NO tiene cabecera, usaremos estos nombres:
AVANCE_COLS = [
    "estudiante_id", "NOMBRE", "programa_id", "CATALOGO", "plan_estudios",
    "nivel_sugerido", "curso_id", "curso_nombre", "O_E", "creditos",
    "modalidad", "campus", "periodo_id", "ESTADO"
]

# Mapeo de plan de estudios
PLAN_MAP = {"P2018": "P018", "P2024": "P024"}

def _read_excel(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_excel(path, usecols=usecols, engine="openpyxl")


def _read_csv(path: Path, sep=";", header="infer", names=None, usecols=None, nrows=None) -> pd.DataFrame:
    # Intentamos utf-8 y si falla usamos otras comunes de Excel
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=sep,
                header=header,
                names=names,
                usecols=usecols,
                encoding=enc,
                low_memory=False,
                nrows=nrows
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("No se pudo leer el CSV con encodings comunes.")


def load_table(
    path: Path,
    *,
    table_name: str,
    usecols=None,
    force_no_header: bool = False,
    names_if_no_header=None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = path.suffix.lower()

    if ext == ".xlsx":
        df = _read_excel(path, usecols=usecols)
        return df

    if ext == ".csv":
        if force_no_header:
            df = _read_csv(path, sep=CSV_SEP, header=None, names=names_if_no_header, usecols=usecols)
            return df

        # Detectar cabecera: leemos 1 fila con header=0
        probe = _read_csv(path, sep=CSV_SEP, header=0, nrows=1)
        expected_any = set((names_if_no_header or [])[:3])

        # Si no encontramos columnas esperadas, asumimos que no hay cabecera
        if names_if_no_header and not (set(probe.columns) & expected_any):
            df = _read_csv(path, sep=CSV_SEP, header=None, names=names_if_no_header, usecols=usecols)
        else:
            df = _read_csv(path, sep=CSV_SEP, header=0, usecols=usecols)
        return df

    raise ValueError(f"Extensión no soportada: {ext}")


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_text(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def sqlite_max_variables(conn: sqlite3.Connection) -> int:
    """
    Devuelve el límite de variables SQL compilado en SQLite.
    Si no se puede detectar, asume 999 (valor común).
    """
    try:
        cur = conn.cursor()
        rows = cur.execute("PRAGMA compile_options;").fetchall()
        for (opt,) in rows:
            m = re.match(r"MAX_VARIABLE_NUMBER=(\d+)", opt)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 999


def to_sqlite(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    table: str,
    if_exists: str = "replace",
    chunksize: int = 50000
):
    """
    Inserta DataFrame en SQLite evitando el error 'too many SQL variables'.

    Usamos method='multi' pero limitamos el número de filas por INSERT:
    max_rows = floor(MAX_VARIABLE_NUMBER / num_columnas)
    """
    max_vars = sqlite_max_variables(conn)
    ncols = max(1, len(df.columns))
    max_rows_per_insert = max(1, max_vars // ncols)
    safe_chunksize = min(chunksize, max_rows_per_insert)

    df.to_sql(
        table,
        conn,
        if_exists=if_exists,
        index=False,
        chunksize=safe_chunksize,
        method="multi"
    )


# ====== MAIN
def main():
    print(">> Creando BD:", DB_PATH)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # PRAGMAs para carga rápida
    cur.executescript("""
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA temp_store = MEMORY;
    PRAGMA cache_size = -200000;  -- ~200MB cache (ajusta si quieres)
    PRAGMA foreign_keys = OFF;
    """)
    conn.commit()

    # -----1) Cargar tablas pequeñas (Excel)
    print(">> Cargando estudiantes...")
    df_est = load_table(FILES["estudiantes"], table_name="estudiantes")
    df_est = normalize_strings(df_est)
    df_est = ensure_text(df_est, ["estudiante_id"])
    to_sqlite(df_est, conn, "estudiantes")

    print(">> Cargando cursos...")
    df_cur = load_table(FILES["cursos"], table_name="cursos")
    df_cur = normalize_strings(df_cur)
    df_cur = ensure_text(df_cur, ["curso_id", "plan_estudios"])
    to_sqlite(df_cur, conn, "cursos")

    print(">> Cargando prerequisitos...")
    df_pre = load_table(FILES["prerequisitos"], table_name="prerequisitos")
    df_pre = normalize_strings(df_pre)
    df_pre = ensure_text(df_pre, ["curso_id", "PRE_REQUISITO", "plan_estudios"])
    to_sqlite(df_pre, conn, "prerequisitos")

    print(">> Cargando periodos...")
    df_per = load_table(FILES["periodos"], table_name="periodos")
    df_per = normalize_strings(df_per)
    df_per = ensure_text(df_per, ["periodo_id"])
    to_sqlite(df_per, conn, "periodos")

    # --------2) Cargar avance_estudiante (grande)
    print(">> Cargando avance_estudiante...")
    df_av = load_table(
        FILES["avance_estudiante"],
        table_name="avance_estudiante",
        force_no_header=False,
        names_if_no_header=AVANCE_COLS
    )
    df_av = normalize_strings(df_av)

    if "plan_estudios" not in df_av.columns:
        raise ValueError("avance_estudiante no tiene la columna 'plan_estudios'.")

    # Normalizar plan_estudios (P2018->P018, P2024->P024)
    df_av["plan_estudios_norm"] = (
        df_av["plan_estudios"].astype(str).map(PLAN_MAP).fillna(df_av["plan_estudios"].astype(str))
    )

    df_av = ensure_text(df_av, ["estudiante_id", "curso_id", "periodo_id", "plan_estudios", "plan_estudios_norm"])
    to_sqlite(df_av, conn, "avance_estudiante", chunksize=100000)

    # ---------3) Cargar oferta_secciones (subset)
    print(">> Cargando oferta_secciones (solo columnas necesarias)...")
    oferta_cols = [
        "PERIODO", "CAMPUS", "MODALIDAD", "COD_ASIGNATURA", "CURSO", "SECCION",
        "ESTADO_DESCRIPCION", "CALIFICABLE", "RESTRICCION_PROGRAMA",
        "VACANTES_TOTALES", "MATRICULADOS_TOTALES",
        "MATRICULADOS_CON_RETIRADOS", "MATRICULADOS_SIN_RETIRADOS"
    ]

    df_of = load_table(
        FILES["oferta_secciones"],
        table_name="oferta_secciones",
        usecols=oferta_cols,
        force_no_header=False
    )
    df_of = normalize_strings(df_of)
    df_of = ensure_text(df_of, ["PERIODO", "COD_ASIGNATURA", "SECCION"])

    # Asegurar numéricos
    for c in ["VACANTES_TOTALES", "MATRICULADOS_TOTALES", "MATRICULADOS_CON_RETIRADOS", "MATRICULADOS_SIN_RETIRADOS"]:
        if c in df_of.columns:
            df_of[c] = pd.to_numeric(df_of[c], errors="coerce")

    # retirados = con - sin
    df_of["RETIRADOS"] = (df_of["MATRICULADOS_CON_RETIRADOS"] - df_of["MATRICULADOS_SIN_RETIRADOS"]).fillna(0)

    to_sqlite(df_of, conn, "oferta_secciones", chunksize=100000)

    # -------4) Expandir RESTRICCION_PROGRAMA a oferta_programas
    print(">> Expandir RESTRICCION_PROGRAMA -> oferta_programas...")
    tmp = df_of[["PERIODO", "CAMPUS", "MODALIDAD", "COD_ASIGNATURA", "SECCION", "RESTRICCION_PROGRAMA"]].copy()
    tmp["programas"] = tmp["RESTRICCION_PROGRAMA"].astype(str).str.findall(r"\d+")
    tmp = tmp.explode("programas")
    tmp.rename(columns={"programas": "programa_id"}, inplace=True)
    tmp["programa_id"] = tmp["programa_id"].where(tmp["programa_id"].notna(), None)
    to_sqlite(tmp.drop(columns=["RESTRICCION_PROGRAMA"]), conn, "oferta_programas", chunksize=100000)

    # -----5) Índices
    print(">> Creando índices...")
    cur.executescript("""
    CREATE INDEX IF NOT EXISTS idx_av_estudiante ON avance_estudiante(estudiante_id);
    CREATE INDEX IF NOT EXISTS idx_av_curso ON avance_estudiante(curso_id);
    CREATE INDEX IF NOT EXISTS idx_av_periodo ON avance_estudiante(periodo_id);
    CREATE INDEX IF NOT EXISTS idx_av_plan ON avance_estudiante(plan_estudios_norm);

    CREATE INDEX IF NOT EXISTS idx_cursos_id ON cursos(curso_id);
    CREATE INDEX IF NOT EXISTS idx_pre_curso ON prerequisitos(curso_id);
    CREATE INDEX IF NOT EXISTS idx_pre_req ON prerequisitos(PRE_REQUISITO);

    CREATE INDEX IF NOT EXISTS idx_oferta_key ON oferta_secciones(PERIODO, CAMPUS, MODALIDAD, COD_ASIGNATURA);
    CREATE INDEX IF NOT EXISTS idx_oferta_estado ON oferta_secciones(ESTADO_DESCRIPCION);
    CREATE INDEX IF NOT EXISTS idx_oferta_prog ON oferta_programas(programa_id);
    """)
    conn.commit()

    # ----- 6) Tabla derivada: demanda_historica
    
    print(">> Creando demanda_historica...")
    cur.executescript("""
    DROP TABLE IF EXISTS demanda_historica;

    CREATE TABLE demanda_historica AS
    WITH filtrada AS (
        SELECT
            PERIODO,
            CAMPUS,
            MODALIDAD,
            COD_ASIGNATURA AS curso_id,
            SECCION,
            ESTADO_DESCRIPCION,
            CALIFICABLE,
            VACANTES_TOTALES,
            MATRICULADOS_SIN_RETIRADOS,
            RETIRADOS
        FROM oferta_secciones
        WHERE
            (
              MODALIDAD = 'UC-PRESENCIAL' AND CALIFICABLE = 'Y'
            )
            OR
            (
              MODALIDAD IN ('UC-SEMIPRESENCIAL','UC-A DISTANCIA') AND CALIFICABLE = 'N'
            )
    ),
    agg AS (
        SELECT
            PERIODO,
            CAMPUS,
            MODALIDAD,
            curso_id,
            CASE
              WHEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(VACANTES_TOTALES,0) ELSE 0 END) > 0
              THEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(VACANTES_TOTALES,0) ELSE 0 END)
              ELSE MAX(COALESCE(VACANTES_TOTALES,0))
            END AS vacantes_total,
            CASE
              WHEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(MATRICULADOS_SIN_RETIRADOS,0) ELSE 0 END) > 0
              THEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(MATRICULADOS_SIN_RETIRADOS,0) ELSE 0 END)
              ELSE MAX(COALESCE(MATRICULADOS_SIN_RETIRADOS,0))
            END AS matriculados_final,
            SUM(COALESCE(RETIRADOS,0)) AS retirados_total,
            SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN 1 ELSE 0 END) AS n_secciones_activas,
            COUNT(*) AS n_secciones_total
        FROM filtrada
        GROUP BY PERIODO, CAMPUS, MODALIDAD, curso_id
    )
    SELECT * FROM agg;
    """)
    conn.commit()

    # -----7) Chequeos rápidos

    print(">> Chequeos rápidos:")
    queries = [
        "SELECT COUNT(*) AS n_estudiantes FROM estudiantes;",
        "SELECT COUNT(*) AS n_cursos FROM cursos;",
        "SELECT COUNT(*) AS n_avance FROM avance_estudiante;",
        "SELECT COUNT(*) AS n_oferta FROM oferta_secciones;",
        "SELECT COUNT(*) AS n_demanda FROM demanda_historica;",
        "SELECT MODALIDAD, COUNT(*) c FROM oferta_secciones GROUP BY MODALIDAD ORDER BY c DESC;",
    ]
    for q in queries:
        res = cur.execute(q).fetchall()
        print(q, "->", res)

    conn.close()
    print("\n Listo. BD creada en:", DB_PATH)


if __name__ == "__main__":
    main()
