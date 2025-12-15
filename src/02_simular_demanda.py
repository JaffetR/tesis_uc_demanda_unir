import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import re

# config
PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
OUT_DIR = PROJECT_DIR / "data" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODO_BASE = "202520"   # desde aquí simulamos
PERIODO_OBJ = "202610"    # objetivo

# Topes de creditos
MAX_CREDITOS_REGULAR = 24
MAX_CREDITOS_VERANO = 11
MAX_CREDITOS_ABSOLUTO = 25

# primero obligatorios, luego electivos
PRIORIDAD_OBLIGATORIO = True

# Mapeo plan
PLAN_MAP = {"P2018": "P018", "P2024": "P024"}


def es_verano(periodo: str) -> bool:
    return str(periodo).endswith("00")


def normalizar_plan(plan: str) -> str:
    plan = str(plan)
    return PLAN_MAP.get(plan, plan)


def cargar_catalogos(conn):
    cursos = pd.read_sql_query(
        """
        SELECT
            curso_id,
            creditos,
            plan_estudios,
            nivel_sugerido,
            tipo_electivo,
            COD_PROGRAMA,
            CASE
              WHEN UPPER(COALESCE(COD_TIPO,'')) IN ('O','OBLIGATORIO') THEN 'O'
              WHEN UPPER(COALESCE(COD_TIPO,'')) IN ('E','ELECTIVO') THEN 'E'
              ELSE NULL
            END AS O_E_cat
        FROM cursos
        """,
        conn
    )
    cursos["plan_estudios"] = cursos["plan_estudios"].astype(str)

    prereq = pd.read_sql_query(
        """
        SELECT
            curso_id,
            PRE_REQUISITO AS prereq_id,
            plan_estudios,
            COD_PROGRAMA,
            CRED_PRE_REQUISITO2
        FROM prerequisitos
        """,
        conn
    )
    prereq["plan_estudios"] = prereq["plan_estudios"].astype(str)

    return cursos, prereq


def construir_mapa_prereq(prereq_df: pd.DataFrame):
    """
    Dict:
      key=(programa_id, plan, curso_id) -> set(prereq_id)

    FIX: si prereq_id viene como "A, B, C" en un solo campo,
    lo partimos y guardamos cada prerequisito por separado.
    """
    m = {}
    for _, r in prereq_df.iterrows():
        prog = str(r.get("COD_PROGRAMA")) if pd.notna(r.get("COD_PROGRAMA")) else None
        plan = normalizar_plan(str(r.get("plan_estudios")))
        curso = str(r.get("curso_id")).strip()
        pre_raw = r.get("prereq_id")

        if pre_raw is None or (isinstance(pre_raw, float) and np.isnan(pre_raw)):
            continue

        pre_raw = str(pre_raw).strip()
        if not pre_raw or pre_raw.lower() == "nan":
            continue

        # Split por coma (y limpia espacios). Si no hay coma, queda solo uno
        prereqs_list = [p.strip() for p in pre_raw.split(",") if p and p.strip()]

        key = (prog, plan, curso)
        for pre in prereqs_list:
            m.setdefault(key, set()).add(pre)

    return m


def cargar_creditos_aprobados(conn):
    """
    Dict: estudiante_id (str) -> creditos_aprobados (int)
    """
    try:
        df = pd.read_sql_query(
            """
            SELECT
              CAST(estudiante_id AS TEXT) AS estudiante_id,
              creditos_aprobados
            FROM creditos_estudiante
            """,
            conn
        )
    except Exception:
        return {}

    df["estudiante_id"] = df["estudiante_id"].astype(str)
    df["creditos_aprobados"] = pd.to_numeric(df["creditos_aprobados"], errors="coerce").fillna(0).astype(int)
    return dict(zip(df["estudiante_id"], df["creditos_aprobados"]))


def construir_mapa_credito_minimo(prereq_df: pd.DataFrame):
    """
    Dict: key=(programa_id, plan, curso_id) -> cred_min (int)
    NaN => 0
    """
    col = None
    for c in prereq_df.columns:
        if str(c).upper() == "CRED_PRE_REQUISITO2":
            col = c
            break

    m = {}
    if col is None:
        return m

    for _, r in prereq_df.iterrows():
        prog = str(r.get("COD_PROGRAMA")) if pd.notna(r.get("COD_PROGRAMA")) else None
        plan = normalizar_plan(str(r.get("plan_estudios")))
        curso = str(r.get("curso_id")).strip()

        val = pd.to_numeric(r.get(col), errors="coerce")
        cred_min = 0 if pd.isna(val) else int(val)

        key = (prog, plan, curso)
        m[key] = max(m.get(key, 0), cred_min)

    return m


def crear_demanda_historica_si_no(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demanda_historica';")
    if cur.fetchone():
        return

    cur.executescript("""
    DROP TABLE IF EXISTS demanda_historica;

    CREATE TABLE demanda_historica AS
    WITH filtrada AS (
        SELECT
            CAST(PERIODO AS TEXT) AS PERIODO,
            CAMPUS,
            MODALIDAD,
            COD_ASIGNATURA AS curso_id,
            SECCION,
            ESTADO_DESCRIPCION,
            CALIFICABLE,
            VACANTES_TOTALES,
            MATRICULADOS_SIN_RETIRADOS
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
              WHEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(MATRICULADOS_SIN_RETIRADOS,0) ELSE 0 END) > 0
              THEN SUM(CASE WHEN ESTADO_DESCRIPCION='ACTIVO' THEN COALESCE(MATRICULADOS_SIN_RETIRADOS,0) ELSE 0 END)
              ELSE MAX(COALESCE(MATRICULADOS_SIN_RETIRADOS,0))
            END AS matriculados_final
        FROM filtrada
        GROUP BY PERIODO, CAMPUS, MODALIDAD, curso_id
    )
    SELECT * FROM agg;
    """)
    conn.commit()


def preferencias_electivos(conn, cursos_df: pd.DataFrame):
    oferta_prog = pd.read_sql_query(
        """
        SELECT
          CAST(PERIODO AS TEXT) AS PERIODO,
          CAMPUS, MODALIDAD,
          COD_ASIGNATURA AS curso_id,
          RESTRICCION_PROGRAMA
        FROM oferta_secciones
        """,
        conn
    )
    oferta_prog["programa_id"] = oferta_prog["RESTRICCION_PROGRAMA"].astype(str).str.findall(r"\d+")
    oferta_prog = oferta_prog.explode("programa_id")
    oferta_prog["programa_id"] = oferta_prog["programa_id"].astype(str)

    dh = pd.read_sql_query("SELECT * FROM demanda_historica", conn)
    dh["PERIODO"] = dh["PERIODO"].astype(str)

    tmp = dh.merge(
        oferta_prog[["PERIODO", "CAMPUS", "MODALIDAD", "curso_id", "programa_id"]],
        on=["PERIODO", "CAMPUS", "MODALIDAD", "curso_id"],
        how="left"
    )

    cat = cursos_df[["curso_id", "tipo_electivo", "plan_estudios", "COD_PROGRAMA"]].copy()
    cat["COD_PROGRAMA"] = cat["COD_PROGRAMA"].astype(str)
    tmp = tmp.merge(cat, on="curso_id", how="left")

    tmp = tmp[tmp["tipo_electivo"].notna()].copy()

    g = (
        tmp.groupby(
            ["programa_id", "plan_estudios", "CAMPUS", "MODALIDAD", "tipo_electivo", "curso_id"],
            dropna=False
        )["matriculados_final"]
        .sum()
        .reset_index()
    )
    g = g.sort_values(
        ["programa_id", "plan_estudios", "CAMPUS", "MODALIDAD", "tipo_electivo", "matriculados_final"],
        ascending=[True, True, True, True, True, False]
    )
    top = g.groupby(["programa_id", "plan_estudios", "CAMPUS", "MODALIDAD", "tipo_electivo"], as_index=False).first()
    top.rename(columns={"curso_id": "curso_electivo_top"}, inplace=True)
    return top


def simular_estudiante(
    rows_est: pd.DataFrame,
    prereq_map,
    cursos_cat: pd.DataFrame,
    pref_top: pd.DataFrame,
    periodo_obj: str,
    cred_aprob_map: dict,
    cred_min_map: dict
):
    estudiante_id = str(rows_est["estudiante_id"].iloc[0])
    programa_id = str(rows_est["programa_id"].iloc[0]) if "programa_id" in rows_est.columns else None
    campus = str(rows_est["campus"].iloc[0]) if "campus" in rows_est.columns else None
    modalidad = str(rows_est["modalidad"].iloc[0]) if "modalidad" in rows_est.columns else None
    plan = normalizar_plan(str(rows_est["plan_estudios"].iloc[0])) if "plan_estudios" in rows_est.columns else None

    cred_aprob = int(cred_aprob_map.get(estudiante_id, 0))

    max_creditos = MAX_CREDITOS_VERANO if es_verano(periodo_obj) else MAX_CREDITOS_REGULAR
    max_creditos = min(max_creditos, MAX_CREDITOS_ABSOLUTO)

    pendientes = rows_est[rows_est["ESTADO"].astype(int) == 0].copy()
    pendientes_set = set(pendientes["curso_id"].astype(str).tolist())

    # tipos de electivo ya cubiertos por ESTADO=1
    llevando_df = rows_est[rows_est["ESTADO"].astype(int) == 1][["curso_id"]].copy()
    llevando_df["curso_id"] = llevando_df["curso_id"].astype(str)
    llevando_df = llevando_df.merge(
        cursos_cat[["curso_id", "tipo_electivo"]],
        on="curso_id",
        how="left"
    )
    electivo_tipos_cubiertos = set(
        llevando_df.loc[llevando_df["tipo_electivo"].notna(), "tipo_electivo"].astype(str).tolist()
    )
    electivo_tipos_seleccionados = set(electivo_tipos_cubiertos)

    # prereqs deben estar aprobados antes (si está pendiente en base => NO)
    def es_elegible(curso_id: str) -> bool:
        key = (programa_id, plan, curso_id)
        prereqs = prereq_map.get(key, set())
        for pre in prereqs:
            if pre in pendientes_set:
                return False
        return True

    def cumple_creditos_minimos(curso_id: str) -> bool:
        key = (programa_id, plan, curso_id)
        req = int(cred_min_map.get(key, 0))
        return cred_aprob >= req

    pendientes["nivel_sugerido"] = pd.to_numeric(pendientes["nivel_sugerido"], errors="coerce")
    pendientes = pendientes.sort_values(["nivel_sugerido"])

    seleccionados = []
    creditos_acum = 0

    for ciclo, grp in pendientes.groupby("nivel_sugerido", dropna=False):
        if creditos_acum >= max_creditos:
            break

        grp = grp.copy()
        grp["curso_id"] = grp["curso_id"].astype(str)

        oblig = grp[grp["O_E"].astype(str).str.upper().str.startswith("O")] if PRIORIDAD_OBLIGATORIO else pd.DataFrame()
        elec = grp[grp["O_E"].astype(str).str.upper().str.startswith("E")]

        # 1) Obligatorios
        for _, r in oblig.iterrows():
            cid = str(r["curso_id"])
            cred = int(pd.to_numeric(r["creditos"], errors="coerce") or 0)
            if cred <= 0:
                continue
            if not es_elegible(cid):
                continue
            if creditos_acum + cred > max_creditos:
                continue
            seleccionados.append(cid)
            creditos_acum += cred

        if creditos_acum >= max_creditos:
            break

        # 2) Electivos: 1 por tipo_electivo + fallback
        if len(elec) > 0:
            elec2 = elec.merge(
                cursos_cat[["curso_id", "tipo_electivo"]],
                on="curso_id",
                how="left"
            )

            for tipo, g2 in elec2.groupby("tipo_electivo", dropna=True):
                if creditos_acum >= max_creditos:
                    break

                tipo = str(tipo)
                if tipo in electivo_tipos_seleccionados:
                    continue

                g2 = g2.copy()
                g2["curso_id"] = g2["curso_id"].astype(str)

                top_row = pref_top[
                    (pref_top["programa_id"].astype(str) == str(programa_id)) &
                    (pref_top["plan_estudios"].astype(str) == str(plan)) &
                    (pref_top["CAMPUS"].astype(str) == str(campus)) &
                    (pref_top["MODALIDAD"].astype(str) == str(modalidad)) &
                    (pref_top["tipo_electivo"].astype(str) == tipo)
                ]
                top_choice = str(top_row["curso_electivo_top"].iloc[0]) if len(top_row) > 0 else None

                candidatos = []
                if top_choice and top_choice in set(g2["curso_id"]):
                    candidatos.append(top_choice)
                for cid in g2["curso_id"].tolist():
                    if cid != top_choice:
                        candidatos.append(cid)

                elegido_final = None
                cred_final = 0

                for cid_try in candidatos:
                    row_try = g2[g2["curso_id"] == cid_try]
                    if len(row_try) == 0:
                        continue

                    cred_try = int(pd.to_numeric(row_try["creditos"].iloc[0], errors="coerce") or 0)
                    if cred_try <= 0:
                        continue
                    if not es_elegible(cid_try):
                        continue
                    if not cumple_creditos_minimos(cid_try):
                        continue
                    if creditos_acum + cred_try > max_creditos:
                        continue

                    elegido_final = cid_try
                    cred_final = cred_try
                    break

                if elegido_final is None:
                    continue

                seleccionados.append(elegido_final)
                creditos_acum += cred_final
                electivo_tipos_seleccionados.add(tipo)

    return estudiante_id, programa_id, campus, modalidad, plan, seleccionados, creditos_acum


def main():
    conn = sqlite3.connect(DB_PATH)

    crear_demanda_historica_si_no(conn)

    cursos_cat, prereq_df = cargar_catalogos(conn)
    prereq_map = construir_mapa_prereq(prereq_df)

    cred_aprob_map = cargar_creditos_aprobados(conn)
    cred_min_map = construir_mapa_credito_minimo(prereq_df)

    pref_top = preferencias_electivos(conn, cursos_cat)

    query = f"""
    SELECT
      estudiante_id,
      programa_id,
      plan_estudios,
      nivel_sugerido,
      curso_id,
      curso_nombre,
      O_E,
      creditos,
      modalidad,
      campus,
      CAST(periodo_id AS TEXT) AS periodo_id,
      ESTADO
    FROM avance_estudiante
    WHERE CAST(periodo_id AS TEXT) = '{PERIODO_BASE}'
    ORDER BY estudiante_id
    """

    chunks = pd.read_sql_query(query, conn, chunksize=200000)

    buffer = pd.DataFrame()
    resultados = []

    for chunk in chunks:
        chunk = chunk.copy()
        chunk["plan_estudios"] = chunk["plan_estudios"].astype(str)
        chunk["curso_id"] = chunk["curso_id"].astype(str)
        chunk["estudiante_id"] = chunk["estudiante_id"].astype(str)

        if not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)
            buffer = pd.DataFrame()

        last_id = chunk["estudiante_id"].iloc[-1]
        mask_last = chunk["estudiante_id"] == last_id
        buffer = chunk[mask_last].copy()
        chunk2 = chunk[~mask_last].copy()

        for est_id, rows_est in chunk2.groupby("estudiante_id"):
            _, prog, campus, mod, plan, cursos_sel, cred = simular_estudiante(
                rows_est, prereq_map, cursos_cat, pref_top, PERIODO_OBJ, cred_aprob_map, cred_min_map
            )
            for cid in cursos_sel:
                resultados.append({
                    "periodo_obj": PERIODO_OBJ,
                    "estudiante_id": est_id,
                    "programa_id": prog,
                    "campus": campus,
                    "modalidad": mod,
                    "plan_estudios_norm": plan,
                    "curso_id": cid
                })

    if not buffer.empty:
        for est_id, rows_est in buffer.groupby("estudiante_id"):
            _, prog, campus, mod, plan, cursos_sel, cred = simular_estudiante(
                rows_est, prereq_map, cursos_cat, pref_top, PERIODO_OBJ, cred_aprob_map, cred_min_map
            )
            for cid in cursos_sel:
                resultados.append({
                    "periodo_obj": PERIODO_OBJ,
                    "estudiante_id": est_id,
                    "programa_id": prog,
                    "campus": campus,
                    "modalidad": mod,
                    "plan_estudios_norm": plan,
                    "curso_id": cid
                })

    df_res = pd.DataFrame(resultados)
    if df_res.empty:
        print("No se generaron selecciones. Revisa columnas/periodo_base.")
        return

    demanda_base = (
        df_res.groupby(["periodo_obj", "campus", "modalidad", "curso_id"])
        .size()
        .reset_index(name="demanda_base_simulada")
    )

    out1 = OUT_DIR / f"seleccion_estudiantes_{PERIODO_OBJ}.csv"
    out2 = OUT_DIR / f"demanda_base_simulada_{PERIODO_OBJ}.csv"
    df_res.to_csv(out1, index=False, encoding="utf-8")
    demanda_base.to_csv(out2, index=False, encoding="utf-8")

    df_res.to_sql("seleccion_simulada", conn, if_exists="replace", index=False)
    demanda_base.to_sql("demanda_base_simulada", conn, if_exists="replace", index=False)

    print("Listo:")
    print(" -", out1)
    print(" -", out2)
    print(" - Tablas SQLite: seleccion_simulada, demanda_base_simulada")

    conn.close()


if __name__ == "__main__":
    main()
