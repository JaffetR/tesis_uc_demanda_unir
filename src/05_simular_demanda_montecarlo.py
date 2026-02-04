import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

MODEL_TAG = "xgb"   # o "xgb" o "rf"
PROB_TABLE = f"prob_aprobar_ctx_{MODEL_TAG}"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "db" / "tesis_uc.db"
OUT_DIR = PROJECT_DIR / "data" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODO_OBJ = "202610"

MAX_CRED_REG = 24
MAX_CRED_ABS = 25
MAX_CRED_VERANO = 11

K_SIM = 50
SEED = 42

P_FALLBACK = 0.80

PLAN_MAP = {"P2018": "P018", "P2024": "P024"}

MOD_MAP = {
    "UREG": "UC-PRESENCIAL",
    "UPGT": "UC-SEMIPRESENCIAL",
    "UVIR": "UC-A DISTANCIA",
    "UC-PRESENCIAL": "UC-PRESENCIAL",
    "UC-SEMIPRESENCIAL": "UC-SEMIPRESENCIAL",
    "UC-A DISTANCIA": "UC-A DISTANCIA",
}


def _strip(x):
    return "" if x is None else str(x).strip()


def norm_plan(x):
    x = _strip(x)
    return PLAN_MAP.get(x, x)


def norm_mod(x):
    x = _strip(x)
    return MOD_MAP.get(x, x)


def pick_col(cols, candidates):
    cols_low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None


def table_cols(conn, table):
    df = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
    return df["name"].tolist()


def ensure_col(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df


def load_prob_aprobar(conn):
    cols = table_cols(conn, PROB_TABLE)
    need = ["curso_id", "programa_id", "modalidad", "campus", "p_aprobar_suav"]
    for n in need:
        if n not in cols:
            raise RuntimeError(f"Falta columna '{n}' en prob_aprobar_ctx. Columnas: {cols}")

    df = pd.read_sql_query(
        f"""
        SELECT curso_id, programa_id, modalidad, campus, p_aprobar_suav
        FROM {PROB_TABLE}
        """,
        conn
    )

    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["modalidad"] = df["modalidad"].astype(str).str.strip().map(norm_mod)
    df["campus"] = df["campus"].astype(str).str.strip()
    df["programa_id"] = pd.to_numeric(df["programa_id"], errors="coerce").fillna(0).astype(int)
    df["p_aprobar_suav"] = pd.to_numeric(df["p_aprobar_suav"], errors="coerce").fillna(P_FALLBACK).astype(float)

    return {
        (r["curso_id"], int(r["programa_id"]), r["modalidad"], r["campus"]): float(r["p_aprobar_suav"])
        for _, r in df.iterrows()
    }


def build_prereq_map(conn):
    cols = table_cols(conn, "prerequisitos")
    c_curso = pick_col(cols, ["curso_id", "COD_CUR"])
    c_pre = pick_col(cols, ["PRE_REQUISITO", "pre_requisito", "PREREQUISITO"])
    c_prog = pick_col(cols, ["COD_PROGRAMA", "programa_id"])
    c_plan = pick_col(cols, ["plan_estudios", "PLAN_EST", "plan_estudios_norm"])

    if not all([c_curso, c_pre, c_prog, c_plan]):
        raise RuntimeError(f"No pude detectar columnas en prerequisitos. Columnas: {cols}")

    df = pd.read_sql_query(
        f"""
        SELECT
          {c_curso} AS curso_id,
          {c_pre}   AS prereq_id,
          {c_prog}  AS programa_id,
          {c_plan}  AS plan_estudios
        FROM prerequisitos
        """,
        conn
    )

    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["prereq_id"] = df["prereq_id"].astype(str).str.strip()
    df["programa_id"] = pd.to_numeric(df["programa_id"], errors="coerce").fillna(0).astype(int)
    df["plan_estudios"] = df["plan_estudios"].astype(str).str.strip().map(norm_plan)

    prereq_map = defaultdict(set)
    for _, r in df.iterrows():
        if r["curso_id"] and r["prereq_id"]:
            prereq_map[(r["curso_id"], int(r["programa_id"]), r["plan_estudios"])].add(r["prereq_id"])
    return prereq_map


def load_cursos_catalog(conn):
    cols = table_cols(conn, "cursos")

    c_curso = pick_col(cols, ["curso_id", "COD_CUR"])
    c_prog = pick_col(cols, ["programa_id", "COD_PROGRAMA"])
    c_plan = pick_col(cols, ["plan_estudios", "PLAN_EST", "plan_estudios_norm"])
    c_ciclo = pick_col(cols, ["nivel_sugerido", "ciclo_asig", "CICLO_ASIG"])
    c_oe = pick_col(cols, ["O_E", "o_e"])  # importante: tu columna es O_E
    c_cred = pick_col(cols, ["creditos", "cred", "CREDITO_ASIG", "CRED"])
    c_tipoe = pick_col(cols, ["tipo_electivo", "TIPO_ELECTIVO"])

    if not c_curso:
        raise RuntimeError(f"No pude detectar curso_id en cursos. Columnas: {cols}")

    sel = [f"{c_curso} AS curso_id"]
    if c_prog: sel.append(f"{c_prog} AS programa_id")
    if c_plan: sel.append(f"{c_plan} AS plan_estudios")
    if c_ciclo: sel.append(f"{c_ciclo} AS ciclo_asig")
    if c_oe: sel.append(f"{c_oe} AS o_e")
    if c_cred: sel.append(f"{c_cred} AS cred")
    if c_tipoe: sel.append(f"{c_tipoe} AS tipo_electivo")

    df = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM cursos", conn)

    # asegurar columnas (evita el bug del .get() devolviendo string)
    ensure_col(df, "programa_id", 0)
    ensure_col(df, "plan_estudios", "")
    ensure_col(df, "ciclo_asig", 999)
    ensure_col(df, "o_e", "")
    ensure_col(df, "cred", 0.0)
    ensure_col(df, "tipo_electivo", "")

    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["programa_id"] = pd.to_numeric(df["programa_id"], errors="coerce").fillna(0).astype(int)
    df["plan_estudios"] = df["plan_estudios"].astype(str).str.strip().map(norm_plan)
    df["ciclo_asig"] = pd.to_numeric(df["ciclo_asig"], errors="coerce").fillna(999).astype(int)
    df["o_e"] = df["o_e"].astype(str).str.strip().str.upper()
    df["cred"] = pd.to_numeric(df["cred"], errors="coerce").fillna(0).astype(float)
    df["tipo_electivo"] = df["tipo_electivo"].astype(str).str.strip()

    cat = {}
    for _, r in df.iterrows():
        key = (r["curso_id"], int(r["programa_id"]), r["plan_estudios"])
        cat[key] = {
            "ciclo_asig": int(r["ciclo_asig"]),
            "o_e": r["o_e"],
            "cred": float(r["cred"]),
            "tipo_electivo": r["tipo_electivo"],
        }
    return cat


def load_avance_estudiante(conn):
    cols = table_cols(conn, "avance_estudiante")

    c_est = pick_col(cols, ["estudiante_id"])
    c_prog = pick_col(cols, ["programa_id"])
    c_plan = pick_col(cols, ["plan_estudios", "plan_estudios_norm"])
    c_ciclo = pick_col(cols, ["nivel_sugerido", "ciclo_asig"])
    c_curso = pick_col(cols, ["curso_id"])
    c_oe = pick_col(cols, ["O_E", "o_e"])
    c_cred = pick_col(cols, ["creditos", "cred", "CREDITO_ASIG"])
    c_mod = pick_col(cols, ["modalidad"])
    c_camp = pick_col(cols, ["campus"])
    c_periodo = pick_col(cols, ["periodo_id", "periodo", "PERACAD"])
    c_estado = pick_col(cols, ["ESTADO", "estado"])

    miss = [("estudiante", c_est), ("programa", c_prog), ("plan", c_plan), ("ciclo", c_ciclo),
            ("curso", c_curso), ("O_E", c_oe), ("cred", c_cred),
            ("modalidad", c_mod), ("campus", c_camp), ("periodo", c_periodo), ("estado", c_estado)]
    miss2 = [name for name, col in miss if col is None]
    if miss2:
        raise RuntimeError(f"No pude detectar columnas en avance_estudiante: {miss2}. Columnas: {cols}")

    df = pd.read_sql_query(
        f"""
        SELECT
          {c_est}     AS estudiante_id,
          {c_prog}    AS programa_id,
          {c_plan}    AS plan_estudios,
          {c_ciclo}   AS ciclo_asig,
          {c_curso}   AS curso_id,
          {c_oe}      AS o_e,
          {c_cred}    AS cred,
          {c_mod}     AS modalidad,
          {c_camp}    AS campus,
          {c_periodo} AS periodo,
          {c_estado}  AS estado
        FROM avance_estudiante
        """,
        conn
    )

    df["estudiante_id"] = df["estudiante_id"].astype(str).str.strip()
    df["programa_id"] = pd.to_numeric(df["programa_id"], errors="coerce").fillna(0).astype(int)
    df["plan_estudios"] = df["plan_estudios"].astype(str).str.strip().map(norm_plan)
    df["ciclo_asig"] = pd.to_numeric(df["ciclo_asig"], errors="coerce").fillna(999).astype(int)
    df["curso_id"] = df["curso_id"].astype(str).str.strip()
    df["o_e"] = df["o_e"].astype(str).str.strip().str.upper()
    df["cred"] = pd.to_numeric(df["cred"], errors="coerce").fillna(0).astype(float)
    df["modalidad"] = df["modalidad"].astype(str).str.strip().map(norm_mod)
    df["campus"] = df["campus"].astype(str).str.strip()
    df["periodo"] = df["periodo"].astype(str).str.strip()
    df["estado"] = pd.to_numeric(df["estado"], errors="coerce").fillna(0).astype(int)

    return df


def credit_cap(periodo: str) -> int:
    p = _strip(periodo)
    return MAX_CRED_VERANO if p.endswith("00") else MAX_CRED_REG


def prereq_satisfied(curso_id, programa_id, plan_estudios, prereq_map, passed_proxy):
    reqs = prereq_map.get((curso_id, programa_id, plan_estudios), set())
    if not reqs:
        return True
    for r in reqs:
        if r not in passed_proxy:
            return False
    return True


def main():
    np.random.seed(SEED)
    conn = sqlite3.connect(DB_PATH)

    print("Cargando prob_aprobar_ctx...")
    prob_dict = load_prob_aprobar(conn)

    print("Cargando prerequisitos...")
    prereq_map = build_prereq_map(conn)

    print("Cargando cursos...")
    cursos_cat = load_cursos_catalog(conn)

    print("Cargando avance_estudiante...")
    df_av = load_avance_estudiante(conn)

    periodo_base = df_av["periodo"].max()
    print(f"Periodo base detectado: {periodo_base}")

    grupos = list(df_av.groupby("estudiante_id"))
    total_est = len(grupos)
    print(f"Total estudiantes: {total_est}")

    all_counts = []

    for k in range(K_SIM):
        conteo = defaultdict(int)

        for i, (_, rows) in enumerate(grupos, start=1):
            programa_id = int(rows["programa_id"].iloc[0])
            plan = _strip(rows["plan_estudios"].iloc[0])

            campus = rows["campus"].mode().iloc[0] if len(rows["campus"].mode()) else _strip(rows["campus"].iloc[0])
            modalidad = rows["modalidad"].mode().iloc[0] if len(rows["modalidad"].mode()) else _strip(rows["modalidad"].iloc[0])

            en_prog = rows[rows["estado"] == 1]["curso_id"].astype(str).tolist()
            pend = rows[rows["estado"] == 0].copy()

            # grupos electivos ya satisfechos por estar en progreso (si catálogo tiene tipo_electivo)
            grupos_satis = set()
            for cid in en_prog:
                info = cursos_cat.get((cid, programa_id, plan))
                if info and info.get("o_e") == "E":
                    g = _strip(info.get("tipo_electivo", ""))
                    if g:
                        grupos_satis.add(g)

            # simulación aprobación cursos en progreso
            aprobados_sim = set()
            no_aprob_sim = set()
            for cid in en_prog:
                p = prob_dict.get((cid, programa_id, modalidad, campus), P_FALLBACK)
                if np.random.rand() <= p:
                    aprobados_sim.add(cid)
                    info = cursos_cat.get((cid, programa_id, plan))
                    if info and info.get("o_e") == "E":
                        g = _strip(info.get("tipo_electivo", ""))
                        if g:
                            grupos_satis.add(g)
                else:
                    no_aprob_sim.add(cid)

            pend_set = set(pend["curso_id"].astype(str).tolist())
            no_passed = pend_set.union(no_aprob_sim)

            class PassedProxy(set):
                def __contains__(self, item):
                    return (item not in no_passed) or (item in aprobados_sim)

            passed = PassedProxy()

            # candidatos ordenados por ciclo, obligatorios primero, más crédito primero
            cand = []
            for _, r in pend.iterrows():
                cid = _strip(r["curso_id"])
                info = cursos_cat.get((cid, programa_id, plan), None)

                ciclo = int(info.get("ciclo_asig", r["ciclo_asig"])) if info else int(r["ciclo_asig"])
                oe = _strip(info.get("o_e", r["o_e"])).upper() if info else _strip(r["o_e"]).upper()
                cred = float(info.get("cred", r["cred"])) if info else float(r["cred"])
                tipo_e = _strip(info.get("tipo_electivo", "")) if info else ""

                cand.append((ciclo, 0 if oe == "O" else 1, -cred, cid, oe, cred, tipo_e))

            cand.sort()

            cap = credit_cap(PERIODO_OBJ)
            creditos = 0.0
            grupos_usados = set(grupos_satis)
            seleccion = []

            for ciclo, _, _, cid, oe, cred, tipo_e in cand:
                if creditos >= cap:
                    break
                if creditos + cred > MAX_CRED_ABS:
                    continue

                if oe == "E" and tipo_e and tipo_e in grupos_usados:
                    continue

                if not prereq_satisfied(cid, programa_id, plan, prereq_map, passed):
                    continue

                seleccion.append(cid)
                creditos += cred

                if oe == "E" and tipo_e:
                    grupos_usados.add(tipo_e)

            for cid in seleccion:
                conteo[(cid, campus, modalidad)] += 1

            if i % 50000 == 0:
                print(f"Run {k+1}/{K_SIM} - procesados {i}/{total_est}")

        all_counts.append(conteo)
        print(f"Run {k+1}/{K_SIM} terminado. Filas demanda: {len(conteo)}")

    keys = set()
    for d in all_counts:
        keys.update(d.keys())

    out_rows = []
    for (cid, campus, modalidad) in sorted(keys):
        vals = np.array([d.get((cid, campus, modalidad), 0) for d in all_counts], dtype=float)
        out_rows.append({
            "PERIODO": PERIODO_OBJ,
            "curso_id": cid,
            "CAMPUS": campus,
            "MODALIDAD": modalidad,
            "demanda_mean": int(round(vals.mean())),
            "demanda_p10": int(round(np.percentile(vals, 10))),
            "demanda_p90": int(round(np.percentile(vals, 90))),
            "K_SIM": int(K_SIM),
            "periodo_base": str(periodo_base),
        })

    df_out = pd.DataFrame(out_rows)

    out_csv = OUT_DIR / f"demanda_simulada_{MODEL_TAG}_{PERIODO_OBJ}.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"CSV generado: {out_csv}")

    table = f"demanda_simulada_{MODEL_TAG}_{PERIODO_OBJ}"
    df_out.to_sql(table, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table}(curso_id, CAMPUS, MODALIDAD)")
    conn.commit()
    print(f"Tabla SQLite creada: {table}")

    conn.close()


if __name__ == "__main__":
    main()
