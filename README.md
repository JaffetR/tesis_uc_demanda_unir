# tesis_uc_demanda_unir

Modelo predictivo de demanda académica para apoyar la programación de secciones (grupos) por **curso / campus / modalidad / periodo**, utilizando un enfoque híbrido:
- simulación académica por estudiante (reglas: prerequisitos, tope de créditos, priorización O/E),
- modelo supervisado de aprobación (Random Forest),
- simulación Monte Carlo (K_SIM) para estimar demanda con incertidumbre (p10/p90),
- conversión a matrícula esperada y recomendación de secciones según capacidad y mínimo de apertura.

## Alcance del repositorio (sin datos sensibles)
Este repositorio **NO incluye**:
- Datos institucionales (Excel/CSV originales en `data/raw/`)
- Base SQLite generada (`db/tesis_uc.db`)
- Modelos entrenados pesados (`data/models/*.joblib`)

Por confidencialidad, solo se publica:
- Código fuente (`src/`)
- Configuración y plantillas (`data/config/`)
- Estructura de carpetas con `.gitkeep`
- Figuras/documentación generada sin información identificable (si aplica)

## Estructura del proyecto
- `src/` : scripts del pipeline
- `data/raw/` : entradas (vacío en repo, solo estructura)
- `data/out/` : salidas (vacío en repo, solo estructura)
- `data/config/` : parámetros, catálogos y reglas
- `data/models/` : modelos entrenados (excluidos del repo)
- `db/` : base local SQLite (excluida del repo)
- `figures/` : gráficos/exportables (opcional)

## Pipeline (orden recomendado)
1. `01_cargar_sqlite.py` : integra fuentes y crea `db/tesis_uc.db`
2. `02_simular_demanda.py` : genera demanda base por reglas (determinística)
3. `03_cargar_historico_notas_batch.py` : integra histórico de notas para entrenamiento
4. `04_modelo_aprobacion_rf.py` : entrena Random Forest y genera probabilidades por contexto
5. `05c_simular_demanda_con_rf.py` : simulación Monte Carlo (K_SIM) para demanda p10/p90
6. `06_prediccion_matricula_final.py` : matrícula esperada y variables operativas finales

## Reproducibilidad
- Requiere acceso autorizado a datos institucionales para poblar `data/raw/`.
- Se recomienda ejecutar en un entorno virtual:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt` (si se incluye)

## Autoría
Repositorio de autoría exclusiva del estudiante:
- Jaffet Robie Chanco Porta
