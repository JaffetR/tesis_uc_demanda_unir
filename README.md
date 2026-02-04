# tesis_uc_demanda_unir

**Modelo predictivo de demanda académica para la optimización de apertura de grupos en educación superior**

Este repositorio contiene el **pipeline computacional desarrollado como parte de la tesis de máster** titulada:

> *Modelo predictivo de demanda académica para la optimización de apertura de grupos en educación superior*

El objetivo del proyecto es **apoyar la programación académica** mediante la estimación de demanda y matrícula esperada por curso, campus y modalidad, integrando **reglas académicas reales**, **aprendizaje automático** y **simulación Monte Carlo**, con salidas operativas reproducibles.

---

## 🎯 Alcance del repositorio (confidencialidad)

Por razones de confidencialidad institucional, este repositorio **NO incluye**:

- Datos académicos originales (Excel/CSV en `data/raw/`)
- Base de datos SQLite generada (`db/tesis_uc.db`)
- Modelos entrenados (`.joblib`)
- Salidas con información identificable de estudiantes

El repositorio **SÍ incluye**:

- Código fuente completo del pipeline (`src/`)
- Estructura de carpetas para reproducibilidad (`data/`, `db/` con `.gitkeep`)
- Scripts de simulación, modelado y comparación
- Documentación técnica y figuras sin datos sensibles (si aplica)

---

## 🧠 Enfoque metodológico

El pipeline implementa un **enfoque híbrido** compuesto por:

- **Simulación académica por estudiante**, aplicando reglas reales:
  - prerequisitos
  - topes de créditos
  - priorización de cursos obligatorios/electivos
- **Modelo supervisado de aprobación**:
  - Random Forest (modelo base)
  - XGBoost (modelo comparativo)
- **Simulación Monte Carlo (K_SIM)** para incorporar incertidumbre:
  - estimación de demanda media
  - percentiles p10 / p90
- **Transformación a matrícula esperada** y recomendación de secciones

---

## 📂 Estructura del proyecto

```text
tesis_uc/
├── src/                    # Scripts del pipeline
│   ├── 01_cargar_sqlite.py
│   ├── 02_simular_demanda.py
│   ├── 03_cargar_historico_notas_batch.py
│   ├── 04_modelo_aprobacion_rf.py
│   ├── 04b_modelo_aprobacion_compare.py
│   ├── 05_simular_demanda_montecarlo.py
│   └── 06_prediccion_matricula_final.py
│
├── data/
│   ├── raw/                # Entradas (vacío en el repo)
│   ├── out/                # Salidas (vacío en el repo)
│   ├── config/             # Parámetros, catálogos y reglas
│   └── models/             # Modelos entrenados (excluidos)
│
├── db/
│   └── .gitkeep            # Base SQLite excluida del repo
│
├── .gitignore
└── README.md
