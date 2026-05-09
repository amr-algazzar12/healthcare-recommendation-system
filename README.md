<div align="center">

# 🏥 Healthcare Recommendation System

**End-to-end big-data pipeline for AI-powered clinical decision support**

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5.3-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-24.3-FFCC01?style=flat-square&logo=clickhouse&logoColor=black)](https://clickhouse.com)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.1-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.0-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose)

</div>

---

The Healthcare Recommendation System ingests synthetic EHR records from Synthea™, engineers patient features at scale with Apache Spark, trains three complementary recommendation models (ALS collaborative filtering, TF-IDF content-based, XGBoost hybrid), and serves personalised medication recommendations through a Flask REST API and a five-page Streamlit dashboard — all orchestrated by Apache Airflow, all running from a single `docker compose up`.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [ClickHouse Schema](#clickhouse-schema)
- [Feature Engineering](#feature-engineering)
- [Recommendation Models](#recommendation-models)
- [Flask API](#flask-api)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Airflow DAGs](#airflow-dags)
- [Makefile Reference](#makefile-reference)
- [Service URLs & Ports](#service-urls--ports)
- [Credentials](#credentials)
- [Environment Variables](#environment-variables)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Team](#team)

---

## Architecture

Six Airflow DAGs chain end-to-end via `TriggerDagRunOperator`. Every service lives inside a single Docker Compose network (`healthcare_net`).

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA SOURCE                                                            │
│  Synthea™ EHR generator → 6 CSV files in data/raw/                      │
│  (patients, conditions, medications, observations, encounters,          │
│   procedures)                                                           │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
           ┌─────────────▼─────────────────────────┐
           │  dag_ingest_to_hdfs    [Milestone 1]  │
           │  validate CSVs → upload to HDFS       │
           └─────────────┬─────────────────────────┘
                         │  TriggerDagRunOperator
           ┌─────────────▼─────────────────────────┐
           │  dag_ingest_clickhouse [Milestone 1]  │
           │  bulk-insert raw CSVs into ClickHouse │
           └─────────────┬─────────────────────────┘
                         │  TriggerDagRunOperator
           ┌─────────────▼─────────────────────────┐
           │  dag_spark_processing  [Milestone 2]  │
           │  Spark: clean.py → feature_engineering│
           │  output: HDFS features + CH table     │
           └─────────────┬─────────────────────────┘
                         │  TriggerDagRunOperator
           ┌─────────────▼─────────────────────────┐
           │  dag_train_models      [Milestone 3]  │
           │  ALS → content-based → XGBoost hybrid │
           │  all runs logged to MLflow            │
           └─────────────┬─────────────────────────┘
                         │  TriggerDagRunOperator
           ┌─────────────▼─────────────────────────┐
           │  dag_evaluate_and_register            │
           │  compare models → promote best →      │
           │  MLflow Production stage              │
           └─────────────┬─────────────────────────┘
                         │  TriggerDagRunOperator
           ┌─────────────▼─────────────────────────┐
           │  dag_deploy_and_serve  [Milestone 4]  │
           │  restart app with Production model    │
           └─────────────┬─────────────────────────┘
                         │
           ┌─────────────▼─────────────────────────┐
           │  SERVING LAYER                        │
           │  Flask REST API      :5050            │
           │  Streamlit Dashboard :8501            │
           └───────────────────────────────────────┘

Infrastructure (always running):
  postgres · clickhouse · namenode · datanode
  resourcemanager · nodemanager · historyserver
  spark-master · spark-worker
  airflow-webserver · airflow-scheduler · airflow-init
  mlflow · pgadmin
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Data generation | Synthea™ EHR Simulator | Latest |
| Distributed storage | Apache Hadoop HDFS | 3.2.1 |
| OLAP database | ClickHouse | 24.3 |
| Airflow metadata DB | PostgreSQL | 15.3 |
| Batch processing | Apache Spark + PySpark | 3.5.3 |
| ML — collaborative filtering | Spark MLlib ALS | 3.5.3 |
| ML — content-based | scikit-learn TF-IDF | 1.4.0 |
| ML — hybrid ensemble | XGBoost | 2.0.3 |
| Experiment tracking | MLflow | 2.10.0 |
| Orchestration | Apache Airflow | 2.8.1 |
| API server | Flask + Gunicorn | 3.0 / 21.2 |
| Dashboard | Streamlit | 1.30 |
| Containerisation | Docker Compose v2 | — |

---

## Prerequisites

| Tool | Minimum | Notes |
|---|---|---|
| Docker | 24.x | BuildKit must be enabled |
| Docker Compose | v2.x | Use `docker compose`, not `docker-compose` |
| Java | 11+ | Required only for `make generate-data` (Synthea) |
| GNU Make | 3.8+ | Pre-installed on Linux / macOS |
| RAM | 12 GB free | 16 GB recommended when Spark workers are active |
| Disk | 20 GB | Volumes for HDFS, PostgreSQL, MLflow artifacts, model files |

> **macOS:** in Docker Desktop → Settings → Resources, raise Memory to at least 12 GB.

---

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/amr-algazzar12/healthcare-recommendation-system.git
cd healthcare-recommendation-system
```

### 2. Configure environment

```bash
cp .env.example .env   # defaults work out of the box for local development
```

### 3. Generate synthetic patient data

> Requires Java 11. Runs Synthea with a fixed seed; outputs ~1,000 patient records into `data/raw/`.

```bash
make generate-data
```

### 4. Start the cluster

```bash
make up

make fix-airflow         # change the ownership of log/airflow to 5000 for the aiflow container to access and modify it
```

Wait ~90 seconds for all health checks to pass, then confirm everything is up:

```bash
make verify
```

### 5. Initialise storage

```bash
make init-everything     # creates HDFS dirs + ClickHouse schema + Airflow admin user
make load-hdfs           # uploads data/raw/*.csv to HDFS
make load-clickhouse     # bulk-inserts raw CSVs into ClickHouse
```

### 6. Run the full pipeline

**Via Airflow (recommended):**

Open [http://localhost:8081](http://localhost:8081), log in as `admin / admin_secret_2026`, enable all 6 dags, and trigger `dag_ingest_to_hdfs`. Each DAG automatically triggers the next.


### 7. Open the interfaces

| Interface | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| Flask REST API | http://localhost:5050/health |
| Airflow UI | http://localhost:8081 |
| MLflow UI | http://localhost:5001 |
| Spark Master UI | http://localhost:8080 |
| HDFS Namenode UI | 9870 | http://localhost:9870 |

---

## Project Structure

```
healthcare-recommendation-system/
│
├── dags/
│   ├── dag_ingest_to_hdfs.py           # M1 — validate CSVs, upload to HDFS
│   ├── dag_ingest_clickhouse.py        # M1 — bulk-insert raw tables into ClickHouse
│   ├── dag_spark_processing.py         # M2 — Spark clean + feature engineering
│   ├── dag_train_models.py             # M3 — ALS → content-based → hybrid
│   ├── dag_evaluate_and_register.py    # M3 — evaluate + MLflow registry promotion
│   └── dag_deploy_and_serve.py         # M4 — restart app with Production model
│
├── src/
│   ├── ingestion/
│   │   └── validate.py                 # CSV pre-flight checks (row counts, schema)
│   ├── clickhouse/
│   │   └── load_tables.py              # CSV → ClickHouse bulk insert
│   ├── processing/
│   │   ├── clean.py                    # Spark: type cast, deduplicate, null handling
│   │   └── feature_engineering.py      # Spark: demographics, TF-IDF vectors, flags
│   ├── models/
│   │   ├── collaborative_filtering.py  # ALS (Spark MLlib)
│   │   ├── content_based.py            # TF-IDF cosine similarity (sklearn v3)
│   │   ├── hybrid_model.py             # XGBoost binary classifier ensemble
│   │   ├── evaluate.py                 # Unified evaluation + MLflow promotion
│   │   └── als_grid_search.py          # ALS hyperparameter sweep (α, rank, maxIter)
│   ├── recommender/
│   │   └── engine.py                   # Thread-safe model loader + top-K scorer
│   ├── api/
│   │   └── app.py                      # Flask app — 7 endpoints, CORS enabled
│   ├── dashboard/
│   │   └── app.py                      # Streamlit 5-page dashboard
│   └── utils/
│       ├── config.py                   # Centralised env-var configuration
│       ├── clickhouse_client.py        # ClickHouse connection helper
│       └── spark_session.py            # SparkSession factory
│
├── docker/
│   ├── app/
│   │   ├── Dockerfile                  # Flask + Streamlit + ML libs image
│   │   ├── entrypoint.sh               # Waits for CH + MLflow, launches Gunicorn + Streamlit
│   │   └── requirements.txt            # Pinned Python dependencies
│   ├── spark/
│   │   └── Dockerfile                  # Spark + PySpark + MLflow + XGBoost
│   ├── mlflow/
│   │   └── Dockerfile                  # MLflow tracking server
│   └── clickhouse/
│       ├── config.xml                  # Server config (ports, logging)
│       ├── users.xml                   # User quotas and access policies
│       └── init/
│           └── 01_create_schema.sql    # DDL for all 8 tables
│
├── data/
│   ├── synthea/
│   │   ├── generate_data.sh            # Runs Synthea with fixed seed
│   │   └── synthea.properties          # CSV-only output, no FHIR
│   ├── raw/                            # Synthea CSV output (gitignored)
│   ├── processed/                      # Spark-cleaned Parquet (gitignored)
│   └── features/                       # Engineered feature Parquet (gitignored)
│
├── scripts/
│   ├── airflow_create_user.sh          # Create Airflow admin user
│   ├── load_clickhouse.sh              # Shell-based ClickHouse CSV loader
│   └── postgres/
│       └── init-mlflow-db.sql          # Creates mlflow DB inside PostgreSQL
│
├── models/
│   └── best_model.json                 # Written by evaluate.py after promotion
│
├── configs/
│   └── airflow_notes.md                # Airflow env-var configuration notes
│
├── conda/
│   └── environment.yml                 # Conda env for local development
│
├── notebooks/                          # Exploration notebooks
├── tests/                              # Integration tests
├── mlflow/artifacts/                   # MLflow artifact store (gitignored)
├── logs/airflow/                       # Airflow task logs (gitignored)
├── docker-compose.yml
├── hadoop.env                          # Hadoop cluster environment variables
├── Makefile
├── .env                                # Credentials — never commit
└── .gitignore
```

---

## Data Pipeline

### Source — Synthea™

[Synthea](https://github.com/synthetichealth/synthea) generates realistic, fully de-identified EHR records. The project uses a fixed seed for reproducibility and outputs CSV only (`synthea.properties` disables FHIR, CCDA, and text exports).

```bash
make generate-data   # requires Java 11; writes to data/raw/
```

Six CSV files are produced:

| File | Key columns | Description |
|---|---|---|
| `patients.csv` | `Id`, `BIRTHDATE`, `GENDER`, `RACE`, `STATE` | Core demographics |
| `conditions.csv` | `PATIENT`, `CODE`, `DESCRIPTION`, `START`, `STOP` | Diagnosis history |
| `medications.csv` | `PATIENT`, `CODE`, `DESCRIPTION`, `START`, `STOP` | Prescription records |
| `observations.csv` | `PATIENT`, `CODE`, `VALUE`, `UNITS`, `DATE` | Labs and vitals |
| `encounters.csv` | `PATIENT`, `Id`, `START`, `STOP`, `ENCOUNTERCLASS` | Visit records |
| `procedures.csv` | `PATIENT`, `CODE`, `DESCRIPTION`, `DATE` | Clinical procedures |

### Ingestion

Raw CSVs flow into two parallel stores before any Spark processing begins:

```
data/raw/*.csv
    │
    ├── dag_ingest_to_hdfs ──────► hdfs://namenode:9001/data/raw/
    │   (WebHDFS upload)            (Spark reads from here in cluster mode)
    │
    └── dag_ingest_clickhouse ───► healthcare.{patients, conditions,
        (load_tables.py)             medications, observations,
                                     encounters, procedures}
```

`dag_ingest_to_hdfs` runs `validate.py` first. This confirms every expected CSV exists, has at least one data row, and passes schema sanity checks. The DAG fails immediately on the first violation before touching HDFS.

### Cleaning — `clean.py`

Spark reads raw CSVs from HDFS and applies:

- Type casting for dates, timestamps, and nullable numeric columns
- Null removal for structurally required columns; imputation where appropriate
- Deduplication on primary keys
- Column renaming to snake_case to match the ClickHouse schema exactly
- Safe handling of empty `STOP` dates (active conditions and medications)

Output: Parquet at `hdfs://namenode:9001/data/processed/`

---

## ClickHouse Schema

Eight tables in the `healthcare` database (`docker/clickhouse/init/01_create_schema.sql`). All use MergeTree with ORDER BY optimised for the primary query pattern.

| Table | ORDER BY | Description |
|---|---|---|
| `patients` | `patient_id` | Raw demographics |
| `conditions` | `(patient_id, start_date)` | Diagnosis history |
| `medications` | `(patient_id, start_date)` | Prescription history |
| `observations` | `(patient_id, date)` | Labs and vitals |
| `encounters` | `(patient_id, start_dt)` | Visit records |
| `procedures` | `(patient_id, date)` | Clinical procedures |
| `patient_features` | `patient_id` | Engineered ML features |
| `recommendations` | `(patient_id, created_at)` | Cached model output |

The `patient_features` table is the contract between the feature engineering job and all three models:

```sql
CREATE TABLE healthcare.patient_features (
    patient_id               String,
    age                      Int32,
    gender_encoded           Int8,
    race_encoded             Int8,
    num_conditions           Int32,
    num_medications          Int32,
    num_encounters           Int32,
    has_diabetes             UInt8,
    has_hypertension         UInt8,
    has_asthma               UInt8,
    has_hyperlipidemia       UInt8,
    has_coronary_disease     UInt8,
    condition_vector         Array(Float32),   -- 50-dim TF-IDF over condition codes
    medication_history_flags Array(Float32),   -- 50-dim TF-IDF over medication codes
    feature_version          String,
    created_at               DateTime64(3) DEFAULT now()
) ENGINE = MergeTree() ORDER BY patient_id;
```

---

## Feature Engineering

`src/processing/feature_engineering.py` is a PySpark job that reads cleaned Parquet from HDFS and populates `patient_features`.

### Chronic disease flags

Binary flags use verified Synthea SNOMED-CT codes:

| Column | Condition codes |
|---|---|
| `has_diabetes` | 44054006, 73211009, 314771006 |
| `has_hypertension` | 59621000 |
| `has_asthma` | 195967001 |
| `has_hyperlipidemia` | 55822004 |
| `has_coronary_disease` | 53741008, 414545008, 22298006 |

### TF-IDF feature vectors

`condition_vector` is a 50-dimensional `Array(Float32)` built over the top-50 most frequent condition codes across the patient population (`TOP_N_CONDITIONS = 50`).

`medication_history_flags` is a 50-dimensional `Array(Float32)` built over the top-50 medication codes using the same TF-IDF approach (`TOP_N_MEDICATIONS = 50`).

Both arrays use `Float32` to match the ClickHouse column type and minimise storage.

### Scalar features

| Column | Type | Description |
|---|---|---|
| `age` | `Int32` | Computed from `BIRTHDATE` |
| `gender_encoded` | `Int8` | 0 = F, 1 = M |
| `race_encoded` | `Int8` | Label-encoded from string |
| `num_conditions` | `Int32` | Distinct active diagnoses |
| `num_medications` | `Int32` | Distinct medications ever prescribed |
| `num_encounters` | `Int32` | Total visit count |

Output is written to `hdfs://namenode:9001/data/features/patient_features` (Parquet, Snappy compressed, 8 shuffle partitions) and then loaded into `healthcare.patient_features`.

---

## Recommendation Models

All three models log to the same MLflow experiment (`healthcare-recommendations`). `evaluate.py` runs after training, compares them, and promotes the winner to the `Production` stage, writing the result to `models/best_model.json`.

### Model 1 — ALS Collaborative Filtering

**File:** `src/models/collaborative_filtering.py`  
**Trigger:** `make run-als`  
**Framework:** Spark MLlib  
**MLflow registry name:** `healthcare-collaborative-filtering`

Builds a patient × medication implicit feedback matrix from prescription history and trains ALS:

```python
ALS_PARAMS = {
    "rank":              10,
    "maxIter":           10,
    "regParam":          0.1,
    "alpha":             1.0,       # confidence scaling for implicit feedback
    "implicitPrefs":     True,
    "coldStartStrategy": "drop",
}
```

Evaluated on a held-out 20% split. Logged metrics: `rmse`, `precision_at_k`, `ndcg_at_k`. Model artifact saved to `hdfs://namenode:9001/models/collaborative_filtering`.

---

### Model 2 — Content-Based Filtering (v3)

**File:** `src/models/content_based.py`  
**Trigger:** `make run-content-based`  
**Framework:** scikit-learn  
**MLflow registry name:** `healthcare-content-based`

TF-IDF vectorises condition and medication features, then uses cosine similarity between patient profiles to find relevant medications. Version 3 adds:

- **ROC-AUC** as the primary optimisation metric (macro-averaged per patient)
- Score calibration: min-max normalisation → sigmoid sharpening
- Condition-cohort IDF: weights computed within condition subgroups rather than globally
- Neighbour confidence weighting: `sim² / (sim + ε)` per neighbour
- Score floor blending with global medication popularity to prevent zero-probability negatives
- Optional grid search over `recency_decay` and `jaccard_weight` (enable with `RUN_GRID_SEARCH=True`)

Logged metrics: `mean_auc_roc`, `mean_auc_pr`, `ndcg_at_k`.

---

### Model 3 — Hybrid XGBoost Ensemble

**File:** `src/models/hybrid_model.py`  
**Trigger:** `make run-hybrid`  
**Framework:** XGBoost 2.0.3  
**MLflow registry name:** `healthcare-hybrid-xgboost`

Frames recommendation as binary classification: positive samples are `(patient, medication)` pairs that were actually prescribed; negatives are randomly sampled at a 3:1 ratio.

Feature set per pair:

```python
SCALAR_FEATURE_COLS = [
    "age", "gender_encoded", "race_encoded",
    "num_conditions", "num_medications", "num_encounters",
    "has_diabetes", "has_hypertension", "has_asthma",
    "has_hyperlipidemia", "has_coronary_disease",
]
# Two additional content-based signals:
ALL_FEATURE_COLS = SCALAR_FEATURE_COLS + ["sim_med_count", "med_prevalence"]
```

Training hyperparameters:

```python
XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}
```

Model serialised with `joblib` and saved to `hdfs://namenode:9001/models/hybrid_xgboost/model.joblib`. Logged metrics: `auc_roc`, `auc_pr`, `precision_at_k`, `ndcg_at_k`.

This is the **current best model** (score `0.9684`) and is in Production by default.

---

### Evaluation and Promotion

`evaluate.py` is pure Python — no Spark session required. It:

1. Queries MLflow for the latest finished run per `model_type` parameter
2. Normalises each model's primary metric to a comparable score:

| Model | Primary metric | Normalisation |
|---|---|---|
| `collaborative_filtering` | `rmse` | `1 / (1 + rmse)` — lower RMSE → higher score |
| `content_based` | `ndcg_at_k` | higher is better (used directly) |
| `hybrid_xgboost` | `auc_roc` | higher is better (used directly) |

3. Transitions the winner to `Production` via `MlflowClient.transition_model_version_stage()`
4. Writes `models/best_model.json` with the model name, registry name, version, and score

---

## Flask API

**File:** `src/api/app.py` · **Port:** 5050 · **Served by:** Gunicorn (2 workers) · **CORS:** enabled

`engine.py` pre-loads the Production XGBoost model at Flask startup using a thread-safe singleton (`threading.Lock` + module-level `_model_cache` dict). The model is fetched from HDFS via WebHDFS REST and deserialised with `joblib`.

### Endpoints

#### `GET /health`

Liveness check. Also pings ClickHouse with `SELECT 1` to confirm database connectivity.

```json
{
  "status": "ok",
  "service": "healthcare-recommendation-api",
  "clickhouse": "ok"
}
```

#### `GET /model/info`

Returns the active Production model version, read from `models/best_model.json`.

```json
{
  "status": "ok",
  "best_model": "hybrid_xgboost",
  "registry_name": "healthcare-hybrid-xgboost",
  "version": "1",
  "score": 0.9684
}
```

#### `POST /recommend`

Generate top-K medication recommendations for a patient.

```json
// Request
{ "patient_id": "abc123-...", "top_k": 10 }

// Response
{
  "status": "ok",
  "patient_id": "abc123-...",
  "recommendations": [
    { "rank": 1, "treatment_code": "316049", "treatment_name": "Metformin 500 MG", "score": 0.923, "explanation": "..." }
  ]
}
```

#### `GET /recommend/<patient_id>`

Fetch previously stored recommendations from `healthcare.recommendations`.

#### `GET /patients`

List all patients with a `patient_features` row (processed through the feature pipeline).

#### `GET /patients/<patient_id>`

Return a patient's full feature summary — demographics, chronic disease flags, condition and medication counts.

#### `POST /model/reload`

Hot-reload the Production model from HDFS without restarting the container. Acquires `_model_lock` before swapping the in-memory cache.

---

## Streamlit Dashboard

**File:** `src/dashboard/app.py` · **Port:** 8501

Uses `@st.cache_resource` for the ClickHouse client and `@st.cache_data(ttl=300)` for population-level queries to avoid recomputing statistics on every render.

| Page | What it shows |
|---|---|
| **Patient Lookup** | Search by patient ID; view demographics, chronic disease flags, condition and medication history |
| **Recommendations** | Enter a patient ID and top-K count; calls `POST /recommend`; displays ranked medication table with scores |
| **Population Analytics** | Condition prevalence bar chart, chronic disease counts, age distribution histogram |
| **Model Performance** | Live metrics from MLflow; comparison table across all three registered models |
| **Data Explorer** | Free-form ClickHouse query UI; results rendered as a paginated DataFrame |

---

## Airflow DAGs

All DAGs set `schedule_interval=None` (manual trigger only) and `catchup=False`. They are chained via `TriggerDagRunOperator` on the final task of each DAG.

| DAG | Tags | Key tasks |
|---|---|---|
| `dag_ingest_to_hdfs` | milestone-1, ingestion, hdfs | `validate_csvs` → `ensure_hdfs_dirs` → `upload_to_hdfs` → trigger next |
| `dag_ingest_clickhouse` | milestone-1, ingestion, clickhouse | `truncate_tables` → `load_tables` → `verify_counts` → trigger next |
| `dag_spark_processing` | milestone-2, spark | `submit_clean` → `submit_features` → `load_features_to_clickhouse` → `verify_features` → trigger next |
| `dag_train_models` | milestone-3, training | `train_collaborative_filtering` → `train_content_based` → `train_hybrid_xgboost` → trigger next |
| `dag_evaluate_and_register` | milestone-3, evaluation, mlflow | `evaluate_and_register` → `write_best_model_json` → trigger next |
| `dag_deploy_and_serve` | milestone-4, deploy | `health_check_api` → `reload_model` → `health_check_dashboard` |

**Default retry policy:** `retries=2, retry_delay=timedelta(minutes=5)`.

**Airflow → ClickHouse:** DAGs connect directly using `clickhouse-connect` (installed via `_PIP_ADDITIONAL_REQUIREMENTS` in the Airflow image). Credentials come from the `x-airflow-common` env block in `docker-compose.yml`.

**Airflow → Spark:** model training DAGs submit jobs via the Spark REST API (`http://spark-master:6066/v1/submissions`), using `mainClass: org.apache.spark.deploy.PythonRunner`. `evaluate.py` is run as a `PythonOperator` directly in Airflow (it has no Spark dependency).

---

## Makefile Reference

### Infrastructure

| Command | Description |
|---|---|
| `make up` | Build images and start all 14 services in detached mode |
| `make down` | Stop and remove containers (volumes preserved) |
| `make restart` | `down` then `up` |
| `make logs` | Stream logs from all containers (`--tail=100`) |
| `make ps` | Show container status |
| `make verify` | Health-check ClickHouse, Airflow, MLflow, Spark, HDFS |
| `make fix-airflow` | Fix log directory permissions and restart Airflow services |
| `make clean` | `down -v --remove-orphans` and wipe generated data, logs, and models |

### Shell access

| Command | Description |
|---|---|
| `make shell-namenode` | `bash` inside the HDFS NameNode container |
| `make shell-clickhouse` | ClickHouse client as `healthcare_user` |
| `make shell-airflow` | `bash` inside `airflow-webserver` |
| `make shell-spark` | `bash` inside `spark-master` |

### Data

| Command | Description |
|---|---|
| `make generate-data` | Run Synthea (requires Java 11 on the host) |
| `make init-hdfs` | Create `/data/raw`, `/data/processed`, `/data/features` in HDFS |
| `make load-hdfs` | Upload `data/raw/*.csv` to HDFS |
| `make clean-hdfs` | Remove raw files from HDFS |
| `make init-clickhouse` | Execute `01_create_schema.sql` inside ClickHouse |
| `make load-clickhouse` | Truncate all tables then run `load_clickhouse.sh` |
| `make clean-clickhouse` | Truncate all ClickHouse tables |
| `make create-airflow-user` | Create `admin` user in Airflow (idempotent) |
| `make init-everything` | `init-hdfs` + `init-clickhouse` + `create-airflow-user` |

### Pipeline triggers

| Command | Description |
|---|---|
| `make pipeline` | Trigger `dag_ingest_to_hdfs` (chains all subsequent DAGs automatically) |
| `make pipeline-m2` | Trigger `dag_spark_processing` directly |
| `make pipeline-m3` | Trigger `dag_train_models` directly |

### Manual model runs (on spark-master)

| Command | Description |
|---|---|
| `make run-clean` | `spark-submit clean.py` |
| `make run-features` | `spark-submit feature_engineering.py` |
| `make run-als` | Train ALS collaborative filtering |
| `make run-content-based` | Train content-based TF-IDF model |
| `make run-hybrid` | Train XGBoost hybrid model |
| `make run-evaluate` | Evaluate all models and promote best to Production |
| `make train-all` | Run all four steps above sequentially |

### Development

| Command | Description |
|---|---|
| `make notebook` | Launch JupyterLab inside the `app` container on port 8888 |
| `make explore` | Open `notebooks/01_data_exploration.ipynb` locally |

---

## Service URLs & Ports

| Service | Host port | URL |
|---|---|---|
| Streamlit Dashboard | 8501 | http://localhost:8501 |
| Flask REST API | 5050 | http://localhost:5050 |
| Airflow Webserver | 8081 | http://localhost:8081 |
| MLflow Tracking | 5001 | http://localhost:5001 |
| Spark Master UI | 8080 | http://localhost:8080 |
| Spark Worker UI | 8082 | http://localhost:8082 |
| JupyterLab | 8888 | http://localhost:8888 |
| HDFS NameNode UI | 9870 | http://localhost:9870 |
| YARN ResourceManager | 8088 | http://localhost:8088 |
| YARN History Server | 8188 | http://localhost:8188 |
| ClickHouse HTTP | 8123 | http://localhost:8123 |
| ClickHouse native TCP | 9900 | `make shell-clickhouse` |
| PostgreSQL | 5432 | (psql / pgAdmin) |
| pgAdmin | 5055 | http://localhost:5055 |

---

## Credentials

> **Development only.** Rotate all secrets before any non-local deployment.

| Service | Username | Password |
|---|---|---|
| Airflow | `admin` | `admin_secret_2026` |
| ClickHouse | `healthcare_user` | `ch_secret_2026` |
| PostgreSQL | `airflow` | `airflow_secret_2026` |
| pgAdmin | `admin@healthcare.com` | `pgadmin_secret_2026` |

---

## Environment Variables

All variables are read from `.env` and injected via `env_file` in `docker-compose.yml`. The values below are the defaults that work out of the box.

| Variable | Default | Used by |
|---|---|---|
| `CLICKHOUSE_HOST` | `clickhouse` | All services |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP connections |
| `CLICKHOUSE_DB` | `healthcare` | All services |
| `CLICKHOUSE_USER` | `healthcare_user` | All services |
| `CLICKHOUSE_PASSWORD` | `ch_secret_2026` | All services |
| `HDFS_NAMENODE_HOST` | `namenode` | Airflow DAGs, Spark jobs |
| `HDFS_NAMENODE_PORT` | `9001` | HDFS RPC connections |
| `SPARK_MASTER` | `spark://spark-master:7077` | All `spark-submit` calls |
| `SPARK_EXECUTOR_MEMORY` | `2g` | Spark submit executor memory |
| `SPARK_DRIVER_MEMORY` | `1g` | Spark submit driver memory |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5001` | All model training scripts |
| `DATA_DIR` | `/opt/airflow/data/raw` | Ingestion scripts |
| `MODELS_DIR` | `/opt/airflow/models` | Model scripts |
| `AIRFLOW__CORE__EXECUTOR` | `LocalExecutor` | Airflow |
| `AIRFLOW__CORE__DEFAULT_TIMEZONE` | `Africa/Cairo` | Airflow |
| `_AIRFLOW_WWW_USER_USERNAME` | `admin` | Airflow init container |
| `_AIRFLOW_WWW_USER_PASSWORD` | `admin_secret_2026` | Airflow init container |
| `_PIP_ADDITIONAL_REQUIREMENTS` | `clickhouse-connect>=0.7.0 pandas>=2.0.0 hdfs` | Airflow worker pip install |

> `.env` is listed in `.gitignore` and must never be committed to version control.

---

## Design Decisions

**ClickHouse instead of Hive** eliminates four containers (`hive-server`, `hive-metastore`, `hive-metastore-postgresql`, `hive-metastore-db`) in favour of one. ClickHouse's columnar MergeTree storage is faster for the OLAP queries this system runs: patient cohort filtering, feature aggregation, and ranked recommendation retrieval. DAGs connect via the `clickhouse-connect` Python SDK rather than `SSHOperator → beeline`.

**HDFS for Spark intermediates** — raw CSVs go into both HDFS (so Spark can read them natively in cluster mode) and ClickHouse (for SQL queries and model serving). Cleaned Parquet and feature files written by Spark land in `hdfs://namenode:9001/data/features/` and are subsequently loaded into `patient_features` for API and dashboard access.

**Single `docker-compose.yml`** defines all 14 services using YAML anchors (`x-airflow-common`, `x-hadoop-common`) to share configuration without duplication. No external cluster manager is required.

**LocalExecutor for Airflow** is sufficient for this workload and removes the complexity of a Celery broker or Redis. Spark jobs run in Spark cluster mode so Airflow itself stays lightweight.

**Thread-safe model singleton in `engine.py`** — the XGBoost model is loaded once at Flask startup using `threading.Lock` + a module-level `_model_cache` dict. The lock is released immediately after loading so it does not block concurrent inference requests.

---

## Troubleshooting

**Airflow tasks fail with log directory permission errors:**
```bash
make fix-airflow
```

**ClickHouse tables are empty after `make up`:**
```bash
make init-clickhouse && make load-clickhouse
```

**HDFS NameNode stuck in safe mode:**
```bash
docker exec namenode hdfs dfsadmin -safemode leave
```

**Spark submit times out or refuses connection:**
```bash
make ps       # confirm spark-master shows as healthy
make verify   # check all service health endpoints
```

**Out-of-memory during model training:**  
Raise `SPARK_EXECUTOR_MEMORY` and `SPARK_WORKER_MEMORY` in `.env`, then `make restart`.

**MLflow shows no registered model versions:**  
Confirm training completed (`make train-all`), then check http://localhost:5001 → Models.

**`make generate-data` fails with "java: command not found":**  
Install Java 11: `sudo apt install openjdk-11-jdk` (Ubuntu) or `brew install openjdk@11` (macOS).

---

## Team

| Member | Role | Responsibilities |
|---|---|---|
| **Amr Mohamed Hussien** | Infrastructure & Data Engineering | Docker Compose environment setup, Synthea data generation, `dag_ingest_to_hdfs`, ClickHouse schema design, `dag_ingest_clickhouse` |
| **Mahmoud Mohamed Ali** | Processing & Dashboard | `clean.py`, `feature_engineering.py`, `dag_spark_processing`, Streamlit five-page dashboard |
| **Mohamed Helmy Elsayed** | Models & API | ALS collaborative filtering, XGBoost hybrid model, `evaluate.py`, MLflow experiment tracking, Flask REST API, `engine.py` |
| **Abdulrahman Ahmed Mahmoud** | Models & Evaluation | Content-based model v3 (AUC-focused), `dag_train_models`, `dag_evaluate_and_register`, MLflow model registry and automatic promotion |

---

<div align="center">
<sub>Built with Apache Spark · ClickHouse · MLflow · Airflow · Flask · Streamlit · Hadoop HDFS</sub>
</div>