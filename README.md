# Healthcare Recommendation System — ClickHouse Edition

**Stack:** Hadoop HDFS 3.2.1 · YARN · Spark 3.5.3 · ClickHouse 24.3 · Airflow 2.8.1 · MLflow 2.10 · Flask · Streamlit  
**Data:** Synthea™ synthetic patient generator (~10,000 patients, fixed seed)  
**Status:** Milestone 1 complete ✓

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│                                                          │
│  namenode  ── datanode                                   │
│  resourcemanager ── nodemanager ── historyserver         │
│  spark-master ── spark-worker                            │
│  clickhouse                                              │
│  postgres  ── airflow-webserver ── airflow-scheduler     │
│  mlflow                                                  │
│  app (Flask + Streamlit)                                 │
│  pgadmin                                                 │
└─────────────────────────────────────────────────────────┘

Airflow pipeline (6 DAGs, chained via TriggerDagRunOperator):

dag_ingest_to_hdfs
  └─► dag_ingest_clickhouse
        └─► dag_spark_processing        [Milestone 2]
              └─► dag_train_models      [Milestone 3]
                    └─► dag_evaluate_and_register
                          └─► dag_deploy_and_serve
```

---

## Quick start

```bash
# 1. Generate synthetic patient data (requires Java 11+)
make generate-data

# 2. Start the full cluster
make up

# 3. Initialize HDFS directories
make init-hdfs

# 4. Initialize ClickHouse schema
make init_clickhouse

# 5. Load CSVs into HDFS
make load-hdfs

# 6. Load CSVs into ClickHouse
make load-clickhouse

# 7. Verify everything
make verify
```

Then open Airflow at **http://localhost:8081**, enable `dag_ingest_to_hdfs`, and trigger it. The remaining DAGs chain automatically.

---

## Port reference

| Service              | Host port | URL                        |
|----------------------|-----------|----------------------------|
| Airflow UI           | 8081      | http://localhost:8081      |
| Spark Master UI      | 8080      | http://localhost:8080      |
| Spark Worker UI      | 8082      | http://localhost:8082      |
| HDFS Namenode UI     | 9870      | http://localhost:9870      |
| YARN ResourceManager | 8088      | http://localhost:8088      |
| YARN History Server  | 8188      | http://localhost:8188      |
| ClickHouse HTTP      | 8123      | http://localhost:8123      |
| ClickHouse native    | 9900      | `make shell-clickhouse`    |
| PostgreSQL           | 5432      |                            |
| pgAdmin              | 5055      | http://localhost:5055      |
| MLflow UI            | 5001      | http://localhost:5001      |
| Flask API            | 5050      | http://localhost:5050      |
| Streamlit dashboard  | 8501      | http://localhost:8501      |

---

## Credentials (development only)

| Service    | User                   | Password             |
|------------|------------------------|----------------------|
| Airflow    | admin                  | admin_secret_2026    |
| ClickHouse | healthcare_user        | ch_secret_2026       |
| PostgreSQL | airflow                | airflow_secret_2026  |
| pgAdmin    | admin@healthcare.com   | pgadmin_secret_2026  |

---

## Project structure

```
healthcare-recommendation-system/
├── dags/
│   ├── dag_ingest_to_hdfs.py         # M1 ✓ — validate + HDFS load
│   ├── dag_ingest_clickhouse.py      # M1 ✓ — ClickHouse load
│   ├── dag_spark_processing.py       # M2   — stub
│   ├── dag_train_models.py           # M3   — stub
│   ├── dag_evaluate_and_register.py  # M3   — stub
│   └── dag_deploy_and_serve.py       # M4   — stub
│
├── src/
│   ├── ingestion/
│   │   └── validate.py               # M1 ✓ — CSV pre-flight checks
│   ├── clickhouse/
│   │   └── load_tables.py            # M1 ✓ — CSV → ClickHouse
│   ├── processing/                   # M2
│   ├── models/                       # M3
│   ├── api/                          # M4
│   ├── dashboard/                    # M4
│   └── utils/
│       ├── config.py                 # M1 ✓
│       └── clickhouse_client.py      # M1 ✓
│
├── docker/
│   ├── clickhouse/
│   │   ├── config.xml
│   │   ├── users.xml
│   │   └── init/01_create_schema.sql     
│   ├── mlflow/Dockerfile
│   └── app/Dockerfile
│
├── data/
│   ├── raw/          # Synthea CSVs (gitignored)
│   ├── processed/    # Spark output — Parquet (gitignored)
│   ├── features/     # Feature store — Parquet (gitignored)
│   └── synthea/
│       ├── generate_data.sh
│       └── synthea.properties
│
├── scripts/
│   ├── load_clickhouse.sh
│   └── postgres/init-mlflow-db.sql
│
├── conda/environment.yml
├── docker-compose.yml
├── hadoop.env
├── Makefile
└── .env
```

---

## Makefile reference

| Target               | What it does                                        |
|----------------------|-----------------------------------------------------|
| `make up`            | Build and start all containers                      |
| `make down`          | Stop all containers                                 |
| `make ps`            | Show container status                               |
| `make logs`          | Tail all logs                                       |
| `make verify`        | Health-check all services                           |
| `make generate-data` | Run Synthea on host (Java 11+ required)             |
| `make init-hdfs`     | Create HDFS directories                             |
| `make init_clickhouse` | Run schema DDL inside ClickHouse                  |
| `make load-hdfs`     | Upload CSVs from `data/raw/` into HDFS             |
| `make load-clickhouse` | Load CSVs into ClickHouse tables                  |
| `make shell-namenode`  | Bash shell inside namenode                        |
| `make shell-clickhouse`| ClickHouse client shell                           |
| `make shell-airflow`   | Bash shell inside airflow-webserver               |
| `make shell-spark`     | Bash shell inside spark-master                    |
| `make pipeline`        | Trigger `dag_ingest_to_hdfs` in Airflow           |
| `make clean`           | Tear down volumes and wipe data/logs              |

---

## Milestone plan

| Milestone | Scope                                                    | Status      |
|-----------|----------------------------------------------------------|-------------|
| 1         | Cluster setup, data ingestion into HDFS + ClickHouse     | ✓ Complete  |
| 2         | PySpark cleaning + feature engineering → `patient_features` | In progress |
| 3         | ALS, content-based, hybrid models + MLflow tracking      | Pending     |
| 4         | Flask API + Streamlit dashboard                          | Pending     |
| 5         | Evaluation, tuning, documentation                        | Pending     |

---

## Design decisions

**ClickHouse instead of Hive** — replaces 4 containers (hive-server, hive-metastore, hive-metastore-postgresql, hive-metastore-db) with one. ClickHouse's columnar storage is faster for the OLAP queries this system runs (patient cohort filtering, feature aggregation, recommendation ranking). DAGs use `clickhouse-connect` Python SDK instead of SSHOperator → beeline.

**Custom Docker Compose** — built from scratch using official images (`bde2020/hadoop-*`, `apache/airflow`, `apache/spark`, `clickhouse/clickhouse-server`). No dependency on the mrugankray pre-built cluster.

**HDFS for Spark intermediates** — raw CSVs go into both HDFS (for Spark to read natively in cluster mode) and ClickHouse (for SQL queries and model serving). Processed Parquet and feature files written by Spark land in `hdfs://namenode:9001/data/features/` and are also volume-mounted into ClickHouse at `/mnt/features`.