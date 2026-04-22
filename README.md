# Healthcare Recommendation System — ClickHouse Edition

**Stack:** Hadoop HDFS · YARN · Spark 3.5 · ClickHouse 24.3 · Airflow 2.8 · MLflow · Flask · Streamlit

## Quick start

```bash
# 1. Generate synthetic patient data (requires Java 11+)
make generate-data

# 2. Start the full cluster
make up

# 3. Initialize HDFS directories
make init-hdfs

# 4. Load CSVs into HDFS
make load-hdfs

# 5. Load CSVs into ClickHouse
make load-clickhouse

# 6. Verify everything is healthy
make verify
```

## Port reference

| Service              | Host port | URL                       |
|----------------------|-----------|---------------------------|
| Airflow UI           | 8081      | http://localhost:8081     |
| Spark Master UI      | 8080      | http://localhost:8080     |
| HDFS Namenode UI     | 9870      | http://localhost:9870     |
| YARN ResourceManager | 8088      | http://localhost:8088     |
| ClickHouse HTTP      | 8123      | http://localhost:8123     |
| ClickHouse native    | 9900      | clickhouse-client :9900   |
| PostgreSQL           | 5432      |                           |
| pgAdmin              | 5055      | http://localhost:5055     |
| MLflow UI            | 5001      | http://localhost:5001     |
| Flask API            | 5050      | http://localhost:5050     |
| Streamlit dashboard  | 8501      | http://localhost:8501     |

## Credentials (development only — see .env)

| Service      | User                    | Password           |
|--------------|-------------------------|--------------------|
| Airflow      | admin                   | admin_secret_2026  |
| ClickHouse   | healthcare_user         | ch_secret_2026     |
| PostgreSQL   | airflow                 | airflow_secret_2026|
| pgAdmin      | admin@healthcare.local  | pgadmin_secret_2026|

## Architecture changes from v4 (Hive → ClickHouse)

- **ClickHouse** replaces Hive + hive-server + hive-metastore + hive-metastore-postgresql (4 containers → 1)
- **dag_ingest_clickhouse** replaces dag_create_hive_tables (PythonOperator + clickhouse-connect instead of SSHOperator → beeline)
- **HDFS** is still used for Spark intermediates (clean + feature Parquet)
- **ClickHouse** reads feature Parquet from /mnt/features for serving queries
- Entire cluster is defined from scratch in docker-compose.yml — no dependency on mrugankray images
