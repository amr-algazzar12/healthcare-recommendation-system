"""
dag_spark_processing.py — Milestone 2
Runs the PySpark cleaning and feature engineering jobs via spark-submit,
then loads the resulting Parquet into ClickHouse patient_features table.

Task chain:
    clean_data
        → feature_engineering
            → load_features_to_clickhouse
                → verify_features
                    → trigger_dag_train_models (commented until M3)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # M3

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ── Paths (all inside the Airflow container — volume-mounted from host) ────────
SRC_DIR   = "/opt/airflow/src/processing"
SPARK_MASTER = "spark://spark-master:7077"

# ClickHouse connection (matches docker-compose.yml env)
CH_HOST  = "clickhouse"
CH_PORT  = "8123"
CH_DB    = "healthcare"
CH_USER  = "healthcare_user"
CH_PASS  = "ch_secret_2026"

# HDFS features path (also bind-mounted into ClickHouse at /mnt/features)
HDFS_FEATURES = "hdfs://namenode:9001/data/features/patient_features"

# Host path where Parquet lands (bind-mounted ./data/features → /mnt/features in ClickHouse)
LOCAL_FEATURES = "/opt/airflow/data/features/patient_features"

with DAG(
    dag_id="dag_spark_processing",
    default_args=default_args,
    description="PySpark cleaning + feature engineering (Milestone 2)",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-2", "spark"],
) as dag:

    # ── Task 1 — data cleaning ────────────────────────────────────────────────
    clean_data = BashOperator(
        task_id="clean_data",
        bash_command=f"""
            set -e
            echo "==> Submitting clean.py to Spark..."
            /opt/spark/bin/spark-submit \
                --master {SPARK_MASTER} \
                --deploy-mode client \
                --executor-memory 2g \
                --driver-memory 1g \
                --conf spark.sql.shuffle.partitions=8 \
                --conf spark.network.timeout=300s \
                --conf spark.executor.heartbeatInterval=60s \
                {SRC_DIR}/clean.py
            echo "==> clean.py completed"
        """,
        execution_timeout=timedelta(minutes=60),
    )

    # ── Task 2 — feature engineering ─────────────────────────────────────────
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"""
            set -e
            echo "==> Submitting feature_engineering.py to Spark..."
            /opt/spark/bin/spark-submit \
                --master {SPARK_MASTER} \
                --deploy-mode client \
                --executor-memory 2g \
                --driver-memory 1g \
                --conf spark.sql.shuffle.partitions=8 \
                --conf spark.network.timeout=300s \
                --conf spark.executor.heartbeatInterval=60s \
                {SRC_DIR}/feature_engineering.py
            echo "==> feature_engineering.py completed"
        """,
        execution_timeout=timedelta(minutes=60),
    )

    # ── Task 3 — load features into ClickHouse ────────────────────────────────
    # Spark writes Parquet to hdfs://namenode:9001/data/features/patient_features
    # which is also bind-mounted on the host at ./data/features/
    # ClickHouse has ./data/features mounted at /mnt/features (read-only)
    # We use a PythonOperator with clickhouse-connect to INSERT from local Parquet
    def _load_features_to_clickhouse(**ctx):
        import sys
        import os
        import glob

        src_root = "/opt/airflow/src"
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        import pandas as pd
        from utils.clickhouse_client import get_client

        client = get_client()

        # Truncate first — idempotent re-runs
        client.command(f"TRUNCATE TABLE IF EXISTS {CH_DB}.patient_features")
        print("==> Truncated healthcare.patient_features")

        # Read Parquet files from the local volume-mount path
        parquet_dir = LOCAL_FEATURES
        if not os.path.exists(parquet_dir):
            raise FileNotFoundError(
                f"Features Parquet not found at {parquet_dir}. "
                "Did feature_engineering.py complete successfully?"
            )

        parquet_files = glob.glob(f"{parquet_dir}/*.parquet")
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

        print(f"==> Loading {len(parquet_files)} Parquet file(s) into ClickHouse...")
        total = 0
        for pf in sorted(parquet_files):
            df = pd.read_parquet(pf)
            # Keep only columns present in the ClickHouse schema
            CH_COLS = [
                "patient_id", "age", "gender_encoded", "race_encoded",
                "num_conditions", "num_medications", "num_encounters",
                "has_diabetes", "has_hypertension", "has_asthma",
                "has_hyperlipidemia", "has_coronary_disease",
                "condition_vector", "medication_history_flags",
                "feature_version",
            ]
            keep = [c for c in CH_COLS if c in df.columns]
            df = df[keep]
            client.insert_df(f"{CH_DB}.patient_features", df)
            total += len(df)
            print(f"    {os.path.basename(pf)}: {len(df):,} rows")

        print(f"==> Total rows inserted: {total:,}")

    load_features = PythonOperator(
        task_id="load_features_to_clickhouse",
        python_callable=_load_features_to_clickhouse,
    )

    # ── Task 4 — verify features in ClickHouse ────────────────────────────────
    def _verify_features(**ctx):
        import sys

        src_root = "/opt/airflow/src"
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        from utils.clickhouse_client import get_client

        client = get_client()

        count = client.query(
            f"SELECT count() FROM {CH_DB}.patient_features"
        ).result_rows[0][0]

        if count == 0:
            raise ValueError("patient_features table is empty after load.")

        # Spot-check: verify key columns are non-null
        nulls = client.query(f"""
            SELECT
                countIf(patient_id = '') AS empty_ids,
                countIf(age = 0)         AS zero_age,
                countIf(isNull(age))     AS null_age
            FROM {CH_DB}.patient_features
        """).result_rows[0]

        print(f"  patient_features row count: {count:,}")
        print(f"  Empty patient_ids: {nulls[0]}")
        print(f"  Zero age:          {nulls[1]}")
        print(f"  Null age:          {nulls[2]}")

        if nulls[0] > 0:
            raise ValueError(f"{nulls[0]} rows have empty patient_id")

        print("==> patient_features verified OK")

    verify_features = PythonOperator(
        task_id="verify_features",
        python_callable=_verify_features,
    )

    # ── Task 5 — trigger M3 (commented until Milestone 3 is ready) ────────────
    # trigger_train = TriggerDagRunOperator(
    #     task_id="trigger_dag_train_models",
    #     trigger_dag_id="dag_train_models",
    #     wait_for_completion=False,
    # )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    clean_data >> feature_engineering >> load_features >> verify_features
    # verify_features >> trigger_train