"""
dag_spark_processing.py — Milestone 2

Submits PySpark jobs to spark-master via the Spark REST submission API
(port 6066). No docker socket, no SSH, no spark-submit binary in Airflow.

Task chain:
    clean_data
        → feature_engineering
            → load_features_to_clickhouse
                → verify_features
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
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

# ── Constants ─────────────────────────────────────────────────────────────────
SPARK_REST_URL   = "http://spark-master:6066/v1/submissions"
SPARK_MASTER_URL = "spark://spark-master:7077"
SRC_DIR          = "/opt/airflow/src/processing"   # mounted into spark-master
CH_HOST  = "clickhouse"
CH_PORT  = "8123"
CH_DB    = "healthcare"
CH_USER  = "healthcare_user"
CH_PASS  = "ch_secret_2026"
LOCAL_FEATURES   = "/opt/airflow/data/features/patient_features"


# ── Spark REST submission helper ──────────────────────────────────────────────

def _submit_spark_job(script_path: str, app_name: str, timeout_minutes: int = 60):
    """
    Submit a PySpark script to spark-master via the REST API and poll
    until it completes or times out. Raises on failure.
    """
    import time
    import urllib.request
    import urllib.error

    payload = {
        "action": "CreateSubmissionRequest",
        "appResource": script_path,
        "clientSparkVersion": "3.5.3",
        "mainClass": "org.apache.spark.deploy.SparkSubmit",
        "environmentVariables": {
            "SPARK_ENV_LOADED": "1",
            "SPARK_MASTER": SPARK_MASTER_URL,
        },
        "sparkProperties": {
            "spark.master":                    SPARK_MASTER_URL,
            "spark.app.name":                  app_name,
            "spark.submit.deployMode":         "cluster",
            "spark.executor.memory":           "2g",
            "spark.driver.memory":             "1g",
            "spark.sql.shuffle.partitions":    "8",
            "spark.network.timeout":           "300s",
            "spark.executor.heartbeatInterval":"60s",
            "spark.jars.packages":             "",
        },
    }

    # Submit
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{SPARK_REST_URL}/create",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    submission_id = result.get("submissionId")
    if not submission_id:
        raise RuntimeError(f"Spark submission failed: {result}")
    print(f"  Submitted {app_name} — submission ID: {submission_id}")

    # Poll for completion
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        time.sleep(15)
        status_req = urllib.request.Request(
            f"{SPARK_REST_URL}/status/{submission_id}",
            method="GET",
        )
        with urllib.request.urlopen(status_req, timeout=30) as resp:
            status = json.loads(resp.read())

        driver_state = status.get("driverState", "UNKNOWN")
        print(f"  [{app_name}] driver state: {driver_state}")

        if driver_state == "FINISHED":
            print(f"  {app_name} completed successfully.")
            return
        elif driver_state in ("FAILED", "KILLED", "ERROR"):
            raise RuntimeError(
                f"{app_name} failed with state: {driver_state}\n"
                f"Full status: {json.dumps(status, indent=2)}"
            )
        # RUNNING / SUBMITTED / QUEUED — keep polling

    raise TimeoutError(
        f"{app_name} did not complete within {timeout_minutes} minutes. "
        f"Last state: {driver_state}"
    )


# ── Task callables ────────────────────────────────────────────────────────────

def _clean_data(**ctx):
    _submit_spark_job(
        script_path=f"{SRC_DIR}/clean.py",
        app_name="healthcare-clean",
        timeout_minutes=60,
    )


def _feature_engineering(**ctx):
    _submit_spark_job(
        script_path=f"{SRC_DIR}/feature_engineering.py",
        app_name="healthcare-feature-engineering",
        timeout_minutes=60,
    )


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
    client.command(f"TRUNCATE TABLE IF EXISTS {CH_DB}.patient_features")
    print("==> Truncated healthcare.patient_features")

    parquet_dir = LOCAL_FEATURES
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(
            f"Features Parquet not found at {parquet_dir}. "
            "Did feature_engineering complete successfully?"
        )

    parquet_files = glob.glob(f"{parquet_dir}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    CH_COLS = [
        "patient_id", "age", "gender_encoded", "race_encoded",
        "num_conditions", "num_medications", "num_encounters",
        "has_diabetes", "has_hypertension", "has_asthma",
        "has_hyperlipidemia", "has_coronary_disease",
        "condition_vector", "medication_history_flags",
        "feature_version",
    ]

    total = 0
    for pf in sorted(parquet_files):
        df = pd.read_parquet(pf)
        keep = [c for c in CH_COLS if c in df.columns]
        client.insert_df(f"{CH_DB}.patient_features", df[keep])
        total += len(df)
        print(f"  {os.path.basename(pf)}: {len(df):,} rows")

    print(f"==> Total inserted: {total:,}")


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
        raise ValueError("patient_features is empty after load.")

    print(f"==> patient_features: {count:,} rows — OK")


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag_spark_processing",
    default_args=default_args,
    description="PySpark cleaning + feature engineering (Milestone 2)",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-2", "spark"],
) as dag:

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_clean_data,
        execution_timeout=timedelta(minutes=70),
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_feature_engineering,
        execution_timeout=timedelta(minutes=70),
    )

    load_features = PythonOperator(
        task_id="load_features_to_clickhouse",
        python_callable=_load_features_to_clickhouse,
    )

    verify_features = PythonOperator(
        task_id="verify_features",
        python_callable=_verify_features,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    clean_data >> feature_engineering >> load_features >> verify_features
    # verify_features >> trigger_train  # uncomment for M3