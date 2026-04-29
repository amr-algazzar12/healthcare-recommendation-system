"""
dag_train_models.py — Milestone 3

Submits all three model training scripts to spark-master via REST API,
sequentially. Each script reads from ClickHouse, trains, and logs to MLflow.

Task chain:
    train_collaborative_filtering
        → train_content_based
            → train_hybrid_xgboost
                → trigger_dag_evaluate_and_register
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

SPARK_REST_URL   = "http://spark-master:6066/v1/submissions"
SPARK_MASTER_URL = "spark://spark-master:7077"
SRC_DIR          = "/opt/airflow/src/models"


# ── Spark REST helper (same pattern as dag_spark_processing) ──────────────────

def _submit_spark_job(script_path: str, app_name: str,
                      timeout_minutes: int = 90):
    payload = {
        "action":            "CreateSubmissionRequest",
        "appResource":       f"file://{script_path}",
        "clientSparkVersion": "3.5.3",
        "mainClass":         "org.apache.spark.deploy.PythonRunner",
        "appArgs":           [f"file://{script_path}", ""],
        "environmentVariables": {
            "SPARK_ENV_LOADED":        "1",
            "SPARK_MASTER":            SPARK_MASTER_URL,
            "MLFLOW_TRACKING_URI":     "http://mlflow:5001",
            "CLICKHOUSE_HOST":         "clickhouse",
            "CLICKHOUSE_HTTP_PORT":    "8123",
            "CLICKHOUSE_DB":           "healthcare",
            "CLICKHOUSE_USER":         "healthcare_user",
            "CLICKHOUSE_PASSWORD":     "ch_secret_2026",
        },
        "sparkProperties": {
            "spark.master":                    SPARK_MASTER_URL,
            "spark.app.name":                  app_name,
            "spark.submit.deployMode":         "cluster",
            "spark.executor.memory":           "2g",
            "spark.driver.memory":             "2g",
            "spark.sql.shuffle.partitions":    "8",
            "spark.network.timeout":           "300s",
            "spark.executor.heartbeatInterval": "60s",
            "spark.jars":                      "",
            "spark.files":                     "",
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{SPARK_REST_URL}/create",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8")
        raise RuntimeError(f"Spark REST rejected request: {msg}")

    submission_id = result.get("submissionId")
    if not submission_id:
        raise RuntimeError(f"No submission ID returned: {result}")
    print(f"  Submitted {app_name} → {submission_id}")

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        time.sleep(20)
        status_req = urllib.request.Request(
            f"{SPARK_REST_URL}/status/{submission_id}", method="GET"
        )
        with urllib.request.urlopen(status_req, timeout=30) as resp:
            status = json.loads(resp.read())

        state = status.get("driverState", "UNKNOWN")
        print(f"  [{app_name}] state: {state}")

        if state == "FINISHED":
            print(f"  {app_name} — SUCCESS")
            return
        elif state in ("FAILED", "KILLED", "ERROR"):
            raise RuntimeError(
                f"{app_name} failed: {state}\n"
                f"{json.dumps(status, indent=2)}"
            )

    raise TimeoutError(f"{app_name} timed out after {timeout_minutes}m")


# ── Task callables ────────────────────────────────────────────────────────────

def _train_collaborative_filtering(**ctx):
    _submit_spark_job(
        script_path=f"{SRC_DIR}/collaborative_filtering.py",
        app_name="healthcare-collaborative-filtering",
        timeout_minutes=90,
    )


def _train_content_based(**ctx):
    _submit_spark_job(
        script_path=f"{SRC_DIR}/content_based.py",
        app_name="healthcare-content-based",
        timeout_minutes=60,
    )


def _train_hybrid_xgboost(**ctx):
    _submit_spark_job(
        script_path=f"{SRC_DIR}/hybrid_model.py",
        app_name="healthcare-hybrid-xgboost",
        timeout_minutes=90,
    )


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag_train_models",
    default_args=default_args,
    description="Train ALS, content-based, and XGBoost models (Milestone 3)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["milestone-3", "training"],
) as dag:

    train_als = PythonOperator(
        task_id="train_collaborative_filtering",
        python_callable=_train_collaborative_filtering,
        execution_timeout=timedelta(minutes=100),
    )

    train_cb = PythonOperator(
        task_id="train_content_based",
        python_callable=_train_content_based,
        execution_timeout=timedelta(minutes=70),
    )

    train_xgb = PythonOperator(
        task_id="train_hybrid_xgboost",
        python_callable=_train_hybrid_xgboost,
        execution_timeout=timedelta(minutes=100),
    )

    trigger_evaluate = TriggerDagRunOperator(
        task_id="trigger_dag_evaluate_and_register",
        trigger_dag_id="dag_evaluate_and_register",
        wait_for_completion=False,
    )

    train_als >> train_cb >> train_xgb >> trigger_evaluate