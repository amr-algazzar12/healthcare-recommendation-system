"""
dag_evaluate_and_register.py — Milestone 3

Submits evaluate.py to spark-master, which compares all three models,
picks the best, and promotes it to Production in the MLflow registry.

Task chain:
    evaluate_and_register
        → verify_production_model
            → trigger_dag_deploy_and_serve
"""

from __future__ import annotations

import json
import os
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
MLFLOW_URI       = "http://mlflow:5001"

MODEL_REGISTRY = {
    "collaborative_filtering": "healthcare-collaborative-filtering",
    "content_based":           "healthcare-content-based",
    "hybrid_xgboost":          "healthcare-hybrid-xgboost",
}


# ── Spark REST helper ─────────────────────────────────────────────────────────

def _submit_spark_job(script_path: str, app_name: str,
                      timeout_minutes: int = 30):
    payload = {
        "action":             "CreateSubmissionRequest",
        "appResource":        f"file://{script_path}",
        "clientSparkVersion": "3.5.3",
        "mainClass":          "org.apache.spark.deploy.PythonRunner",
        "appArgs":            [f"file://{script_path}", ""],
        "environmentVariables": {
            "SPARK_ENV_LOADED":     "1",
            "SPARK_MASTER":         SPARK_MASTER_URL,
            "MLFLOW_TRACKING_URI":  MLFLOW_URI,
            "CLICKHOUSE_HOST":      "clickhouse",
            "CLICKHOUSE_HTTP_PORT": "8123",
            "CLICKHOUSE_DB":        "healthcare",
            "CLICKHOUSE_USER":      "healthcare_user",
            "CLICKHOUSE_PASSWORD":  "ch_secret_2026",
        },
        "sparkProperties": {
            "spark.master":                     SPARK_MASTER_URL,
            "spark.app.name":                   app_name,
            "spark.submit.deployMode":          "cluster",
            "spark.executor.memory":            "2g",
            "spark.driver.memory":              "1g",
            "spark.network.timeout":            "300s",
            "spark.executor.heartbeatInterval": "60s",
            "spark.jars":                       "",
            "spark.files":                      "",
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
        raise RuntimeError(f"Spark REST rejected: {e.read().decode()}")

    submission_id = result.get("submissionId")
    if not submission_id:
        raise RuntimeError(f"No submission ID: {result}")
    print(f"  Submitted {app_name} → {submission_id}")

    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        time.sleep(15)
        with urllib.request.urlopen(
            urllib.request.Request(
                f"{SPARK_REST_URL}/status/{submission_id}", method="GET"
            ), timeout=30
        ) as resp:
            status = json.loads(resp.read())

        state = status.get("driverState", "UNKNOWN")
        print(f"  [{app_name}] state: {state}")

        if state == "FINISHED":
            print(f"  {app_name} — SUCCESS")
            return
        elif state in ("FAILED", "KILLED", "ERROR"):
            raise RuntimeError(f"{app_name} failed: {state}\n{json.dumps(status, indent=2)}")

    raise TimeoutError(f"{app_name} timed out after {timeout_minutes}m")


# ── Task callables ────────────────────────────────────────────────────────────

def _evaluate_and_register(**ctx):
    """Submit evaluate.py to spark-master via REST."""
    _submit_spark_job(
        script_path=f"{SRC_DIR}/evaluate.py",
        app_name="healthcare-evaluate",
        timeout_minutes=30,
    )


def _verify_production_model(**ctx):
    """
    Verify that exactly one model is in Production stage in MLflow registry.
    Reads the best_model.json written by evaluate.py and confirms
    the model is actually registered as Production.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    # Read result file written by evaluate.py
    result_path = "/opt/airflow/models/best_model.json"
    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"{result_path} not found. "
            "Did evaluate.py complete successfully?"
        )

    with open(result_path) as f:
        result = json.load(f)

    best_model    = result["best_model"]
    registry_name = result["registry_name"]
    version       = result["version"]
    score         = result["score"]

    print(f"==> Best model result from evaluate.py:")
    print(f"    Model:    {best_model}")
    print(f"    Registry: {registry_name}")
    print(f"    Version:  {version}")
    print(f"    Score:    {score:.4f}")

    # Verify via MLflow API
    import urllib.request
    resp = urllib.request.urlopen(
        f"{MLFLOW_URI}/api/2.0/mlflow/registered-models/get"
        f"?name={registry_name}",
        timeout=15,
    )
    model_info = json.loads(resp.read())
    latest_versions = model_info.get("registered_model", {}).get(
        "latest_versions", []
    )

    production_versions = [
        v for v in latest_versions if v.get("current_stage") == "Production"
    ]

    if not production_versions:
        raise ValueError(
            f"No Production version found for {registry_name} in MLflow. "
            "Check evaluate.py logs."
        )

    prod_version = production_versions[0]["version"]
    print(f"==> Verified: {registry_name} v{prod_version} is in Production ✓")

    # Push best model info to XCom for deploy DAG
    return result


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag_evaluate_and_register",
    default_args=default_args,
    description="Evaluate models, pick best, promote to Production (Milestone 3)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["milestone-3", "evaluation"],
) as dag:

    evaluate = PythonOperator(
        task_id="evaluate_and_register",
        python_callable=_evaluate_and_register,
        execution_timeout=timedelta(minutes=40),
    )

    verify = PythonOperator(
        task_id="verify_production_model",
        python_callable=_verify_production_model,
    )

    trigger_deploy = TriggerDagRunOperator(
        task_id="trigger_dag_deploy_and_serve",
        trigger_dag_id="dag_deploy_and_serve",
        wait_for_completion=False,
    )

    evaluate >> verify >> trigger_deploy