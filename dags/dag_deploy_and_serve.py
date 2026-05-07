"""
dag_deploy_and_serve.py — Milestone 4

Deploys the Production model to the app container:
  1. Verifies best_model.json exists and model is in HDFS
  2. Runs make run-hybrid to save/refresh model on HDFS (if needed)
  3. Hot-reloads the model in the running Flask API via POST /model/reload
  4. Runs a smoke test against POST /recommend
  5. Verifies recommendations were persisted to ClickHouse

Task chain:
    verify_model_artifacts
        → reload_api_model
            → smoke_test
                → verify_clickhouse_recommendations
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner":            "healthcare",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "email_on_failure": False,
    "email_on_retry":   False,
}

API_BASE        = "http://app:5050"
HDFS_BASE       = "http://namenode:9870"
HDFS_MODEL_PATH = "/models/hybrid_xgboost/model.joblib"
BEST_MODEL_JSON = "/opt/airflow/models/best_model.json"
CH_DB           = "healthcare"
CH_HOST         = "clickhouse"
CH_PORT         = "8123"
CH_USER         = "healthcare_user"
CH_PASS         = "ch_secret_2026"


# ── Task 1 — verify artifacts ─────────────────────────────────────────────────

def _verify_model_artifacts(**ctx):
    """
    Check:
      1. best_model.json exists and has required keys
      2. Model file exists on HDFS via WebHDFS
    """
    # Check best_model.json
    if not os.path.exists(BEST_MODEL_JSON):
        raise FileNotFoundError(
            f"{BEST_MODEL_JSON} not found. "
            "Run dag_evaluate_and_register first."
        )

    with open(BEST_MODEL_JSON) as f:
        best = json.load(f)

    required_keys = ["best_model", "registry_name", "version", "score"]
    missing = [k for k in required_keys if k not in best]
    if missing:
        raise ValueError(f"best_model.json missing keys: {missing}")

    print(f"==> best_model.json:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # Check HDFS model file
    url = (
        f"{HDFS_BASE}/webhdfs/v1{HDFS_MODEL_PATH}"
        f"?op=GETFILESTATUS"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            status = json.loads(resp.read())
        length = status["FileStatus"]["length"]
        if length == 0:
            raise ValueError(f"HDFS model file is empty: {HDFS_MODEL_PATH}")
        print(f"==> HDFS model: {HDFS_MODEL_PATH} ({length / 1024:.1f} KB) ✓")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"Model not found on HDFS: {HDFS_MODEL_PATH}\n"
                "Run make run-hybrid to save the model to HDFS."
            )
        raise

    # Push model version to XCom for downstream tasks
    ctx["ti"].xcom_push(key="model_version", value=best["version"])
    ctx["ti"].xcom_push(key="best_model",    value=best["best_model"])
    print("==> Artifact verification passed ✓")


# ── Task 2 — hot-reload model in Flask API ────────────────────────────────────

def _reload_api_model(**ctx):
    """
    POST /model/reload to the running Flask API.
    This triggers the engine to re-download and cache the model from HDFS
    without restarting the container.
    """
    url = f"{API_BASE}/model/reload"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Retry up to 5 times — API may take a moment to download from HDFS
    last_error = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            if result.get("status") == "ok":
                version = result.get("model_version", "?")
                print(f"==> Model reloaded successfully (v{version}) ✓")
                ctx["ti"].xcom_push(key="reloaded_version", value=version)
                return
            else:
                raise RuntimeError(f"Reload returned non-ok: {result}")

        except Exception as e:
            last_error = e
            print(f"    Attempt {attempt}/5 failed: {e} — retrying in 10s...")
            time.sleep(10)

    raise RuntimeError(
        f"Model reload failed after 5 attempts: {last_error}"
    )


# ── Task 3 — smoke test ───────────────────────────────────────────────────────

def _smoke_test(**ctx):
    """
    1. Hit GET /health — must return 200
    2. Pick a real patient from ClickHouse
    3. POST /recommend for that patient
    4. Validate response structure and score values
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from utils.clickhouse_client import get_client

    # Health check
    with urllib.request.urlopen(f"{API_BASE}/health", timeout=15) as resp:
        health = json.loads(resp.read())
    assert health.get("status") == "ok", f"Health check failed: {health}"
    print(f"==> /health: OK")

    # Pick a real patient
    client = get_client()
    result = client.query(
        f"SELECT patient_id FROM {CH_DB}.patient_features LIMIT 1"
    )
    if not result.result_rows:
        raise ValueError("No patients in patient_features table")

    test_patient_id = result.result_rows[0][0]
    print(f"==> Smoke test patient: {test_patient_id}")

    # Run recommendation
    payload = json.dumps({
        "patient_id": test_patient_id,
        "top_k": 5,
    }).encode()

    req = urllib.request.Request(
        f"{API_BASE}/recommend",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        rec_result = json.loads(resp.read())

    # Validate
    assert rec_result.get("status") == "ok", \
        f"Recommend failed: {rec_result}"

    recs = rec_result.get("recommendations", [])
    assert len(recs) > 0, "No recommendations returned"
    assert len(recs) <= 5, f"Too many recommendations: {len(recs)}"

    for rec in recs:
        assert "treatment_code" in rec, "Missing treatment_code"
        assert "treatment_name" in rec, "Missing treatment_name"
        assert "score" in rec, "Missing score"
        assert "rank" in rec, "Missing rank"
        assert "explanation" in rec, "Missing explanation"
        assert 0.0 <= rec["score"] <= 1.0, \
            f"Score out of range: {rec['score']}"

    print(f"==> /recommend: {len(recs)} recommendations returned ✓")
    print(f"    Top recommendation: {recs[0]['treatment_name']} "
          f"(score={recs[0]['score']:.4f})")

    ctx["ti"].xcom_push(key="smoke_test_patient", value=test_patient_id)
    ctx["ti"].xcom_push(key="smoke_test_recs",    value=len(recs))


# ── Task 4 — verify ClickHouse persistence ────────────────────────────────────

def _verify_clickhouse_recommendations(**ctx):
    """
    Confirm that the smoke test recommendations were persisted
    to the ClickHouse recommendations table.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from utils.clickhouse_client import get_client

    test_pid = ctx["ti"].xcom_pull(
        task_ids="smoke_test", key="smoke_test_patient"
    )

    if not test_pid:
        raise ValueError("smoke_test_patient not found in XCom")

    client = get_client()
    count  = client.query(f"""
        SELECT count()
        FROM {CH_DB}.recommendations
        WHERE patient_id = '{test_pid}'
    """).result_rows[0][0]

    if count == 0:
        raise ValueError(
            f"No recommendations found in ClickHouse for patient {test_pid}. "
            "Persistence may have failed — check engine.py logs."
        )

    print(f"==> ClickHouse recommendations: {count} rows for patient "
          f"{test_pid[:8]}... ✓")

    # Total recommendations across all patients
    total = client.query(
        f"SELECT count() FROM {CH_DB}.recommendations"
    ).result_rows[0][0]
    print(f"==> Total recommendations in table: {total:,}")

    print("\n==> Milestone 4 deployment complete ✓")
    print(f"    Flask API:       {API_BASE}")
    print(f"    Streamlit UI:    http://app:8501")
    print(f"    MLflow:          http://mlflow:5001")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="dag_deploy_and_serve",
    default_args=default_args,
    description="Deploy Production model to Flask API + smoke test (Milestone 4)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["milestone-4", "deploy"],
) as dag:

    verify_artifacts = PythonOperator(
        task_id="verify_model_artifacts",
        python_callable=_verify_model_artifacts,
    )

    reload_model = PythonOperator(
        task_id="reload_api_model",
        python_callable=_reload_api_model,
        execution_timeout=timedelta(minutes=10),
    )

    smoke_test = PythonOperator(
        task_id="smoke_test",
        python_callable=_smoke_test,
        execution_timeout=timedelta(minutes=5),
    )

    verify_ch = PythonOperator(
        task_id="verify_clickhouse_recommendations",
        python_callable=_verify_clickhouse_recommendations,
    )

    verify_artifacts >> reload_model >> smoke_test >> verify_ch
