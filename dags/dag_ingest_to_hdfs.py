"""
DAG 1: dag_ingest_to_hdfs
Validates Synthea CSVs then loads them into HDFS.

HDFS strategy mirrors the Makefile load-hdfs target exactly:
  docker cp  →  hdfs dfs -moveFromLocal  →  rm temp

Triggers dag_ingest_clickhouse on success.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# ── Constants ─────────────────────────────────────────────────────────────────
# These paths are from the Airflow container's perspective (volume-mounted).
RAW_DATA_DIR = "/opt/airflow/data/raw"
HDFS_RAW = "hdfs://namenode:9001/data/raw"

EXPECTED_TABLES = [
    "patients",
    "conditions",
    "medications",
    "observations",
    "encounters",
    "procedures",
]

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── Validation callable ───────────────────────────────────────────────────────

def validate_csvs():
    """
    Check every expected CSV exists and has at least one data row.
    Raises on the first missing or empty file so the DAG fails fast
    before touching HDFS.
    """
    import os
    import sys

    sys.path.insert(0, "/opt/airflow/src")
    from ingestion.validate import validate_all

    validate_all(raw_dir=RAW_DATA_DIR, tables=EXPECTED_TABLES)


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag_ingest_to_hdfs",
    default_args=default_args,
    description="Validate Synthea CSVs and load them into HDFS",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-1", "ingestion", "hdfs"],
) as dag:

    # Task 1 — validate CSVs exist and are non-empty
    validate = PythonOperator(
        task_id="validate_csvs",
        python_callable=validate_csvs,
    )

    # Task 2 — ensure HDFS directories exist
    ensure_hdfs_dirs = BashOperator(
        task_id="ensure_hdfs_dirs",
        bash_command="""
            curl -sf -X PUT "http://namenode:9870/webhdfs/v1/data/raw?op=MKDIRS&permission=777" && \
            curl -sf -X PUT "http://namenode:9870/webhdfs/v1/data/processed?op=MKDIRS&permission=777" && \
            curl -sf -X PUT "http://namenode:9870/webhdfs/v1/data/features?op=MKDIRS&permission=777" && \
            echo "HDFS directories ready"
        """,
    )

    # Task 3 — load CSVs into HDFS
    # Mirrors Makefile load-hdfs exactly:
    #   docker cp is not available inside the container, so we use
    #   hdfs dfs -put from the volume-mounted path instead.
    #   The Makefile uses docker cp + moveFromLocal from the host;
    #   from inside Airflow we use -put -f directly (same result,
    #   no temp copy needed because the volume is already mounted).
    def _put_to_hdfs():
        import os, requests, logging

        namenode = "http://namenode:9870/webhdfs/v1"
        raw_dir  = RAW_DATA_DIR

        # Resolve Synthea subdirectory if present
        if os.path.isdir(os.path.join(raw_dir, "csv")):
            raw_dir = os.path.join(raw_dir, "csv")

        logging.info(f"==> Uploading CSVs from {raw_dir}")

        for table in EXPECTED_TABLES:
            csv_path = os.path.join(raw_dir, f"{table}.csv")
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"{csv_path} not found")

            hdfs_path = f"/data/raw/{table}.csv"
            url = f"{namenode}{hdfs_path}?op=CREATE&overwrite=true&noredirect=true"

            # Step 1 — get the datanode redirect URL
            r1 = requests.put(url, allow_redirects=False)
            if r1.status_code != 200:
                raise RuntimeError(f"Step 1 failed for {table}: {r1.status_code} {r1.text}")
            upload_url = r1.json()["Location"]

            # Step 2 — stream the file to the datanode
            logging.info(f"  Uploading {table}.csv ({os.path.getsize(csv_path) // 1_000_000} MB)...")
            with open(csv_path, "rb") as fh:
                r2 = requests.put(upload_url, data=fh,
                                  headers={"Content-Type": "application/octet-stream"})
            if r2.status_code not in (200, 201):
                raise RuntimeError(f"Step 2 failed for {table}: {r2.status_code} {r2.text}")
            logging.info(f"  {table}.csv — OK")

        logging.info("==> All CSVs uploaded to HDFS")

    put_hdfs = PythonOperator(
        task_id="put_to_hdfs",
        python_callable=_put_to_hdfs,
    )

    # Task 4 — verify row counts in HDFS
    def _verify_hdfs():
        import requests, logging

        namenode = "http://namenode:9870/webhdfs/v1"

        for table in EXPECTED_TABLES:
            hdfs_path = f"/data/raw/{table}.csv"
            url = f"{namenode}{hdfs_path}?op=GETFILESTATUS"
            r = requests.get(url)
            if r.status_code != 200:
                raise RuntimeError(f"{table}.csv not found in HDFS: {r.status_code} {r.text}")
            length = r.json()["FileStatus"]["length"]
            if length < 10:
                raise RuntimeError(f"{table}.csv is empty in HDFS (length={length})")
            logging.info(f"  {table}.csv — {length // 1_000_000} MB — OK")

        logging.info("==> All tables verified in HDFS")

    verify_hdfs = PythonOperator(
        task_id="verify_hdfs",
        python_callable=_verify_hdfs,
    )

    # Task 5 — trigger next DAG
    trigger_clickhouse = TriggerDagRunOperator(
        task_id="trigger_dag_ingest_clickhouse",
        trigger_dag_id="dag_ingest_clickhouse",
        wait_for_completion=False,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    validate >> ensure_hdfs_dirs >> put_hdfs >> verify_hdfs >> trigger_clickhouse