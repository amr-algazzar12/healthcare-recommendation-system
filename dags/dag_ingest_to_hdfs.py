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
            hdfs dfs -mkdir -p {raw} && \
            hdfs dfs -mkdir -p hdfs://namenode:9001/data/processed && \
            hdfs dfs -mkdir -p hdfs://namenode:9001/data/features && \
            hdfs dfs -chmod -R 777 hdfs://namenode:9001/data && \
            echo "HDFS directories ready"
        """.format(raw=HDFS_RAW),
    )

    # Task 3 — load CSVs into HDFS
    # Mirrors Makefile load-hdfs exactly:
    #   docker cp is not available inside the container, so we use
    #   hdfs dfs -put from the volume-mounted path instead.
    #   The Makefile uses docker cp + moveFromLocal from the host;
    #   from inside Airflow we use -put -f directly (same result,
    #   no temp copy needed because the volume is already mounted).
    put_hdfs = BashOperator(
        task_id="put_to_hdfs",
        bash_command="""
            set -e
            RAW_DIR="{raw_dir}"

            # Resolve Synthea subdirectory if present
            if [ -d "$RAW_DIR/csv" ]; then
                RAW_DIR="$RAW_DIR/csv"
            fi

            echo "==> Uploading CSVs from $RAW_DIR to {hdfs_raw}"

            for TABLE in {tables}; do
                CSV="$RAW_DIR/$TABLE.csv"
                if [ ! -f "$CSV" ]; then
                    echo "ERROR: $CSV not found"
                    exit 1
                fi
                echo "  Putting $TABLE.csv..."
                hdfs dfs -put -f "$CSV" {hdfs_raw}/
            done

            echo "==> HDFS listing:"
            hdfs dfs -ls {hdfs_raw}/
        """.format(
            raw_dir=RAW_DATA_DIR,
            hdfs_raw=HDFS_RAW,
            tables=" ".join(f"{t}" for t in EXPECTED_TABLES),
        ),
    )

    # Task 4 — verify row counts in HDFS
    verify_hdfs = BashOperator(
        task_id="verify_hdfs",
        bash_command="""
            set -e
            echo "==> Verifying HDFS row counts:"
            for TABLE in {tables}; do
                LINES=$(hdfs dfs -cat {hdfs_raw}/$TABLE.csv 2>/dev/null | wc -l)
                # subtract 1 for header
                ROWS=$((LINES - 1))
                echo "  $TABLE: $ROWS rows"
                if [ "$ROWS" -lt 1 ]; then
                    echo "ERROR: $TABLE is empty in HDFS"
                    exit 1
                fi
            done
            echo "==> All tables verified OK"
        """.format(
            hdfs_raw=HDFS_RAW,
            tables=" ".join(EXPECTED_TABLES),
        ),
    )

    # Task 5 — trigger next DAG
    trigger_clickhouse = TriggerDagRunOperator(
        task_id="trigger_dag_ingest_clickhouse",
        trigger_dag_id="dag_ingest_clickhouse",
        wait_for_completion=False,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    validate >> ensure_hdfs_dirs >> put_hdfs >> verify_hdfs >> trigger_clickhouse