"""
DAG 2: dag_ingest_clickhouse
Loads raw Synthea CSVs into ClickHouse tables using clickhouse-connect.
Replaces the old SSHOperator → beeline → Hive pattern entirely.

Mirrors scripts/load_clickhouse.sh but runs inside Airflow as a
PythonOperator — no shell, no curl, no memory limits.

Triggers dag_spark_processing on success.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

RAW_DATA_DIR = "/opt/airflow/data/raw"

TABLES = {
    "patients":     "patients.csv",
    "conditions":   "conditions.csv",
    "medications":  "medications.csv",
    "observations": "observations.csv",
    "encounters":   "encounters.csv",
    "procedures":   "procedures.csv",
}

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ── Callables ─────────────────────────────────────────────────────────────────

def load_tables_to_clickhouse():
    """
    Stream each CSV directly into ClickHouse via clickhouse-connect.
    Uses FORMAT CSVWithNames so the header row maps to column names
    automatically — no manual column ordering needed.
    """
    import os
    import sys

    sys.path.insert(0, "/opt/airflow/src")
    from clickhouse.load_tables import load_all_tables

    raw_dir = RAW_DATA_DIR
    # Synthea sometimes outputs to a csv/ subdirectory
    if os.path.isdir(os.path.join(raw_dir, "csv")):
        raw_dir = os.path.join(raw_dir, "csv")

    load_all_tables(raw_dir=raw_dir, tables=TABLES)


def verify_row_counts():
    """
    Assert every table has at least 1 row in ClickHouse.
    Raises ValueError if any table is empty so Airflow marks the task failed.
    """
    import sys

    sys.path.insert(0, "/opt/airflow/src")
    from utils.clickhouse_client import get_client
    from utils.config import CLICKHOUSE_DB

    client = get_client()
    results = {}
    for table in TABLES:
        count = client.query(
            f"SELECT count() FROM {CLICKHOUSE_DB}.{table}"
        ).result_rows[0][0]
        results[table] = count
        print(f"  {table}: {count:,} rows")

    empty = [t for t, c in results.items() if c == 0]
    if empty:
        raise ValueError(f"Tables are empty in ClickHouse: {empty}")

    print("==> All ClickHouse tables verified OK")


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag_ingest_clickhouse",
    default_args=default_args,
    description="Load Synthea CSVs into ClickHouse",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-1", "clickhouse", "ingestion"],
) as dag:

    load = PythonOperator(
        task_id="load_csvs_to_clickhouse",
        python_callable=load_tables_to_clickhouse,
    )

    verify = PythonOperator(
        task_id="verify_row_counts",
        python_callable=verify_row_counts,
    )

    # trigger_next = TriggerDagRunOperator(
    #     task_id="trigger_dag_spark_processing",
    #     trigger_dag_id="dag_spark_processing",
    #     wait_for_completion=False,
    # )

    load >> verify # >> trigger_next