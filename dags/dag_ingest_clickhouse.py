"""
dag_ingest_clickhouse.py
Milestone 1 — ingest Synthea CSVs into ClickHouse.

Uses BashOperator to stream CSVs directly into ClickHouse via
clickhouse-client inside the ClickHouse container — identical to
scripts/load_clickhouse.sh but orchestrated by Airflow.

This avoids loading multi-GB files (observations: 2GB) into the
Airflow container memory via pandas, which causes the task to hang.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # M2

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

TABLES = [
    "patients",
    "conditions",
    "medications",
    "observations",
    "encounters",
    "procedures",
]

# Credentials — match docker-compose.yml ClickHouse env
CH_HOST = "clickhouse"
CH_PORT = "8123"          # native TCP — faster than HTTP for bulk inserts
CH_DB   = "healthcare"
CH_USER = "healthcare_user"
CH_PASS = "ch_secret_2026"

# Raw data dir inside Airflow container (volume-mounted from host ./data/raw)
RAW_DIR = "/opt/airflow/data/raw"

with DAG(
    dag_id="dag_ingest_clickhouse",
    default_args=default_args,
    description="Load Synthea CSVs into ClickHouse (Milestone 1)",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-1", "clickhouse"],
) as dag:

    # ── Task 1 — truncate all tables ──────────────────────────────────────────
    # Always truncate first so re-runs don't stack duplicate rows
    truncate = BashOperator(
        task_id="truncate_tables",
        bash_command="""
            set -e
            for TABLE in patients conditions medications observations encounters procedures; do
                echo "  Truncating healthcare.$TABLE ..."
                # Use --data to send a POST request
                curl -s -u "{user}:{passwd}" \
                     "http://{host}:8123/" \
                     --data "TRUNCATE TABLE IF EXISTS {db}.$TABLE" \
                     --fail
            done
            echo "==> All tables truncated"
        """.format(
            host=CH_HOST,
            db=CH_DB,
            user=CH_USER,
            passwd=CH_PASS,
        ),
    )

    # ── Task 2 — stream CSVs into ClickHouse ─────────────────────────────────
    load = BashOperator(
    task_id="load_csvs_to_clickhouse",
    # Add the space at the end to avoid the TemplateNotFound error
    bash_command="python3 /opt/airflow/src/clickhouse/load_tables.py ",
    env={
        'CH_HOST': 'clickhouse',              # Service name in docker-compose
        'RAW_DIR': '/opt/airflow/data/raw',    # Path inside Airflow container
        'CH_USER': 'healthcare_user',
        'CH_PASS': 'ch_secret_2026',
        'CH_DB':   'healthcare'
    },
    execution_timeout=timedelta(minutes=60),
)

    # ── Task 3 — verify row counts ────────────────────────────────────────────
    def _verify_row_counts(**ctx):
        import sys
        src_root = "/opt/airflow/src"
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

        from utils.clickhouse_client import get_client

        client = get_client()
        empty = []
        for table in TABLES:
            cnt = client.query(
                f"SELECT count() FROM {CH_DB}.{table}"
            ).result_rows[0][0]
            print(f"  {CH_DB}.{table}: {cnt:,} rows")
            if cnt == 0:
                empty.append(table)

        if empty:
            raise ValueError(f"Empty tables after load: {empty}")

        print("==> All ClickHouse tables verified OK")

    verify = PythonOperator(
        task_id="verify_row_counts",
        python_callable=_verify_row_counts,
    )

    # ── Task 4 — trigger M2 (commented out until Milestone 2 is ready) ────────
    # trigger_spark = TriggerDagRunOperator(
    #     task_id="trigger_dag_spark_processing",
    #     trigger_dag_id="dag_spark_processing",
    #     wait_for_completion=False,
    # )

    # ── Dependencies ──────────────────────────────────────────────────────────
    truncate >> load >> verify
    # verify >> trigger_spark