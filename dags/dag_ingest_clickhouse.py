"""
DAG 2: dag_ingest_clickhouse
Loads raw CSVs into ClickHouse tables.
Replaces dag_create_hive_tables from v4.
Triggers: dag_spark_processing
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import sys
sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dag_ingest_clickhouse",
    default_args=default_args,
    description="Bulk-load Synthea CSVs into ClickHouse",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-1", "clickhouse"],
) as dag:

    def _load_tables():
        from clickhouse.load_tables import load_all_tables
        load_all_tables()

    load = PythonOperator(
        task_id="load_csvs_to_clickhouse",
        python_callable=_load_tables,
    )

    verify = PythonOperator(
        task_id="verify_row_counts",
        python_callable=lambda: __import__(
            "clickhouse.load_tables", fromlist=["load_all_tables"]
        ),  # placeholder — replace with real verification
    )

    trigger_next = TriggerDagRunOperator(
        task_id="trigger_spark_processing",
        trigger_dag_id="dag_spark_processing",
        wait_for_completion=False,
    )

    load >> verify >> trigger_next
