"""
DAG 1: dag_ingest_to_hdfs
Validates Synthea CSVs and uploads them to HDFS.
Triggers: dag_ingest_clickhouse
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dag_ingest_to_hdfs",
    default_args=default_args,
    description="Validate CSVs and put them into HDFS",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["milestone-1", "ingestion"],
) as dag:

    validate = BashOperator(
        task_id="validate_csvs",
        bash_command="""
        python /opt/airflow/src/ingestion/validate.py
        """,
    )

    put_hdfs = BashOperator(
        task_id="put_to_hdfs",
        bash_command="""
        hdfs dfs -mkdir -p hdfs://namenode:9001/data/raw
        hdfs dfs -put -f /opt/airflow/data/raw/*.csv hdfs://namenode:9001/data/raw/
        hdfs dfs -ls hdfs://namenode:9001/data/raw/
        """,
    )

    trigger_next = TriggerDagRunOperator(
        task_id="trigger_ingest_clickhouse",
        trigger_dag_id="dag_ingest_clickhouse",
        wait_for_completion=False,
    )

    validate >> put_hdfs >> trigger_next
