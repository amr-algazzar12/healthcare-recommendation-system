"""
dag_spark_processing — STUB
Replace with full implementation in subsequent milestones.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "healthcare",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dag_spark_processing",
    default_args=default_args,
    description="Stub DAG — dag_spark_processing",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["stub"],
) as dag:

    stub = BashOperator(
        task_id="stub_task",
        bash_command='echo "dag_spark_processing stub — implement in later milestones"',
    )
