"""
dag_evaluate_and_register — STUB
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
    dag_id="dag_evaluate_and_register",
    default_args=default_args,
    description="Stub DAG — dag_evaluate_and_register",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["stub"],
) as dag:

    stub = BashOperator(
        task_id="stub_task",
        bash_command='echo "dag_evaluate_and_register stub — implement in later milestones"',
    )
