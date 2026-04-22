"""
load_tables.py — Load Synthea CSVs into ClickHouse.
Called by dag_ingest_clickhouse Airflow DAG via PythonOperator.
Replaces the old SSHOperator → Beeline → Hive pattern.
"""
import os
import pandas as pd
from src.utils.clickhouse_client import get_client
from src.utils.config import CLICKHOUSE_DB

RAW_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/data/raw")

TABLES = {
    "patients": "patients.csv",
    "conditions": "conditions.csv",
    "medications": "medications.csv",
    "observations": "observations.csv",
    "encounters": "encounters.csv",
    "procedures": "procedures.csv",
}


def load_all_tables():
    client = get_client()
    for table, filename in TABLES.items():
        csv_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(csv_path):
            # Synthea sometimes puts output in a csv/ subdirectory
            csv_path = os.path.join(RAW_DIR, "csv", filename)
        if not os.path.exists(csv_path):
            print(f"  WARN: {filename} not found, skipping {table}")
            continue
        print(f"  Loading {table} from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        client.insert_df(f"{CLICKHOUSE_DB}.{table}", df)
        count = client.query(f"SELECT count() FROM {CLICKHOUSE_DB}.{table}").result_rows[0][0]
        print(f"  {table}: {count:,} rows loaded")


if __name__ == "__main__":
    load_all_tables()
