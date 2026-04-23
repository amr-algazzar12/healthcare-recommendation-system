"""
load_tables.py — Bulk-load Synthea CSVs into ClickHouse.
Called by dag_ingest_clickhouse via PythonOperator.

Uses clickhouse-connect which streams the file in chunks —
no curl memory limits, no shell dependencies.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.clickhouse_client import get_client
from utils.config import CLICKHOUSE_DB

# Column dtype hints to avoid pandas misreading Synthea CSVs
DTYPE_OVERRIDES = {
    "patients": {
        "ZIP":          str,
        "SSN":          str,
        "DRIVERS":      str,
        "PASSPORT":     str,
        "PREFIX":       str,
        "SUFFIX":       str,
        "MAIDEN":       str,
        "MARITAL":      str,
        "ETHNICITY":    str,
        "GENDER":       str,
        "BIRTHPLACE":   str,
        "ADDRESS":      str,
        "CITY":         str,
        "STATE":        str,
        "COUNTY":       str,
    },
    "observations": {
        "VALUE":        str,
        "UNITS":        str,
        "TYPE":         str,
    },
    "medications": {
        "REASONCODE":       str,
        "REASONDESCRIPTION": str,
        "DISPENSES":        str,
        "TOTALCOST":        str,
        "PAYER":            str,
        "PAYER_COVERAGE":   str,
    },
    "encounters": {
        "REASONCODE":        str,
        "REASONDESCRIPTION": str,
        "PAYER":             str,
    },
    "procedures": {
        "REASONCODE":        str,
        "REASONDESCRIPTION": str,
    },
    "conditions": {
        "STOP": str,
    },
}


def _column_map(table: str, df_columns: list) -> dict:
    """
    Map Synthea uppercase CSV columns to ClickHouse lowercase schema columns.
    Only renames columns that exist in both sets — extra Synthea columns are
    dropped, missing optional columns are ignored.
    """
    # ClickHouse schema columns (lowercase) per table
    SCHEMA = {
        "patients": [
            "patient_id", "birthdate", "deathdate", "ssn", "first", "last",
            "gender", "race", "ethnicity", "city", "state", "zip",
            "lat", "lon", "healthcare_expenses", "healthcare_coverage",
        ],
        "conditions": [
            "start_date", "stop_date", "patient_id", "encounter_id",
            "code", "description",
        ],
        "medications": [
            "start_date", "stop_date", "patient_id", "encounter_id",
            "code", "description", "reasoncode", "reasondescription",
        ],
        "observations": [
            "date", "patient_id", "encounter_id", "code", "description",
            "value", "units", "type",
        ],
        "encounters": [
            "encounter_id", "start_dt", "stop_dt", "patient_id",
            "encounterclass", "code", "description",
            "reasoncode", "reasondescription",
            "base_encounter_cost", "total_claim_cost", "payer_coverage",
        ],
        "procedures": [
            "date", "patient_id", "encounter_id", "code", "description",
            "reasoncode", "reasondescription", "base_cost",
        ],
    }

    # Synthea CSV header → ClickHouse column name
    RENAME = {
        "patients": {
            "Id": "patient_id", "BIRTHDATE": "birthdate",
            "DEATHDATE": "deathdate", "SSN": "ssn",
            "FIRST": "first", "LAST": "last", "GENDER": "gender",
            "RACE": "race", "ETHNICITY": "ethnicity",
            "CITY": "city", "STATE": "state", "ZIP": "zip",
            "LAT": "lat", "LON": "lon",
            "HEALTHCARE_EXPENSES": "healthcare_expenses",
            "HEALTHCARE_COVERAGE": "healthcare_coverage",
        },
        "conditions": {
            "START": "start_date", "STOP": "stop_date",
            "PATIENT": "patient_id", "ENCOUNTER": "encounter_id",
            "CODE": "code", "DESCRIPTION": "description",
        },
        "medications": {
            "START": "start_date", "STOP": "stop_date",
            "PATIENT": "patient_id", "ENCOUNTER": "encounter_id",
            "CODE": "code", "DESCRIPTION": "description",
            "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
        },
        "observations": {
            "DATE": "date", "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id", "CODE": "code",
            "DESCRIPTION": "description", "VALUE": "value",
            "UNITS": "units", "TYPE": "type",
        },
        "encounters": {
            "Id": "encounter_id", "START": "start_dt", "STOP": "stop_dt",
            "PATIENT": "patient_id", "ENCOUNTERCLASS": "encounterclass",
            "CODE": "code", "DESCRIPTION": "description",
            "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
            "BASE_ENCOUNTER_COST": "base_encounter_cost",
            "TOTAL_CLAIM_COST": "total_claim_cost",
            "PAYER_COVERAGE": "payer_coverage",
        },
        "procedures": {
            "DATE": "date", "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id", "CODE": "code",
            "DESCRIPTION": "description", "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
            "BASE_COST": "base_cost",
        },
    }
    return RENAME.get(table, {})


def load_table(client, raw_dir: str, table: str, filename: str):
    """Load a single CSV into ClickHouse, returning the row count inserted."""
    csv_path = os.path.join(raw_dir, filename)
    if not os.path.isfile(csv_path):
        print(f"  WARN: {csv_path} not found — skipping {table}")
        return 0

    print(f"  Loading {table} from {csv_path}...")

    dtype = DTYPE_OVERRIDES.get(table, {})
    df = pd.read_csv(csv_path, dtype=dtype, low_memory=False)

    rename_map = _column_map(table, list(df.columns))
    df = df.rename(columns=rename_map)

    # Keep only columns that exist in the ClickHouse schema
    schema_cols = {
        "patients":     ["patient_id","birthdate","deathdate","ssn","first","last","gender","race","ethnicity","city","state","zip","lat","lon","healthcare_expenses","healthcare_coverage"],
        "conditions":   ["start_date","stop_date","patient_id","encounter_id","code","description"],
        "medications":  ["start_date","stop_date","patient_id","encounter_id","code","description","reasoncode","reasondescription"],
        "observations": ["date","patient_id","encounter_id","code","description","value","units","type"],
        "encounters":   ["encounter_id","start_dt","stop_dt","patient_id","encounterclass","code","description","reasoncode","reasondescription","base_encounter_cost","total_claim_cost","payer_coverage"],
        "procedures":   ["date","patient_id","encounter_id","code","description","reasoncode","reasondescription","base_cost"],
    }
    keep = [c for c in schema_cols.get(table, []) if c in df.columns]
    df = df[keep]

    client.insert_df(f"{CLICKHOUSE_DB}.{table}", df)
    count = client.query(
        f"SELECT count() FROM {CLICKHOUSE_DB}.{table}"
    ).result_rows[0][0]
    print(f"  {table}: {count:,} rows total")
    return count


def load_all_tables(raw_dir: str, tables: dict):
    """
    Load all tables. tables is a dict of {table_name: filename}.
    """
    client = get_client()
    totals = {}
    for table, filename in tables.items():
        totals[table] = load_table(client, raw_dir, table, filename)

    print("\n==> ClickHouse load summary:")
    for table, count in totals.items():
        print(f"  {table}: {count:,} rows")
    return totals


if __name__ == "__main__":
    # Quick manual run: python src/clickhouse/load_tables.py ./data/raw
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    if os.path.isdir(os.path.join(raw, "csv")):
        raw = os.path.join(raw, "csv")
    TABLES = {
        "patients":     "patients.csv",
        "conditions":   "conditions.csv",
        "medications":  "medications.csv",
        "observations": "observations.csv",
        "encounters":   "encounters.csv",
        "procedures":   "procedures.csv",
    }
    load_all_tables(raw_dir=raw, tables=TABLES)



# """
# load_tables.py — Load Synthea CSVs into ClickHouse.
# Called by dag_ingest_clickhouse Airflow DAG via PythonOperator.
# Replaces the old SSHOperator → Beeline → Hive pattern.
# """
# import os
# import pandas as pd
# from src.utils.clickhouse_client import get_client
# from src.utils.config import CLICKHOUSE_DB

# RAW_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/data/raw")

# TABLES = {
#     "patients": "patients.csv",
#     "conditions": "conditions.csv",
#     "medications": "medications.csv",
#     "observations": "observations.csv",
#     "encounters": "encounters.csv",
#     "procedures": "procedures.csv",
# }


# def load_all_tables():
#     client = get_client()
#     for table, filename in TABLES.items():
#         csv_path = os.path.join(RAW_DIR, filename)
#         if not os.path.exists(csv_path):
#             # Synthea sometimes puts output in a csv/ subdirectory
#             csv_path = os.path.join(RAW_DIR, "csv", filename)
#         if not os.path.exists(csv_path):
#             print(f"  WARN: {filename} not found, skipping {table}")
#             continue
#         print(f"  Loading {table} from {csv_path}...")
#         df = pd.read_csv(csv_path, low_memory=False)
#         client.insert_df(f"{CLICKHOUSE_DB}.{table}", df)
#         count = client.query(f"SELECT count() FROM {CLICKHOUSE_DB}.{table}").result_rows[0][0]
#         print(f"  {table}: {count:,} rows loaded")


# if __name__ == "__main__":
#     load_all_tables()
