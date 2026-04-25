"""
Bulk-load Synthea CSVs into ClickHouse (healthcare database).

Design choices
--------------
* pandas is used only for column renaming + dtype coercion — NOT for
  holding the full observations.csv in memory.  For that file we
  stream in chunks (CHUNK_ROWS) to avoid OOM on the ~1.9 GB file.
* clickhouse-connect client.insert() is used for all tables.
* Column names are normalised from Synthea's mixed-case/uppercase
  headers to the lowercase snake_case used by the ClickHouse schema.
* If a Synthea column is not present in the ClickHouse schema it is
  silently dropped (schema_filter=True).

Environment variables (all read via src/utils/config.py)
---------------------------------------------------------
  CLICKHOUSE_HOST      default: clickhouse
  CLICKHOUSE_PORT      default: 8123   (HTTP interface)
  CLICKHOUSE_USER      default: healthcare_user
  CLICKHOUSE_PASSWORD  default: ch_secret_2026
  CLICKHOUSE_DATABASE  default: healthcare
  DATA_DIR             default: /opt/airflow/data/raw
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk size for large files (observations.csv can be ~1.9 GB)
# ---------------------------------------------------------------------------
CHUNK_ROWS = 50_000

# ---------------------------------------------------------------------------
# Per-table configuration
# ---------------------------------------------------------------------------
# Each entry:
#   csv_filename  : name of the Synthea output file
#   ch_table      : ClickHouse table name (within healthcare database)
#   col_map       : { synthea_column -> clickhouse_column }
#                   Only columns that need renaming are listed.
#                   Columns absent from the ClickHouse schema are dropped.
#   dtype_overrides: passed to pd.read_csv to avoid mis-parsing IDs as ints
# ---------------------------------------------------------------------------
TABLE_CONFIG: list[dict] = [
    {
        "csv_filename": "patients.csv",
        "ch_table": "patients",
        "col_map": {
            "Id": "patient_id",
            "BIRTHDATE": "birthdate",
            "DEATHDATE": "deathdate",
            "SSN": "ssn",
            "FIRST": "first",
            "LAST": "last",
            "GENDER": "gender",
            "RACE": "race",
            "ETHNICITY": "ethnicity",
            "CITY": "city",
            "STATE": "state",
            "ZIP": "zip",
            "LAT": "lat",
            "LON": "lon",
            "HEALTHCARE_EXPENSES": "healthcare_expenses",
            "HEALTHCARE_COVERAGE": "healthcare_coverage",
        },
        "dtype_overrides": {"Id": str, "ZIP": str},
    },
    {
        "csv_filename": "conditions.csv",
        "ch_table": "conditions",
        "col_map": {
            "START": "start_date",
            "STOP": "stop_date",
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "code",
            "DESCRIPTION": "description",
        },
        "dtype_overrides": {"PATIENT": str, "ENCOUNTER": str, "CODE": str},
    },
    {
        "csv_filename": "medications.csv",
        "ch_table": "medications",
        "col_map": {
            "START": "start_date",
            "STOP": "stop_date",
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "code",
            "DESCRIPTION": "description",
            "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
        },
        "dtype_overrides": {
            "PATIENT": str,
            "ENCOUNTER": str,
            "CODE": str,
            "REASONCODE": str,
        },
    },
    {
        "csv_filename": "procedures.csv",
        "ch_table": "procedures",
        "col_map": {
            "START": "date",
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "code",
            "DESCRIPTION": "description",
            "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
            "BASE_COST": "base_cost",
        },
        "dtype_overrides": {
            "PATIENT": str,
            "ENCOUNTER": str,
            "CODE": str,
            "REASONCODE": str,
        },
    },
    {
        "csv_filename": "observations.csv",
        "ch_table": "observations",
        "col_map": {
            "DATE": "date",
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "code",
            "DESCRIPTION": "description",
            "VALUE": "value",
            "UNITS": "units",
            "TYPE": "type",
        },
        "dtype_overrides": {
            "PATIENT": str,
            "ENCOUNTER": str,
            "CODE": str,
            "VALUE": str,
        },
    },
    {
        "csv_filename": "encounters.csv",
        "ch_table": "encounters",
        "col_map": {
            "Id": "encounter_id",
            "START": "start_dt",
            "STOP": "stop_dt",
            "PATIENT": "patient_id",
            "ENCOUNTERCLASS": "encounterclass",
            "CODE": "code",
            "DESCRIPTION": "description",
            "REASONCODE": "reasoncode",
            "REASONDESCRIPTION": "reasondescription",
            "BASE_ENCOUNTER_COST": "base_encounter_cost",
            "TOTAL_CLAIM_COST": "total_claim_cost",
            "PAYER_COVERAGE": "payer_coverage",
        },
        "dtype_overrides": {
            "Id": str,
            "PATIENT": str,
            "CODE": str,
            "REASONCODE": str,
        },
    },
]


# ---------------------------------------------------------------------------
# Date/DateTime columns per table (post-rename ClickHouse names).
# ---------------------------------------------------------------------------
DATE_COLS: dict[str, list[str]] = {
    "patients":     ["birthdate", "deathdate"],
    "conditions":   ["start_date", "stop_date"],
    "medications":  ["start_date", "stop_date"],
    "procedures":   ["date"],
    "observations": [],
    "encounters":   [],
}

DATETIME_COLS: dict[str, list[str]] = {
    "patients":     [],
    "conditions":   [],
    "medications":  [],
    "procedures":   [],
    "observations": ["date"],
    "encounters":   ["start_dt", "stop_dt"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_schema_columns(client, database: str, table: str) -> list[str]:
    """Return the list of column names defined in the ClickHouse schema."""
    result = client.query(
        f"SELECT name FROM system.columns "
        f"WHERE database = '{database}' AND table = '{table}'"
    )
    return [row[0] for row in result.result_rows]


def _coerce_temporals(chunk: pd.DataFrame, table: str) -> pd.DataFrame:
    import datetime
    import pandas as pd
    
    # Ensure these are simple naive objects
    CH_MIN_DATE = datetime.date(1925, 1, 1)
    CH_MIN_DT = datetime.datetime(1925, 1, 1, 0, 0, 0)

    for col in DATE_COLS.get(table, []):
        if col not in chunk.columns:
            continue
        # .dt.tz_localize(None) strips timezone info if it exists
        parsed = pd.to_datetime(chunk[col], errors="coerce").dt.tz_localize(None)
        
        chunk[col] = [
            max(v.date(), CH_MIN_DATE) if pd.notnull(v) else None
            for v in parsed
        ]

    for col in DATETIME_COLS.get(table, []):
        if col not in chunk.columns:
            continue
        # .dt.tz_localize(None) is the key here
        parsed = pd.to_datetime(chunk[col], errors="coerce").dt.tz_localize(None)
        
        chunk[col] = [
            max(v.to_pydatetime(), CH_MIN_DT) if pd.notnull(v) else None
            for v in parsed
        ]

    return chunk

def _iter_chunks(
    csv_path: str,
    col_map: dict,
    dtype_overrides: dict,
    schema_cols: list[str],
    table: str,
    chunk_size: int = CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    """
    Yield renamed, schema-filtered, type-coerced DataFrame chunks.
    """
    for chunk in pd.read_csv(
        csv_path,
        dtype=dtype_overrides,
        chunksize=chunk_size,
        low_memory=False,
    ):
        # 1. Rename columns to ClickHouse convention
        chunk = chunk.rename(columns=col_map)

        # 2. Drop columns not present in the ClickHouse schema
        cols_to_keep = [c for c in chunk.columns if c in schema_cols]
        chunk = chunk[cols_to_keep]

        # 3. Coerce date/datetime strings → Python date/datetime objects
        chunk = _coerce_temporals(chunk, table)

        # 4. Replace NaN / NaT with None so clickhouse-connect maps to NULL
        chunk = chunk.where(pd.notnull(chunk), None)

        yield chunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_tables(data_dir: str | None = None) -> None:
    """
    Truncate and reload all six Synthea tables into ClickHouse.
    """
    import sys

    src_root = "/opt/airflow/src"
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    from utils.clickhouse_client import get_client
    from utils.config import CLICKHOUSE_DATABASE

    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/opt/airflow/data/raw")

    client = get_client()
    database = CLICKHOUSE_DATABASE

    for cfg in TABLE_CONFIG:
        csv_path = os.path.join(data_dir, cfg["csv_filename"])
        table = cfg["ch_table"]
        fq_table = f"{database}.{table}"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Expected CSV not found: {csv_path}. "
                "Run 'make generate-data' first."
            )

        logger.info("Truncating %s ...", fq_table)
        client.command(f"TRUNCATE TABLE IF EXISTS {fq_table}")

        schema_cols = _get_schema_columns(client, database, table)
        if not schema_cols:
            raise RuntimeError(
                f"No columns found for {fq_table}. "
                "Run 'make init_clickhouse' to create the schema."
            )

        total_rows = 0
        logger.info("Loading %s → %s ...", cfg["csv_filename"], fq_table)

        for chunk in _iter_chunks(
            csv_path=csv_path,
            col_map=cfg["col_map"],
            dtype_overrides=cfg["dtype_overrides"],
            schema_cols=schema_cols,
            table=table,
        ):
            if chunk.empty:
                continue
            client.insert_df(fq_table, chunk)
            total_rows += len(chunk)
            logger.info("  %s: %d rows inserted so far ...", table, total_rows)

        logger.info("  %s: done — %d total rows", table, total_rows)

    logger.info("load_all_tables complete.")

if __name__ == "__main__":
    # Configure logging so we can actually see the output in the terminal/Airflow
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    load_all_tables()