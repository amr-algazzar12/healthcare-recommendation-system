"""
validate.py — Pre-flight validation for Synthea CSV files.
Called by dag_ingest_to_hdfs before anything touches HDFS or ClickHouse.
"""
import os
import csv


# Minimum expected columns per table (subset — not exhaustive).
# Validation fails fast if a column is missing entirely.
REQUIRED_COLUMNS = {
    "patients":     {"Id", "BIRTHDATE", "GENDER", "RACE"},
    "conditions":   {"START", "PATIENT", "CODE", "DESCRIPTION"},
    "medications":  {"START", "PATIENT", "CODE", "DESCRIPTION"},
    "observations": {"DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE"},
    "encounters":   {"Id", "START", "PATIENT", "ENCOUNTERCLASS"},
    "procedures":   {"DATE", "PATIENT", "CODE", "DESCRIPTION"},
}


def validate_file(csv_path: str, table: str) -> int:
    """
    Validate a single CSV file.
    Returns the row count (excluding header).
    Raises FileNotFoundError or ValueError on any problem.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    size = os.path.getsize(csv_path)
    if size == 0:
        raise ValueError(f"Empty file: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])

        required = REQUIRED_COLUMNS.get(table, set())
        missing = required - header
        if missing:
            raise ValueError(
                f"{table}.csv is missing required columns: {missing}"
            )

        row_count = sum(1 for _ in reader)

    if row_count == 0:
        raise ValueError(f"{table}.csv has a header but no data rows")

    return row_count


def validate_all(raw_dir: str, tables: list) -> dict:
    """
    Validate all expected CSVs in raw_dir.
    Returns a dict of {table: row_count}.
    Raises on the first problem encountered.
    """
    # Synthea sometimes puts output in a csv/ subdirectory
    if os.path.isdir(os.path.join(raw_dir, "csv")):
        raw_dir = os.path.join(raw_dir, "csv")

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run 'make generate-data' first."
        )

    counts = {}
    print(f"==> Validating CSVs in {raw_dir}")
    for table in tables:
        csv_path = os.path.join(raw_dir, f"{table}.csv")
        count = validate_file(csv_path, table)
        counts[table] = count
        print(f"  {table}: {count:,} rows — OK")

    print("==> All CSVs valid")
    return counts


if __name__ == "__main__":
    # Allow running directly for quick spot-checks:
    #   python src/ingestion/validate.py
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    tables = list(REQUIRED_COLUMNS.keys())
    validate_all(raw_dir=raw, tables=tables)