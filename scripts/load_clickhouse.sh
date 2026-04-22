#!/usr/bin/env bash
# =============================================================================
# load_clickhouse.sh — Bulk-load Synthea CSVs into ClickHouse tables
# Run after: make generate-data && make up
# =============================================================================
set -euo pipefail

CH_HOST="${CLICKHOUSE_HOST:-localhost}"
CH_PORT="${CLICKHOUSE_HTTP_PORT:-8123}"
CH_DB="${CLICKHOUSE_DB:-healthcare}"
CH_USER="${CLICKHOUSE_USER:-healthcare_user}"
CH_PASS="${CLICKHOUSE_PASSWORD:-ch_secret_2026}"
RAW_DIR="./data/raw"

load_table() {
  local table="$1"
  local csv_file="$RAW_DIR/$2"
  if [ ! -f "$csv_file" ]; then
    echo "  WARN: $csv_file not found, skipping $table"
    return
  fi
  echo "  Loading $table from $csv_file..."
  # Skip header row, stream CSV directly into ClickHouse HTTP interface
  tail -n +2 "$csv_file" | curl -s \
    "http://${CH_HOST}:${CH_PORT}/?query=INSERT+INTO+${CH_DB}.${table}+FORMAT+CSV" \
    -u "${CH_USER}:${CH_PASS}" \
    --data-binary @-
  echo "  Done: $table"
}

echo "==> Loading Synthea CSVs into ClickHouse..."
# Synthea outputs to data/raw/csv/ or data/raw/ depending on version
CSV_SUBDIR="$RAW_DIR/csv"
[ -d "$CSV_SUBDIR" ] && RAW_DIR="$CSV_SUBDIR"

load_table patients       patients.csv
load_table conditions     conditions.csv
load_table medications    medications.csv
load_table observations   observations.csv
load_table encounters     encounters.csv
load_table procedures     procedures.csv

echo "==> Verifying row counts..."
for t in patients conditions medications observations encounters procedures; do
  COUNT=$(curl -s "http://${CH_HOST}:${CH_PORT}/?query=SELECT+count()+FROM+${CH_DB}.${t}" \
    -u "${CH_USER}:${CH_PASS}")
  printf "  %-25s %s rows\n" "$t" "$COUNT"
done
