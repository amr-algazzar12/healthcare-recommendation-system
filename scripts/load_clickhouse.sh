#!/usr/bin/env bash
set -euo pipefail

CH_HOST="${CLICKHOUSE_HOST:-localhost}"
CH_PORT="${CLICKHOUSE_HTTP_PORT:-8123}"
CH_DB="${CLICKHOUSE_DB:-healthcare}"

RAW_DIR="./data/raw"
[ -d "$RAW_DIR/csv" ] && RAW_DIR="$RAW_DIR/csv"

load_table() {
  local table="$1"
  local csv_file="$RAW_DIR/$2"
  if [ ! -f "$csv_file" ]; then
    echo "  WARN: $csv_file not found, skipping"
    return
  fi
  echo "  Loading $table..."
  # Stream line by line via clickhouse-client inside the container
  # avoids curl memory limits on large files entirely
  docker exec -i clickhouse clickhouse-client \
    --query "INSERT INTO ${CH_DB}.${table} FORMAT CSVWithNames" \
    < "$csv_file"
  local count
  count=$(docker exec clickhouse clickhouse-client \
    --query "SELECT count() FROM ${CH_DB}.${table}")
  echo "  $table: $count rows"
}

echo "==> Loading Synthea CSVs into ClickHouse..."
load_table patients      patients.csv
load_table conditions    conditions.csv
load_table medications   medications.csv
load_table observations  observations.csv
load_table encounters    encounters.csv
load_table procedures    procedures.csv

echo ""
echo "==> Summary:"
docker exec clickhouse clickhouse-client \
  --query "SELECT table, formatReadableQuantity(sum(rows)) as rows
           FROM system.parts
           WHERE database='${CH_DB}'
           GROUP BY table
           ORDER BY table"
