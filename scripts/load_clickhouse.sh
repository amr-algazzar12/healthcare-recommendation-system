#!/usr/bin/env bash
set -euo pipefail

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

  case "$table" in
    "patients")
      # 16 columns
      MAP="Id as patient_id, BIRTHDATE as birthdate, DEATHDATE as deathdate, SSN as ssn, FIRST as first, LAST as last, GENDER as gender, RACE as race, ETHNICITY as ethnicity, CITY as city, STATE as state, ZIP as zip, LAT as lat, LON as lon, HEALTHCARE_EXPENSES as healthcare_expenses, HEALTHCARE_COVERAGE as healthcare_coverage"
      ;;
    "conditions")
      # 6 columns
      MAP="toDate(parseDateTime64BestEffort(START)) as start_date, if(STOP = '', NULL, toDate(parseDateTime64BestEffort(STOP))) as stop_date, PATIENT as patient_id, ENCOUNTER as encounter_id, CODE as code, DESCRIPTION as description"
      ;;
    "medications")
      # 8 columns
      MAP="toDate(parseDateTime64BestEffort(START)) as start_date, if(STOP = '', NULL, toDate(parseDateTime64BestEffort(STOP))) as stop_date, PATIENT as patient_id, ENCOUNTER as encounter_id, CODE as code, DESCRIPTION as description, REASONCODE as reasoncode, REASONDESCRIPTION as reasondescription"
      ;;
    "procedures")
      # 8 columns
      MAP="toDate(parseDateTime64BestEffort(START)) as date, PATIENT as patient_id, ENCOUNTER as encounter_id, CODE as code, DESCRIPTION as description, REASONCODE as reasoncode, REASONDESCRIPTION as reasondescription, BASE_COST as base_cost"
      ;;
    "observations")
      # 8 columns
      MAP="parseDateTime64BestEffort(DATE) as date, PATIENT as patient_id, ENCOUNTER as encounter_id, CODE as code, DESCRIPTION as description, VALUE as value, UNITS as units, TYPE as type"
      ;;
    "encounters")
      # 12 columns
      MAP="Id as encounter_id, parseDateTime64BestEffort(START) as start_dt, if(STOP = '', NULL, parseDateTime64BestEffort(STOP)) as stop_dt, PATIENT as patient_id, ENCOUNTERCLASS as encounterclass, CODE as code, DESCRIPTION as description, REASONCODE as reasoncode, REASONDESCRIPTION as reasondescription, BASE_ENCOUNTER_COST as base_encounter_cost, TOTAL_CLAIM_COST as total_claim_cost, PAYER_COVERAGE as payer_coverage"
      ;;
  esac

  # Generate the header structure so input() knows all identifiers
  HEADER_STRUCT=$(head -n 1 "$csv_file" | sed 's/\r//g' | sed 's/,/ String, /g' | sed 's/$/ String/')

  # Execute streaming load
  docker exec -i clickhouse clickhouse-client \
    --database="${CH_DB}" \
    --input_format_null_as_default=1 \
    --input_format_skip_unknown_fields=1 \
    --date_time_input_format='best_effort' \
    --query="INSERT INTO ${table} SELECT ${MAP} FROM input('${HEADER_STRUCT}') FORMAT CSVWithNames" \
    < "$csv_file"

  local count
  count=$(docker exec clickhouse clickhouse-client --query "SELECT count() FROM ${CH_DB}.${table}")
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
echo "==> Final Summary:"
docker exec clickhouse clickhouse-client \
  --query="SELECT table, formatReadableQuantity(sum(rows)) as rows FROM system.parts WHERE database='${CH_DB}' GROUP BY table ORDER BY table"
# #!/usr/bin/env bash
# set -euo pipefail

# # 1. Dynamic Environment Variables
# # Defaults to localhost (for your Ubuntu host) but accepts overrides (from Airflow)
# CH_HOST="${CH_HOST:-localhost}"
# CH_PORT="${CH_PORT:-8123}"
# CH_DB="${CH_DB:-healthcare}"
# CH_USER="${CH_USER:-healthcare_user}"
# CH_PASS="${CH_PASS:-ch_secret_2026}"

# # 2. Path resolution
# # Defaults to local path, but allows Airflow to point to /opt/airflow/data/raw
# RAW_DIR="${RAW_DIR:-./data/raw}"

# load_table() {
#   local table="$1"
#   local csv_file="$RAW_DIR/$2"
  
#   if [ ! -f "$csv_file" ]; then
#     echo "  WARN: $csv_file not found, skipping"
#     return
#   fi

#   echo "  Loading $table from $csv_file..."

#   # THE FINAL FIX: 
#   # 1. Use --data-binary @- to keep the URL clean.
#   # 2. Use < "$csv_file" to stream the file into curl's stdin.
#   # 3. Add the Chunked header to prevent curl from buffering the whole file.
#   curl -s -u "${CH_USER}:${CH_PASS}" \
#     -H "Transfer-Encoding: chunked" \
#     -X POST "http://${CH_HOST}:${CH_PORT}/?query=INSERT+INTO+${CH_DB}.${table}+FORMAT+CSVWithNames" \
#     --data-binary @- \
#     < "$csv_file" \
#     --fail

#   # Verification
#   local count
#   count=$(curl -s -u "${CH_USER}:${CH_PASS}" \
#     "http://${CH_HOST}:${CH_PORT}/" \
#     --data "SELECT count() FROM ${CH_DB}.${table}")
  
#   echo "  $table: $count rows"
# }

# echo "==> Loading Synthea CSVs into ClickHouse at $CH_HOST..."
# load_table patients      patients.csv
# load_table conditions    conditions.csv
# load_table medications   medications.csv
# load_table observations  observations.csv
# load_table encounters    encounters.csv
# load_table procedures    procedures.csv

# echo ""
# echo "==> Summary:"
# # 5. Final summary via POST request
# curl -s -u "${CH_USER}:${CH_PASS}" "http://${CH_HOST}:${CH_PORT}/" \
#   --data "SELECT table, formatReadableQuantity(sum(rows)) as rows 
#            FROM system.parts 
#            WHERE database='${CH_DB}' 
#            GROUP BY table 
#            ORDER BY table"