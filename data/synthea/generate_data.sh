#!/usr/bin/env bash
# =============================================================================
# generate_data.sh
# Generates a reproducible Synthea patient dataset on the HOST machine.
#
# Prerequisites (on host):
#   - Java 11+  (check: java -version)
#   - Internet access to download the Synthea JAR on first run
#
# Usage:
#   chmod +x generate_data.sh
#   ./data/synthea/generate_data.sh
#
# Output:
#   data/raw/patients.csv
#   data/raw/conditions.csv
#   data/raw/medications.csv
#   data/raw/observations.csv
#   data/raw/encounters.csv
#   data/raw/procedures.csv
#   (+ other Synthea output files — only the above six are used)
#
# Reproducibility:
#   --seed 42 and --population 10000 are pinned.
#   Do NOT change these values without updating the proposal and team.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — change only if deliberately updating the dataset
# ---------------------------------------------------------------------------
SYNTHEA_VERSION="3.2.0"
SYNTHEA_JAR="synthea-with-dependencies.jar"
SYNTHEA_URL="https://github.com/synthetichealth/synthea/releases/download/v${SYNTHEA_VERSION}/${SYNTHEA_JAR}"

POPULATION=10000
SEED=42
STATE="Massachusetts"   # Synthea requires a US state

# Resolve paths relative to the project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/raw"
SYNTHEA_DIR="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  Synthea Data Generation"
echo "  Population : $POPULATION patients"
echo "  Seed       : $SEED"
echo "  State      : $STATE"
echo "  Output     : $OUTPUT_DIR"
echo "================================================================"

# Check Java
if ! command -v java &>/dev/null; then
    echo "[ERROR] Java not found. Install Java 11+ before running this script."
    echo "        Ubuntu/Debian: sudo apt-get install openjdk-11-jdk"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 11 ]; then
    echo "[ERROR] Java 11+ required. Found Java $JAVA_VERSION."
    exit 1
fi

echo "[OK] Java $JAVA_VERSION detected."

# ---------------------------------------------------------------------------
# Download Synthea JAR if not present
# ---------------------------------------------------------------------------
JAR_PATH="$SYNTHEA_DIR/$SYNTHEA_JAR"

if [ ! -f "$JAR_PATH" ]; then
    echo "Downloading Synthea v${SYNTHEA_VERSION}..."
    curl -L --progress-bar "$SYNTHEA_URL" -o "$JAR_PATH"
    echo "[OK] Downloaded $SYNTHEA_JAR"
else
    echo "[OK] Synthea JAR already present: $JAR_PATH"
fi

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Run Synthea
# ---------------------------------------------------------------------------
echo "Generating $POPULATION patients (seed=$SEED)..."
echo "This takes approximately 3–6 minutes..."

java -jar "$JAR_PATH" \
    --exporter.csv.export=true \
    --exporter.fhir.export=false \
    --exporter.hospital.fhir.export=false \
    --exporter.practitioner.fhir.export=false \
    --exporter.baseDirectory="$OUTPUT_DIR" \
    --generate.only_alive_patients=false \
    -p "$POPULATION" \
    -s "$SEED" \
    "$STATE"

# ---------------------------------------------------------------------------
# Synthea writes CSVs to $OUTPUT_DIR/csv/ — move them up to $OUTPUT_DIR/
# ---------------------------------------------------------------------------
if [ -d "$OUTPUT_DIR/csv" ]; then
    echo "Moving CSV files from $OUTPUT_DIR/csv/ to $OUTPUT_DIR/..."
    mv "$OUTPUT_DIR/csv/"*.csv "$OUTPUT_DIR/"
    rmdir "$OUTPUT_DIR/csv" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------
REQUIRED_FILES=("patients.csv" "conditions.csv" "medications.csv"
                "observations.csv" "encounters.csv" "procedures.csv")

echo ""
echo "Verifying output files..."
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    FILEPATH="$OUTPUT_DIR/$f"
    if [ -f "$FILEPATH" ]; then
        ROWS=$(wc -l < "$FILEPATH")
        echo "  [OK] $f — $ROWS lines (including header)"
    else
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "================================================================"
    echo "  Data generation complete."
    echo "  Run 'make up' and then 'make upload-hdfs' to ingest the data."
    echo "================================================================"
else
    echo "[ERROR] Some files are missing. Check Synthea logs above."
    exit 1
fi