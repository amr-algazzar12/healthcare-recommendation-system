#!/bin/bash
set -e

FLASK_APP_PATH="/app/src/api/app.py"
STREAMLIT_APP_PATH="/app/src/dashboard/app.py"
ENGINE_PATH="/app/src/recommender/engine.py"

echo "==> Healthcare Recommendation System — App Container"
echo "    Flask:     ${FLASK_APP_PATH}"
echo "    Streamlit: ${STREAMLIT_APP_PATH}"
echo "    Engine:    ${ENGINE_PATH}"
echo ""

# ── Wait for ClickHouse to be ready ──────────────────────────────────────────
echo "==> Waiting for ClickHouse..."
for i in $(seq 1 30); do
    if curl -sf "http://${CLICKHOUSE_HOST:-clickhouse}:${CLICKHOUSE_HTTP_PORT:-8123}/ping" > /dev/null 2>&1; then
        echo "    ClickHouse ready."
        break
    fi
    echo "    Attempt $i/30 — retrying in 3s..."
    sleep 3
done

# ── Wait for MLflow to be ready ───────────────────────────────────────────────
echo "==> Waiting for MLflow..."
for i in $(seq 1 20); do
    if curl -sf "http://${MLFLOW_TRACKING_URI:-http://mlflow:5001}/health" > /dev/null 2>&1; then
        echo "    MLflow ready."
        break
    fi
    sleep 3
done

# ── Launch Flask API ──────────────────────────────────────────────────────────
if [ -f "$FLASK_APP_PATH" ] && grep -q "Flask\|flask" "$FLASK_APP_PATH" 2>/dev/null; then
    echo "==> Starting Flask API on port 5050..."
    PYTHONPATH=/app/src python "$FLASK_APP_PATH" &
    FLASK_PID=$!
    echo "    Flask PID: $FLASK_PID"
else
    echo "==> Flask stub: src/api/app.py not found, running minimal stub..."
    PYTHONPATH=/app/src python -c "
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'stub'})

@app.route('/recommend', methods=['POST'])
def recommend():
    return jsonify({'status': 'stub', 'message': 'implement src/api/app.py'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
" &
    FLASK_PID=$!
fi

# Give Flask a moment to bind
sleep 3

# Confirm Flask started
if kill -0 $FLASK_PID 2>/dev/null; then
    echo "    Flask running (PID $FLASK_PID)"
else
    echo "    WARN: Flask may have failed to start — check logs"
fi

# ── Launch Streamlit ──────────────────────────────────────────────────────────
if [ -f "$STREAMLIT_APP_PATH" ] && grep -q "streamlit\|st\." "$STREAMLIT_APP_PATH" 2>/dev/null; then
    echo "==> Starting Streamlit dashboard on port 8501..."
    PYTHONPATH=/app/src streamlit run "$STREAMLIT_APP_PATH" \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        --server.fileWatcherType none
else
    echo "==> Streamlit stub: src/dashboard/app.py not found, running minimal stub..."
    PYTHONPATH=/app/src streamlit run - \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false << 'STUB'
import streamlit as st
st.title("Healthcare Recommendation System")
st.info("Dashboard stub — implement src/dashboard/app.py")
st.write("Flask API: http://localhost:5050")
st.write("MLflow:    http://localhost:5001")
STUB
fi
