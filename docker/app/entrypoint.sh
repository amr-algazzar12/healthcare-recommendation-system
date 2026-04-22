#!/bin/bash
set -e

# If the real Flask app exists, use it; otherwise run the stub
FLASK_APP_PATH="/app/src/api/app.py"
STREAMLIT_APP_PATH="/app/src/dashboard/app.py"

if [ -f "$FLASK_APP_PATH" ] && grep -q "Flask\|flask" "$FLASK_APP_PATH" 2>/dev/null; then
  echo "Starting Flask API..."
  python "$FLASK_APP_PATH" &
else
  echo "Flask stub: real app not implemented yet, running health endpoint..."
  python -c "
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'stub — implement src/api/app.py'})

@app.route('/recommend', methods=['POST'])
def recommend():
    return jsonify({'status': 'stub', 'message': 'Model serving not yet implemented'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
" &
fi

if [ -f "$STREAMLIT_APP_PATH" ] && grep -q "streamlit\|st\." "$STREAMLIT_APP_PATH" 2>/dev/null; then
  echo "Starting Streamlit dashboard..."
  streamlit run "$STREAMLIT_APP_PATH" --server.port 8501 --server.address 0.0.0.0
else
  echo "Streamlit stub: real dashboard not implemented yet..."
  python -c "
import streamlit as st
st.title('Healthcare Recommendation System')
st.info('Dashboard stub — implement src/dashboard/app.py')
st.markdown('### Services')
st.write('Flask API: http://localhost:5050')
st.write('MLflow: http://localhost:5001')
" &
  # Keep container alive
  tail -f /dev/null
fi
