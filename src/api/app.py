"""
app.py — Flask REST API (Milestone 4)

Endpoints:
    GET  /health                    — liveness check
    GET  /model/info                — loaded model version + metadata
    POST /recommend                 — generate recommendations for a patient
    GET  /recommend/<patient_id>    — fetch stored recommendations from ClickHouse
    GET  /patients                  — list patients with feature data
    GET  /patients/<patient_id>     — patient profile + feature summary
    POST /model/reload              — hot-reload model from HDFS

Run:
    python src/api/app.py
    (launched by docker/app/entrypoint.sh)
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from recommender.engine import (
    load_model,
    recommend,
    get_patient_features,
    get_patient_known_medications,
    _get_ch_client,
    CH_DB,
    DEFAULT_TOP_K,
)

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Pre-load model at startup so first request isn't slow
try:
    load_model()
    logger.info("Model pre-loaded at startup")
except Exception as e:
    logger.warning("Could not pre-load model at startup: %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error(message: str, status: int = 400) -> tuple:
    return jsonify({"status": "error", "message": message}), status


def _ok(data: dict) -> tuple:
    return jsonify({"status": "ok", **data}), 200


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — always returns 200 if Flask is running."""
    try:
        client = _get_ch_client()
        ch_ok  = client.query("SELECT 1").result_rows[0][0] == 1
    except Exception:
        ch_ok = False

    return _ok({
        "service":    "healthcare-recommendation-api",
        "clickhouse": "ok" if ch_ok else "unreachable",
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return metadata about the currently loaded model."""
    try:
        bundle = load_model()
        return _ok({
            "model_version":  bundle["version"],
            "feature_count":  len(bundle["feature_cols"]),
            "feature_cols":   bundle["feature_cols"],
            "model_type":     type(bundle["model"]).__name__,
        })
    except Exception as e:
        return _error(f"Model not loaded: {e}", 503)


@app.route("/model/reload", methods=["POST"])
def model_reload():
    """Hot-reload the model from HDFS without restarting the container."""
    try:
        bundle = load_model(force_reload=True)
        return _ok({
            "message":       "Model reloaded from HDFS",
            "model_version": bundle["version"],
        })
    except Exception as e:
        return _error(f"Reload failed: {e}", 500)


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    """
    Generate medication recommendations for a patient.

    Request body (JSON):
        {
            "patient_id": "uuid-string",
            "top_k": 10          (optional, default 10)
        }

    Response:
        {
            "status": "ok",
            "patient_id": "...",
            "model_version": "1",
            "recommendations": [
                {
                    "rank": 1,
                    "treatment_code": "313782",
                    "treatment_name": "Acetaminophen ...",
                    "score": 0.923,
                    "explanation": "Ranked #1 ..."
                },
                ...
            ]
        }
    """
    body = request.get_json(silent=True) or {}

    patient_id = body.get("patient_id", "").strip()
    if not patient_id:
        return _error("patient_id is required")

    top_k = body.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = int(top_k)
        if top_k < 1 or top_k > 50:
            return _error("top_k must be between 1 and 50")
    except (ValueError, TypeError):
        return _error("top_k must be an integer")

    try:
        recs    = recommend(patient_id, top_k=top_k)
        bundle  = load_model()
        return _ok({
            "patient_id":      patient_id,
            "model_version":   bundle["version"],
            "top_k":           top_k,
            "recommendations": recs,
        })
    except ValueError as e:
        return _error(str(e), 404)
    except Exception as e:
        logger.error("recommend() failed: %s\n%s", e, traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


@app.route("/recommend/<patient_id>", methods=["GET"])
def get_stored_recommendations(patient_id: str):
    """
    Fetch previously stored recommendations for a patient from ClickHouse.
    Does not re-run the model — reads from the recommendations table.
    """
    try:
        client = _get_ch_client()
        df = client.query_df(f"""
            SELECT
                rank, treatment_code, treatment_name,
                score, explanation, model_version,
                toString(created_at) AS created_at
            FROM {CH_DB}.recommendations
            WHERE patient_id = '{patient_id}'
            ORDER BY rank ASC
        """)

        if df.empty:
            return _ok({
                "patient_id":      patient_id,
                "recommendations": [],
                "message": "No stored recommendations — call POST /recommend first",
            })

        return _ok({
            "patient_id":      patient_id,
            "model_version":   df["model_version"].iloc[0],
            "created_at":      df["created_at"].iloc[0],
            "recommendations": df.drop(columns=["model_version", "created_at"])
                                  .to_dict(orient="records"),
        })
    except Exception as e:
        return _error(f"Failed to fetch recommendations: {e}", 500)


@app.route("/patients", methods=["GET"])
def list_patients():
    """
    List patients available in the feature store.

    Query params:
        limit  (default 20, max 200)
        offset (default 0)
        has_diabetes / has_hypertension / has_asthma (0 or 1 filter)
    """
    try:
        limit  = min(int(request.args.get("limit",  20)),  200)
        offset = max(int(request.args.get("offset",  0)),    0)

        filters = []
        for flag in ["has_diabetes", "has_hypertension",
                     "has_asthma", "has_hyperlipidemia", "has_coronary_disease"]:
            val = request.args.get(flag)
            if val is not None:
                filters.append(f"{flag} = {int(val)}")

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        client = _get_ch_client()
        df = client.query_df(f"""
            SELECT
                patient_id, age, gender_encoded, race_encoded,
                num_conditions, num_medications, num_encounters,
                has_diabetes, has_hypertension, has_asthma,
                has_hyperlipidemia, has_coronary_disease
            FROM {CH_DB}.patient_features
            {where}
            ORDER BY patient_id
            LIMIT {limit} OFFSET {offset}
        """)

        total = client.query(
            f"SELECT count() FROM {CH_DB}.patient_features {where}"
        ).result_rows[0][0]

        return _ok({
            "total":    int(total),
            "limit":    limit,
            "offset":   offset,
            "patients": df.to_dict(orient="records"),
        })
    except Exception as e:
        return _error(f"Failed to list patients: {e}", 500)


@app.route("/patients/<patient_id>", methods=["GET"])
def get_patient(patient_id: str):
    """
    Return patient profile: features + known medications + stored recommendations.
    """
    try:
        features = get_patient_features(patient_id)
        if features is None:
            return _error(f"Patient '{patient_id}' not found", 404)

        known_meds = get_patient_known_medications(patient_id)

        client = _get_ch_client()

        # Medication details
        med_df = client.query_df(f"""
            SELECT DISTINCT
                code AS medication_code,
                any(description) AS medication_name,
                toString(max(start_date)) AS last_prescribed
            FROM {CH_DB}.medications
            WHERE patient_id = '{patient_id}'
            GROUP BY code
            ORDER BY last_prescribed DESC
        """)

        # Stored recommendations
        rec_df = client.query_df(f"""
            SELECT rank, treatment_code, treatment_name, score
            FROM {CH_DB}.recommendations
            WHERE patient_id = '{patient_id}'
            ORDER BY rank ASC
            LIMIT 10
        """)

        return _ok({
            "patient_id":       patient_id,
            "features":         {
                k: (int(v) if hasattr(v, 'item') else v)
                for k, v in features.items()
                if k != "patient_id"
            },
            "medications":      med_df.to_dict(orient="records"),
            "recommendations":  rec_df.to_dict(orient="records"),
        })
    except Exception as e:
        return _error(f"Failed to get patient: {e}", 500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5050))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info("Starting Flask API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
