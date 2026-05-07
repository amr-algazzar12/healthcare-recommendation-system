"""
engine.py — Recommendation engine (Milestone 4)

Loads the Production model from HDFS (XGBoost via joblib),
builds candidate (patient, medication) pairs, scores them,
and returns ranked medication recommendations with explanations.

Designed to be imported by both the Flask API and Streamlit dashboard.
Thread-safe: model is loaded once at import time and reused.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HDFS_BASE        = os.environ.get("HDFS_NAMENODE_URL", "http://namenode:9870")
HDFS_MODEL_PATH  = "/models/hybrid_xgboost/model.joblib"
CH_HOST          = os.environ.get("CLICKHOUSE_HOST",      "clickhouse")
CH_PORT          = int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123"))
CH_DB            = os.environ.get("CLICKHOUSE_DB",         "healthcare")
CH_USER          = os.environ.get("CLICKHOUSE_USER",        "healthcare_user")
CH_PASS          = os.environ.get("CLICKHOUSE_PASSWORD",    "ch_secret_2026")
MLFLOW_URI       = os.environ.get("MLFLOW_TRACKING_URI",   "http://mlflow:5001")
BEST_MODEL_PATH  = os.environ.get("BEST_MODEL_JSON",
                                  "/app/models/best_model.json")
DEFAULT_TOP_K    = 10

# These must match the columns used in hybrid_model.py build_training_dataset
SCALAR_FEATURE_COLS = [
    "age", "gender_encoded", "race_encoded",
    "num_conditions", "num_medications", "num_encounters",
    "has_diabetes", "has_hypertension", "has_asthma",
    "has_hyperlipidemia", "has_coronary_disease",
]
ALL_FEATURE_COLS = SCALAR_FEATURE_COLS + ["sim_med_count", "med_prevalence"]

# ── Singleton model holder ────────────────────────────────────────────────────
_model_lock   = threading.Lock()
_model_cache  = {}     # {"model": xgb, "feature_cols": [...], "version": str}


# ── ClickHouse client ─────────────────────────────────────────────────────────

def _get_ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        database=CH_DB, username=CH_USER, password=CH_PASS,
        compress=False,
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_from_hdfs() -> dict:
    """
    Download the serialised model bundle from HDFS via WebHDFS REST API.
    Returns {"model": xgb_model, "feature_cols": [...]}
    """
    url = (
        f"{HDFS_BASE}/webhdfs/v1{HDFS_MODEL_PATH}"
        f"?op=OPEN&noredirect=true"
    )
    r1  = requests.get(url, timeout=15)
    r1.raise_for_status()

    download_url = r1.json().get("Location")
    if not download_url:
        raise RuntimeError("WebHDFS OPEN returned no Location URL")

    r2 = requests.get(download_url, timeout=60)
    r2.raise_for_status()

    bundle = joblib.load(io.BytesIO(r2.content))
    logger.info("Model loaded from HDFS (%d bytes)", len(r2.content))
    return bundle


def _read_best_model_version() -> str:
    try:
        with open(BEST_MODEL_PATH) as f:
            return json.load(f).get("version", "1")
    except Exception:
        return "1"


def load_model(force_reload: bool = False) -> dict:
    """
    Load model from HDFS with singleton caching.
    Thread-safe — only one thread downloads at a time.
    """
    global _model_cache
    with _model_lock:
        if _model_cache and not force_reload:
            return _model_cache

        logger.info("Loading model from HDFS: %s", HDFS_MODEL_PATH)
        bundle  = _load_from_hdfs()
        version = _read_best_model_version()

        _model_cache = {
            "model":        bundle["model"],
            "feature_cols": bundle.get("feature_cols", ALL_FEATURE_COLS),
            "version":      version,
        }
        logger.info("Model loaded (version=%s, features=%d)",
                    version, len(_model_cache["feature_cols"]))
        return _model_cache


# ── Patient feature retrieval ─────────────────────────────────────────────────

def get_patient_features(patient_id: str) -> Optional[dict]:
    """
    Fetch scalar features for a single patient from ClickHouse.
    Returns None if patient_id not found.
    """
    client = _get_ch_client()
    df = client.query_df(f"""
        SELECT
            patient_id, age, gender_encoded, race_encoded,
            num_conditions, num_medications, num_encounters,
            has_diabetes, has_hypertension, has_asthma,
            has_hyperlipidemia, has_coronary_disease
        FROM {CH_DB}.patient_features
        WHERE patient_id = '{patient_id}'
        LIMIT 1
    """)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_patient_known_medications(patient_id: str) -> set[str]:
    """Medications the patient has already received."""
    client = _get_ch_client()
    df = client.query_df(f"""
        SELECT DISTINCT code AS medication_code
        FROM {CH_DB}.medications
        WHERE patient_id = '{patient_id}'
    """)
    return set(df["medication_code"].tolist()) if not df.empty else set()


def get_medication_catalogue() -> pd.DataFrame:
    """
    All unique medication codes and their descriptions.
    Used to build candidate list and display names.
    """
    client = _get_ch_client()
    return client.query_df(f"""
        SELECT
            code AS medication_code,
            any(description) AS medication_name,
            count(DISTINCT patient_id) AS global_count
        FROM {CH_DB}.medications
        WHERE code != ''
        GROUP BY code
        ORDER BY global_count DESC
    """)


def get_global_medication_prevalence() -> dict[str, float]:
    """medication_code → fraction of patients who received it."""
    client = _get_ch_client()
    total = client.query(
        f"SELECT count(DISTINCT patient_id) FROM {CH_DB}.patient_features"
    ).result_rows[0][0]

    df = client.query_df(f"""
        SELECT code, count(DISTINCT patient_id) AS n
        FROM {CH_DB}.medications
        GROUP BY code
    """)
    return {
        row["code"]: row["n"] / total
        for _, row in df.iterrows()
    }


# ── Feature construction ──────────────────────────────────────────────────────

def build_candidate_features(
    patient_features: dict,
    candidate_codes:  list[str],
    prevalence_map:   dict[str, float],
    known_meds:       set[str],
    sim_med_count:    int = 0,
) -> pd.DataFrame:
    """
    Build one row per candidate medication for scoring.

    Columns match ALL_FEATURE_COLS exactly:
        age, gender_encoded, race_encoded,
        num_conditions, num_medications, num_encounters,
        has_diabetes, has_hypertension, has_asthma,
        has_hyperlipidemia, has_coronary_disease,
        sim_med_count, med_prevalence
    """
    rows = []
    for code in candidate_codes:
        if code in known_meds:
            continue
        rows.append({
            "medication_code": code,
            **{col: patient_features.get(col, 0)
               for col in SCALAR_FEATURE_COLS},
            "sim_med_count":  sim_med_count,
            "med_prevalence": prevalence_map.get(code, 0.0),
        })
    return pd.DataFrame(rows)


# ── Scoring & ranking ─────────────────────────────────────────────────────────

def _build_explanation(rank: int, score: float,
                       patient_features: dict,
                       med_name: str) -> str:
    """
    Generate a human-readable explanation for a recommendation.
    Kept simple and deterministic — no LLM needed.
    """
    reasons = []

    if patient_features.get("has_diabetes"):
        reasons.append("diabetes management")
    if patient_features.get("has_hypertension"):
        reasons.append("hypertension control")
    if patient_features.get("has_asthma"):
        reasons.append("asthma treatment")
    if patient_features.get("has_hyperlipidemia"):
        reasons.append("lipid management")
    if patient_features.get("has_coronary_disease"):
        reasons.append("coronary disease care")

    condition_part = (
        f"aligned with {', '.join(reasons)}"
        if reasons else
        "matches patient profile"
    )

    return (
        f"Ranked #{rank} with confidence {score:.1%}. "
        f"This medication is {condition_part} "
        f"based on {patient_features.get('num_encounters', 0)} encounters "
        f"and {patient_features.get('num_conditions', 0)} active conditions."
    )


def recommend(
    patient_id: str,
    top_k:      int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Main recommendation function.

    1. Load model (cached)
    2. Fetch patient features + known medications
    3. Build candidate feature matrix (all meds patient hasn't received)
    4. Score with XGBoost
    5. Return top_k results with explanations
    6. Persist to ClickHouse recommendations table

    Returns list of dicts:
        [{"rank", "treatment_code", "treatment_name", "score", "explanation"}, ...]
    """
    bundle  = load_model()
    model   = bundle["model"]
    version = bundle["version"]

    # ── Fetch patient data ────────────────────────────────────────────────────
    patient_features = get_patient_features(patient_id)
    if patient_features is None:
        raise ValueError(f"Patient '{patient_id}' not found in patient_features")

    known_meds  = get_patient_known_medications(patient_id)
    catalogue   = get_medication_catalogue()
    prevalence  = get_global_medication_prevalence()

    all_codes   = catalogue["medication_code"].tolist()

    # ── Build candidate feature matrix ────────────────────────────────────────
    candidates_df = build_candidate_features(
        patient_features  = patient_features,
        candidate_codes   = all_codes,
        prevalence_map    = prevalence,
        known_meds        = known_meds,
        sim_med_count     = 0,       # 0 for serving — no similarity lookup needed
    )

    if candidates_df.empty:
        return []

    feature_cols = bundle.get("feature_cols", ALL_FEATURE_COLS)
    X = candidates_df[feature_cols].values.astype(np.float32)

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = model.predict_proba(X)[:, 1]
    candidates_df["score"] = scores

    # ── Rank and take top_k ───────────────────────────────────────────────────
    top_df = (
        candidates_df
        .sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # Merge medication names from catalogue
    name_map = dict(zip(catalogue["medication_code"], catalogue["medication_name"]))

    recommendations = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        code  = row["medication_code"]
        score = float(row["score"])
        name  = name_map.get(code, f"RxNorm {code}")
        expl  = _build_explanation(rank, score, patient_features, name)

        recommendations.append({
            "rank":           rank,
            "treatment_code": code,
            "treatment_name": name,
            "score":          score,
            "explanation":    expl,
        })

    # ── Persist to ClickHouse ─────────────────────────────────────────────────
    _persist_recommendations(patient_id, version, recommendations)

    return recommendations


# ── Persistence ───────────────────────────────────────────────────────────────

def _persist_recommendations(
    patient_id:      str,
    model_version:   str,
    recommendations: list[dict],
) -> None:
    """
    Insert recommendations into healthcare.recommendations table.
    Deletes any existing recommendations for this patient first
    so re-running is idempotent.
    """
    try:
        client = _get_ch_client()

        # Delete old recommendations for this patient
        client.command(f"""
            ALTER TABLE {CH_DB}.recommendations
            DELETE WHERE patient_id = '{patient_id}'
        """)

        rows = []
        for r in recommendations:
            rows.append({
                "patient_id":    patient_id,
                "model_version": model_version,
                "rank":          r["rank"],
                "treatment_code": r["treatment_code"],
                "treatment_name": r["treatment_name"],
                "score":         r["score"],
                "explanation":   r["explanation"],
            })

        df = pd.DataFrame(rows)
        client.insert_df(f"{CH_DB}.recommendations", df)

    except Exception as e:
        logger.warning("Failed to persist recommendations: %s", e)
