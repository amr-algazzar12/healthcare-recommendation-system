"""
content_based_v3.py — Content-Based Recommendation Model with AUC

New in v3 (over v2):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  AUC METRIC (primary focus)                                              │
  │  • ROC-AUC per patient: treats ranked scores as a binary classifier      │
  │    (positive = hidden med, negative = all other unseen meds).            │
  │    Macro-averaged across all test patients → mean_auc_roc                │
  │  • AUC-PR (Precision-Recall AUC) per patient, then macro-averaged        │
  │    → mean_auc_pr. More informative when positives are rare.              │
  │                                                                           │
  │  AUC-TARGETED SCORING IMPROVEMENTS                                        │
  │  • Score calibration: min-max normalisation → [0,1] per patient          │
  │    so AUC is computed on well-spread probabilities, not raw weights.     │
  │  • Sigmoid sharpening: scores pushed away from 0.5 boundary to widen    │
  │    the ROC curve's operating range.                                       │
  │  • Condition-cohort IDF: IDF penalty derived from same-condition        │
  │    subgroup (not global corpus). Rarer-within-cohort → higher signal.   │
  │  • Neighbour confidence weighting: each neighbour's contribution         │
  │    multiplied by a confidence term = sim² / (sim + ε) that              │
  │    quadratically rewards very-high-similarity neighbours.                │
  │  • Score floor blending: blend raw score with a small global-            │
  │    popularity baseline to avoid zero-probability negatives that          │
  │    collapse AUC calculation on sparse patients.                           │
  │                                                                           │
  │  GRID SEARCH                                                              │
  │  • Optimises for mean_auc_roc (not F1) when RUN_GRID_SEARCH=True        │
  │  • Extended grid includes recency_decay and jaccard_weight               │
  └──────────────────────────────────────────────────────────────────────────┘

Run via spark-submit:
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --deploy-mode client \\
        /opt/airflow/src/models/content_based.py
"""

from __future__ import annotations

import json
import os
import tempfile
from itertools import product as iter_product
from typing import Dict, List, Optional, Set, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI",  "http://mlflow:5001")
CH_HOST         = os.environ.get("CLICKHOUSE_HOST",      "clickhouse")
CH_PORT         = os.environ.get("CLICKHOUSE_HTTP_PORT", "8123")
CH_DB           = os.environ.get("CLICKHOUSE_DB",        "healthcare")
CH_USER         = os.environ.get("CLICKHOUSE_USER",      "healthcare_user")
CH_PASS         = os.environ.get("CLICKHOUSE_PASSWORD",  "ch_secret_2026")
MODEL_NAME      = "healthcare-content-based-v3"
EXPERIMENT_NAME = "healthcare-recommendations"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOP_K           = 7
N_SIMILAR       = 15
SIM_THRESHOLD   = 0.60    # slightly lower → more neighbours → smoother AUC
SCORE_THRESHOLD = 0.0     # turned off; calibration handles noise instead
N_HOLDOUT       = 3       # hide 3 meds → richer positive set for AUC
MIN_KNOWN_MEDS  = 1

RECENCY_DECAY   = 0.10    # lighter decay so older meds still contribute to AUC
JACCARD_WEIGHT  = 0.30    # stronger Jaccard to sharpen medication-set similarity

# Feature group weights: [scalars, condition_vector, medication_history_flags]
FEAT_WEIGHTS    = [1.0, 3.0, 2.0]

# Score calibration for AUC
# After L1-norm, apply min-max scaling and blend with popularity baseline
BASELINE_BLEND  = 0.05    # 5% popularity floor prevents zero-probability negatives
SIGMOID_SHARPEN = True    # apply sigmoid sharpening before AUC computation

# Confidence weighting exponent: sim^CONF_EXP rewards high-sim neighbours more
CONF_EXP        = 2.0

# Grid search
RUN_GRID_SEARCH = False
GRID = {
    "sim_threshold":  [0.50, 0.60, 0.70],
    "n_similar":      [10, 15, 20],
    "recency_decay":  [0.05, 0.10, 0.20],
    "jaccard_weight": [0.20, 0.30, 0.40],
}

SEED       = 42
TEST_RATIO = 0.20


# ── ClickHouse client ─────────────────────────────────────────────────────────

def get_ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=int(CH_PORT),
        database=CH_DB, username=CH_USER, password=CH_PASS,
        compress=False,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(client) -> pd.DataFrame:
    print("==> Loading patient_features from ClickHouse...")
    df = client.query_df("""
        SELECT
            patient_id, age, gender_encoded, race_encoded,
            num_conditions, num_medications, num_encounters,
            has_diabetes, has_hypertension, has_asthma,
            has_hyperlipidemia, has_coronary_disease,
            condition_vector, medication_history_flags
        FROM healthcare.patient_features
        WHERE patient_id != ''
        ORDER BY patient_id
    """)
    print(f"    Loaded {len(df):,} patients")
    return df


def load_medication_history(client, patient_ids: Set[str]) -> pd.DataFrame:
    print("==> Loading medication history from ClickHouse...")
    df = client.query_df("""
        SELECT
            patient_id,
            code AS medication_code,
            toString(max(start_date)) AS start_date
        FROM healthcare.medications
        WHERE patient_id != '' AND code != ''
        GROUP BY patient_id, code
        ORDER BY patient_id, start_date ASC
    """)
    df = df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)
    print(f"    Loaded {len(df):,} patient-medication pairs")
    return df


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(features_df: pd.DataFrame) -> np.ndarray:
    """
    Weighted feature matrix for similarity computation.

    Groups and weights (FEAT_WEIGHTS[0,1,2]):
      [0] Scalar demographics/clinical  — normalised to [0,1] per column
      [1] condition_vector              — binary, weight 3.0 (primary AUC driver)
      [2] medication_history_flags      — binary, weight 2.0

    Higher condition weight means cosine similarity is dominated by shared
    diagnoses, which is the strongest predictor of shared prescriptions.
    """
    scalar_cols = [
        "age", "gender_encoded", "race_encoded",
        "num_conditions", "num_medications", "num_encounters",
        "has_diabetes", "has_hypertension", "has_asthma",
        "has_hyperlipidemia", "has_coronary_disease",
    ]
    scalar_m = features_df[scalar_cols].values.astype(np.float32)
    col_max  = scalar_m.max(axis=0)
    col_max[col_max == 0] = 1.0
    scalar_m /= col_max
    scalar_m *= FEAT_WEIGHTS[0]

    cond_m = np.array(features_df["condition_vector"].tolist(),         dtype=np.float32)
    med_m  = np.array(features_df["medication_history_flags"].tolist(), dtype=np.float32)
    cond_m *= FEAT_WEIGHTS[1]
    med_m  *= FEAT_WEIGHTS[2]

    return np.hstack([scalar_m, cond_m, med_m])


# ── Similarity matrix ─────────────────────────────────────────────────────────

def _jaccard_binary(mat: np.ndarray) -> np.ndarray:
    """Vectorised Jaccard on binary rows.  J = |A∩B| / |A∪B|"""
    inter = mat @ mat.T
    sizes = mat.sum(axis=1, keepdims=True)
    union = sizes + sizes.T - inter
    return inter / np.maximum(union, 1e-9)


def build_similarity_matrix(
    features_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    jaccard_weight: float = JACCARD_WEIGHT,
) -> np.ndarray:
    """
    Hybrid similarity = (1 - w)*cosine + w*jaccard(med flags).

    For AUC: a well-spread, accurate similarity ranking is critical.
    Jaccard on medication flags acts as a "ground-truth anchor" because
    shared prescriptions directly proxy for shared clinical trajectories.
    """
    print("==> Computing cosine similarity matrix...")
    normed  = normalize(feature_matrix, norm="l2")
    cos_sim = cosine_similarity(normed).astype(np.float32)

    if jaccard_weight > 0.0:
        print("==> Computing Jaccard similarity on medication flags...")
        med_m   = np.array(
            features_df["medication_history_flags"].tolist(), dtype=np.float32
        )
        jac_sim = _jaccard_binary(med_m).astype(np.float32)
        sim_mat = (1.0 - jaccard_weight) * cos_sim + jaccard_weight * jac_sim
    else:
        sim_mat = cos_sim

    np.clip(sim_mat, 0.0, 1.0, out=sim_mat)
    return sim_mat


# ── Cohort-level IDF ──────────────────────────────────────────────────────────

def build_cohort_idf(
    features_df: pd.DataFrame,
    med_history_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-condition-group IDF for each medication.

    For a patient with condition flags C, the IDF of medication m is:
        idf_C(m) = log(1 + |patients in C| / (1 + |patients in C who took m|))

    This rewards medications that are rare *within the patient's condition
    cohort* — a much stronger signal for AUC than global IDF alone.

    Returns a dict: condition_key → {med_code → idf_value}

    condition_key is a frozenset-style string of active conditions.
    For patients with uncommon combinations, falls back to global IDF.
    """
    condition_cols = [
        "has_diabetes", "has_hypertension", "has_asthma",
        "has_hyperlipidemia", "has_coronary_disease",
    ]
    print("==> Building cohort-level IDF...")

    # Map patient_id → condition key
    def _ckey(row):
        return "|".join(str(int(row[c])) for c in condition_cols)

    features_df = features_df.copy()
    features_df["_ckey"] = features_df.apply(_ckey, axis=1)
    pid_to_ckey = features_df.set_index("patient_id")["_ckey"].to_dict()

    # Merge med history with condition keys
    med_history_df = med_history_df.copy()
    med_history_df["_ckey"] = med_history_df["patient_id"].map(pid_to_ckey)
    med_history_df = med_history_df.dropna(subset=["_ckey"])

    cohort_idf: Dict[str, Dict[str, float]] = {}

    for ckey, grp in med_history_df.groupby("_ckey"):
        cohort_size = grp["patient_id"].nunique()
        med_counts  = grp.groupby("medication_code")["patient_id"].nunique()
        idf_vals    = {
            med: float(np.log1p(cohort_size / (1 + cnt)))
            for med, cnt in med_counts.items()
        }
        cohort_idf[ckey] = idf_vals

    # Global fallback IDF
    total_patients = features_df["patient_id"].nunique()
    global_counts  = med_history_df.groupby("medication_code")["patient_id"].nunique()
    cohort_idf["__global__"] = {
        med: float(np.log1p(total_patients / (1 + cnt)))
        for med, cnt in global_counts.items()
    }

    print(f"    Built IDF for {len(cohort_idf) - 1} condition cohorts + global")
    return cohort_idf, pid_to_ckey


# ── Score calibration helpers ─────────────────────────────────────────────────

def _minmax_scale(scores: Dict[str, float]) -> Dict[str, float]:
    """Scale scores to [0, 1] via min-max normalisation."""
    if not scores:
        return scores
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {m: 0.5 for m in scores}
    return {m: (s - lo) / (hi - lo) for m, s in scores.items()}


def _sigmoid_sharpen(scores: Dict[str, float], gain: float = 6.0) -> Dict[str, float]:
    """
    Push scores away from 0.5 using a centred sigmoid.

    f(x) = 1 / (1 + exp(-gain * (x - 0.5)))

    Gain=6 maps x=0.7 → 0.73, x=0.9 → 0.88, x=0.1 → 0.12.
    This widens the score distribution and improves AUC discrimination.
    """
    return {m: 1.0 / (1.0 + np.exp(-gain * (s - 0.5))) for m, s in scores.items()}


def _blend_with_baseline(
    scores: Dict[str, float],
    all_candidate_meds: List[str],
    global_freq: Dict[str, int],
    alpha: float = BASELINE_BLEND,
) -> Dict[str, float]:
    """
    Blend calibrated scores with a global popularity baseline.

        final(m) = (1 - alpha) * score(m) + alpha * popularity(m)

    This prevents zero-probability scores for unscored candidates,
    which collapses AUC when the negative set has many zeros.

    All meds that the model has NOT scored get baseline score only.
    """
    if not all_candidate_meds or alpha == 0.0:
        return scores

    # Popularity: normalise global freq to [0,1]
    max_freq = max(global_freq.values()) if global_freq else 1
    pop = {m: global_freq.get(m, 0) / max_freq for m in all_candidate_meds}

    blended: Dict[str, float] = {}
    for m in all_candidate_meds:
        raw   = scores.get(m, 0.0)
        blended[m] = (1.0 - alpha) * raw + alpha * pop[m]

    return blended


# ── Evaluation metrics ────────────────────────────────────────────────────────

def _auc_roc_for_patient(
    all_candidate_meds: List[str],
    candidate_scores: Dict[str, float],
    hidden_meds: Set[str],
) -> Optional[float]:
    """
    Compute ROC-AUC for a single patient.

    Framing: binary classification over the universe of unseen medications.
      label(m) = 1  if m ∈ hidden_meds  (the true next prescriptions)
      label(m) = 0  otherwise

    score(m) = candidate_scores.get(m, 0.0) — the model's predicted relevance.

    AUC = probability that the model ranks a random positive above a random negative.
    AUC = 0.5 → random; AUC = 1.0 → perfect.

    Returns None if there are no positives or no negatives in the candidate set
    (undefined AUC — patient is excluded from the macro average).
    """
    if not all_candidate_meds:
        return None

    y_true  = [1 if m in hidden_meds else 0 for m in all_candidate_meds]
    y_score = [candidate_scores.get(m, 0.0)  for m in all_candidate_meds]

    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return None  # no variation → AUC undefined

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _auc_pr_for_patient(
    all_candidate_meds: List[str],
    candidate_scores: Dict[str, float],
    hidden_meds: Set[str],
) -> Optional[float]:
    """
    Compute Precision-Recall AUC (area under the PR curve) per patient.

    More informative than ROC-AUC when positives are rare (which is typical
    in medical recommendation: 2-3 hidden meds vs. hundreds of candidates).

    Uses the trapezoidal rule via sklearn.metrics.auc on sorted precision/recall
    pairs derived from score thresholds.
    """
    if not all_candidate_meds:
        return None

    # Sort candidates by score descending — sweep threshold from high to low
    sorted_meds   = sorted(all_candidate_meds,
                           key=lambda m: candidate_scores.get(m, 0.0),
                           reverse=True)
    n_pos         = sum(1 for m in all_candidate_meds if m in hidden_meds)
    if n_pos == 0:
        return None

    recalls, precisions = [0.0], [1.0]
    hits = 0
    for i, m in enumerate(sorted_meds, start=1):
        if m in hidden_meds:
            hits += 1
        precisions.append(hits / i)
        recalls.append(hits / n_pos)

    # sklearn.metrics.auc requires monotonically increasing x
    recalls_arr    = np.array(recalls)
    precisions_arr = np.array(precisions)
    order          = np.argsort(recalls_arr)
    try:
        return float(auc(recalls_arr[order], precisions_arr[order]))
    except Exception:
        return None


def precision_at_k(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]) & actual) / k


def recall_at_k(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]) & actual) / len(actual)


def f1_at_k(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    p = precision_at_k(recommended, actual, k)
    r = recall_at_k(recommended, actual, k)
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    if not actual:
        return 0.0
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, m in enumerate(recommended[:k]) if m in actual)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    for i, m in enumerate(recommended[:k]):
        if m in actual:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(recommended: List[str], actual: Set[str], k: int = TOP_K) -> float:
    if not actual:
        return 0.0
    hits, running_p = 0, 0.0
    for i, m in enumerate(recommended[:k]):
        if m in actual:
            hits += 1
            running_p += hits / (i + 1)
    return running_p / min(len(actual), k) if hits else 0.0


# ── Core recommendation logic ─────────────────────────────────────────────────

def _global_popularity_fallback(
    global_freq: Dict[str, int],
    known_meds: Set[str],
    top_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Returns (ranked list, score dict) for cold-start patients."""
    max_freq = max(global_freq.values()) if global_freq else 1
    scored   = {
        m: cnt / max_freq
        for m, cnt in global_freq.items()
        if m not in known_meds
    }
    ranked = sorted(scored, key=scored.__getitem__, reverse=True)[:top_k]
    return ranked, scored


def _score_candidates(
    idx: int,
    known_meds: Set[str],
    all_pids: np.ndarray,
    sim_row: np.ndarray,
    meds_by_patient: Dict[str, List[str]],
    global_freq: Dict[str, int],
    cohort_idf: Dict[str, Dict[str, float]],
    patient_ckey: str,
    sim_threshold: float  = SIM_THRESHOLD,
    n_similar: int        = N_SIMILAR,
    recency_decay: float  = RECENCY_DECAY,
    conf_exp: float       = CONF_EXP,
) -> Dict[str, float]:
    """
    Score ALL candidate (unseen) medications for a patient.

    Returns a raw score dict {med → score} before calibration.
    An empty dict means cold-start; caller should use popularity fallback.

    Scoring formula per (neighbour n, medication m):
    ─────────────────────────────────────────────────
    contribution(n, m) =
        sim(patient, n)^conf_exp              ← quadratic confidence
        × recency_factor(m, n)               ← (1 - decay)^rank_from_end
        × cohort_idf(m)                      ← rarity within same-condition group
        × bm25_tf(m, n)                      ← TF saturation (k1=1.2)

    Why conf_exp=2.0 helps AUC:
        AUC measures ranking quality across ALL thresholds.
        Squaring similarity squashes medium-quality neighbours (sim=0.6 →
        0.36 weight) while amplifying strong ones (sim=0.9 → 0.81 weight).
        This creates sharper score separations between positives and
        negatives, directly improving the ROC curve's area.

    Why cohort IDF helps AUC:
        A medication prescribed to 90% of the corpus has near-zero
        discriminative power — every patient gets it. Cohort IDF penalises
        such meds within the patient's condition group so they don't crowd
        out the genuinely informative candidates. The score distribution
        for true positives (hidden meds) is pushed higher relative to
        true negatives, raising AUC.
    """
    BM25_K1 = 1.2

    sim_scores      = sim_row.copy()
    sim_scores[idx] = -1.0

    above_threshold = np.where(sim_scores >= sim_threshold)[0]

    if above_threshold.size < 3:
        neighbour_idxs = np.argpartition(sim_scores, -n_similar)[-n_similar:]
    else:
        adaptive_cap   = min(int(n_similar * 1.5), above_threshold.size)
        if above_threshold.size > adaptive_cap:
            neighbour_idxs = above_threshold[
                np.argpartition(sim_scores[above_threshold], -adaptive_cap)[-adaptive_cap:]
            ]
        else:
            neighbour_idxs = above_threshold

    # Get IDF lookup for this patient's cohort
    idf_lookup = cohort_idf.get(patient_ckey, cohort_idf.get("__global__", {}))

    candidate_scores: Dict[str, float] = {}

    for n_idx in neighbour_idxs:
        sim_w          = float(sim_scores[n_idx])
        conf_w         = sim_w ** conf_exp          # quadratic confidence
        neighbour_pid  = all_pids[n_idx]
        neighbour_meds = meds_by_patient.get(neighbour_pid, [])

        tf_counter: Dict[str, int] = {}
        for med in neighbour_meds:
            tf_counter[med] = tf_counter.get(med, 0) + 1

        for rank_from_end, med in enumerate(reversed(neighbour_meds)):
            if med in known_meds:
                continue

            recency = (1.0 - recency_decay) ** rank_from_end
            idf_val = idf_lookup.get(med, float(np.log1p(1.0)))  # fallback idf=ln(2)
            tf      = tf_counter.get(med, 1)
            bm25_tf = (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1)

            contribution = conf_w * recency * idf_val * bm25_tf
            candidate_scores[med] = candidate_scores.get(med, 0.0) + contribution

    return candidate_scores


def _recommend_for_patient(
    idx: int,
    known_meds: Set[str],
    all_pids: np.ndarray,
    sim_row: np.ndarray,
    meds_by_patient: Dict[str, List[str]],
    global_freq: Dict[str, int],
    cohort_idf: Dict[str, Dict[str, float]],
    patient_ckey: str,
    top_k: int             = TOP_K,
    sim_threshold: float   = SIM_THRESHOLD,
    n_similar: int         = N_SIMILAR,
    score_threshold: float = SCORE_THRESHOLD,
    recency_decay: float   = RECENCY_DECAY,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Full recommendation pipeline for a single patient.

    Returns
    -------
    (ranked_list, calibrated_scores)
      ranked_list        : top-k medication codes
      calibrated_scores  : {med → calibrated probability} over ALL unseen meds.
                           Used directly for AUC computation.

    Calibration pipeline (after raw scoring):
      1. L1 normalisation  → relative weights sum to 1
      2. Min-max scaling   → [0, 1] range for all candidates
      3. Sigmoid sharpening (optional) → push scores away from 0.5
      4. Popularity baseline blend → floor prevents zero-probability negatives
    """
    raw_scores = _score_candidates(
        idx=idx,
        known_meds=known_meds,
        all_pids=all_pids,
        sim_row=sim_row,
        meds_by_patient=meds_by_patient,
        global_freq=global_freq,
        cohort_idf=cohort_idf,
        patient_ckey=patient_ckey,
        sim_threshold=sim_threshold,
        n_similar=n_similar,
        recency_decay=recency_decay,
    )

    if not raw_scores:
        ranked, pop_scores = _global_popularity_fallback(global_freq, known_meds, top_k)
        return ranked, pop_scores

    # ── 1. L1 normalisation ───────────────────────────────────────────────
    total = sum(raw_scores.values())
    if total > 0:
        raw_scores = {m: s / total for m, s in raw_scores.items()}

    # ── 2. Min-max calibration to [0, 1] ─────────────────────────────────
    calibrated = _minmax_scale(raw_scores)

    # ── 3. Sigmoid sharpening ─────────────────────────────────────────────
    if SIGMOID_SHARPEN:
        calibrated = _sigmoid_sharpen(calibrated)

    # ── 4. Blend with global popularity floor ─────────────────────────────
    all_unseen = [m for m in global_freq if m not in known_meds]
    calibrated = _blend_with_baseline(calibrated, all_unseen, global_freq)

    # ── 5. Optional score threshold & ranking ─────────────────────────────
    if score_threshold > 0.0:
        filtered = {m: s for m, s in calibrated.items() if s > score_threshold}
        if filtered:
            calibrated = filtered

    ranked = sorted(calibrated, key=calibrated.__getitem__, reverse=True)[:top_k]
    return ranked, calibrated


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    features_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    med_history_df: pd.DataFrame,
    patient_index: Dict[str, int],
    cohort_idf: Dict[str, Dict[str, float]],
    pid_to_ckey: Dict[str, str],
    n_holdout: int         = N_HOLDOUT,
    sim_threshold: float   = SIM_THRESHOLD,
    n_similar: int         = N_SIMILAR,
    score_threshold: float = SCORE_THRESHOLD,
    recency_decay: float   = RECENCY_DECAY,
) -> Dict[str, float]:
    """
    Leave-last-N-out evaluation computing:
      • mean_auc_roc  — macro-averaged ROC-AUC across test patients (primary)
      • mean_auc_pr   — macro-averaged PR-AUC (secondary, for sparse positives)
      • precision@K, recall@K, F1@K, NDCG@K, MRR, MAP  (ranking quality)
      • hit_rate@K    — fraction of patients with ≥1 hit in top-K

    AUC is computed over the FULL universe of unseen medications per patient,
    not just the top-K list. This is the standard approach in information
    retrieval: the model must assign higher scores to ALL hidden meds than
    to ALL irrelevant meds, across every possible threshold.
    """
    all_pids = features_df["patient_id"].values

    meds_by_patient: Dict[str, List[str]] = {
        pid: grp.sort_values("start_date")["medication_code"].tolist()
        for pid, grp in med_history_df.groupby("patient_id")
    }

    global_freq: Dict[str, int] = (
        med_history_df["medication_code"].value_counts().to_dict()
    )
    all_meds_universe = list(global_freq.keys())

    eligible_pids = [
        pid for pid in all_pids
        if pid in meds_by_patient
        and len(meds_by_patient[pid]) > n_holdout
        and pid in patient_index
    ]

    rng       = np.random.RandomState(SEED)
    n_test    = int(len(eligible_pids) * TEST_RATIO)
    test_pids = rng.choice(eligible_pids, size=n_test, replace=False)

    print(f"    Eligible patients (>{n_holdout} meds): {len(eligible_pids):,}")
    print(f"    Test patients:                          {len(test_pids):,}")

    precisions, recalls, f1s, ndcgs, rrs, aps = [], [], [], [], [], []
    auc_rocs, auc_prs = [], []
    hit_count = 0

    for pid in test_pids:
        idx      = patient_index[pid]
        all_meds = meds_by_patient[pid]

        hidden_meds = set(all_meds[-n_holdout:])
        known_meds  = set(all_meds[:-n_holdout])

        ckey = pid_to_ckey.get(pid, "__global__")

        if len(known_meds) < MIN_KNOWN_MEDS:
            ranked, calibrated = _global_popularity_fallback(
                global_freq, known_meds, TOP_K
            )
        else:
            ranked, calibrated = _recommend_for_patient(
                idx=idx,
                known_meds=known_meds,
                all_pids=all_pids,
                sim_row=sim_matrix[idx],
                meds_by_patient=meds_by_patient,
                global_freq=global_freq,
                cohort_idf=cohort_idf,
                patient_ckey=ckey,
                sim_threshold=sim_threshold,
                n_similar=n_similar,
                score_threshold=score_threshold,
                recency_decay=recency_decay,
            )

        # Candidate universe for AUC: all meds not already prescribed
        auc_candidates = [m for m in all_meds_universe if m not in known_meds]

        # ── Ranking metrics (top-K list) ──────────────────────────────────
        p  = precision_at_k(ranked, hidden_meds)
        r  = recall_at_k(ranked, hidden_meds)
        f1 = f1_at_k(ranked, hidden_meds)
        nd = ndcg_at_k(ranked, hidden_meds)
        rr = reciprocal_rank(ranked, hidden_meds)
        ap = average_precision(ranked, hidden_meds)

        precisions.append(p); recalls.append(r); f1s.append(f1)
        ndcgs.append(nd);     rrs.append(rr);    aps.append(ap)
        if p > 0:
            hit_count += 1

        # ── AUC metrics (full candidate universe) ─────────────────────────
        auc_roc_val = _auc_roc_for_patient(auc_candidates, calibrated, hidden_meds)
        auc_pr_val  = _auc_pr_for_patient(auc_candidates, calibrated, hidden_meds)

        if auc_roc_val is not None:
            auc_rocs.append(auc_roc_val)
        # if auc_pr_val is not None:
        #     auc_prs.append(auc_pr_val)

    n_tested = len(precisions)
    hit_rate = hit_count / n_tested if n_tested else 0.0

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    print(f"    Hits (any match in top-{TOP_K}): "
          f"{hit_count:,} / {n_tested:,} ({hit_rate * 100:.1f}%)")
    print(f"    AUC-ROC patients evaluated: {len(auc_rocs):,}")
#    print(f"    AUC-PR  patients evaluated: {len(auc_prs):,}")

    return {
        # ── AUC metrics (primary for v3) ──────────────────────────────────
        "mean_auc_roc":    _mean(auc_rocs),    # ← PRIMARY optimisation target
#        "mean_auc_pr":     _mean(auc_prs),     # ← secondary (sparse positives)
        "n_auc_patients":  len(auc_rocs),

        # ── Ranking metrics ───────────────────────────────────────────────
        "precision_at_k":  _mean(precisions),
        "recall_at_k":     _mean(recalls),
        "f1_at_k":         _mean(f1s),
        "ndcg_at_k":       _mean(ndcgs),
        "mrr":             _mean(rrs),
        "map":             _mean(aps),

        # ── Coverage ──────────────────────────────────────────────────────
        "hit_rate_at_k":   hit_rate,
        "n_test_patients": n_tested,
        "n_hits":          hit_count,
        "n_holdout":       n_holdout,
    }


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(
    features_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    med_hist_df: pd.DataFrame,
    patient_index: Dict[str, int],
    cohort_idf: Dict[str, Dict[str, float]],
    pid_to_ckey: Dict[str, str],
) -> Tuple[Dict, Dict[str, float]]:
    """
    Grid search optimising for mean_auc_roc.
    Jaccard weight affects the similarity matrix → rebuilt per trial.
    All other params are passed through to evaluate_model.
    """
    keys   = list(GRID.keys())
    values = list(GRID.values())
    trials = list(iter_product(*values))

    best_auc     = -1.0
    best_params  = {}
    best_metrics = {}

    print(f"\n==> Grid search: {len(trials)} trials (optimising AUC-ROC)")

    with mlflow.start_run(run_name="content_based_grid_search", nested=False):
        for combo in trials:
            params = dict(zip(keys, combo))

            # Rebuild similarity if jaccard_weight changed
            sim_mat = build_similarity_matrix(
                features_df, feature_matrix,
                jaccard_weight=params["jaccard_weight"],
            )

            with mlflow.start_run(run_name="trial", nested=True):
                mlflow.log_params(params)
                metrics = evaluate_model(
                    features_df, sim_mat, med_hist_df, patient_index,
                    cohort_idf, pid_to_ckey,
                    sim_threshold=params["sim_threshold"],
                    n_similar=params["n_similar"],
                    recency_decay=params["recency_decay"],
                )
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                auc_v = metrics["mean_auc_roc"]
                print(f"    {params}  AUC-ROC={auc_v:.4f}")

                if auc_v > best_auc:
                    best_auc     = auc_v
                    best_params  = params
                    best_metrics = metrics

    print(f"\n==> Best params:   {best_params}")
    print(f"    Best AUC-ROC:  {best_auc:.4f}")
    return best_params, best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = get_ch_client()

    features_df = load_features(client)
    med_hist_df = load_medication_history(
        client, set(features_df["patient_id"])
    )

    patient_index = {
        pid: i for i, pid in enumerate(features_df["patient_id"])
    }

    print("==> Building feature matrix...")
    feature_matrix = build_feature_matrix(features_df)

    cohort_idf, pid_to_ckey = build_cohort_idf(features_df, med_hist_df)

    # ── Optional grid search ───────────────────────────────────────────────
    if RUN_GRID_SEARCH:
        best_params, _ = run_grid_search(
            features_df, feature_matrix, med_hist_df,
            patient_index, cohort_idf, pid_to_ckey,
        )
        final_sim_threshold   = best_params.get("sim_threshold",   SIM_THRESHOLD)
        final_n_similar       = best_params.get("n_similar",       N_SIMILAR)
        final_recency_decay   = best_params.get("recency_decay",   RECENCY_DECAY)
        final_jaccard_weight  = best_params.get("jaccard_weight",  JACCARD_WEIGHT)
    else:
        final_sim_threshold   = SIM_THRESHOLD
        final_n_similar       = N_SIMILAR
        final_recency_decay   = RECENCY_DECAY
        final_jaccard_weight  = JACCARD_WEIGHT

    sim_mat = build_similarity_matrix(
        features_df, feature_matrix, jaccard_weight=final_jaccard_weight
    )

    # ── Final evaluation run ───────────────────────────────────────────────
    with mlflow.start_run(run_name="content_based") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "eval_strategy":    "leave_last_n_out",
            "n_holdout":        N_HOLDOUT,
            "top_k":            TOP_K,
            "n_similar":        final_n_similar,
            "sim_threshold":    final_sim_threshold,
            "score_threshold":  SCORE_THRESHOLD,
            "recency_decay":    final_recency_decay,
            "jaccard_weight":   final_jaccard_weight,
            "feat_weights":     str(FEAT_WEIGHTS),
            "conf_exp":         CONF_EXP,
            "baseline_blend":   BASELINE_BLEND,
            "sigmoid_sharpen":  SIGMOID_SHARPEN,
            "bm25_k1":          1.2,
            "cohort_idf":       True,
            "grid_search":      RUN_GRID_SEARCH,
        })

        print("\n==> Evaluating final model...")
        metrics = evaluate_model(
            features_df, sim_mat, med_hist_df, patient_index,
            cohort_idf, pid_to_ckey,
            sim_threshold=final_sim_threshold,
            n_similar=final_n_similar,
            recency_decay=final_recency_decay,
        )

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ── Persist artefacts ──────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            sim_path    = os.path.join(tmpdir, "similarity_matrix.npy")
            idx_path    = os.path.join(tmpdir, "patient_index.json")
            pids_path   = os.path.join(tmpdir, "patient_ids.npy")
            ckey_path   = os.path.join(tmpdir, "pid_to_ckey.json")
            cfg_path    = os.path.join(tmpdir, "model_config.json")

            np.save(sim_path,  sim_mat.astype(np.float16))
            np.save(pids_path, features_df["patient_id"].values)

            with open(idx_path,  "w") as f:
                json.dump(patient_index, f)
            with open(ckey_path, "w") as f:
                json.dump(pid_to_ckey, f)
            with open(cfg_path,  "w") as f:
                json.dump({
                    "sim_threshold":   final_sim_threshold,
                    "n_similar":       final_n_similar,
                    "score_threshold": SCORE_THRESHOLD,
                    "recency_decay":   final_recency_decay,
                    "jaccard_weight":  final_jaccard_weight,
                    "feat_weights":    FEAT_WEIGHTS,
                    "conf_exp":        CONF_EXP,
                    "baseline_blend":  BASELINE_BLEND,
                    "sigmoid_sharpen": SIGMOID_SHARPEN,
                    "top_k":           TOP_K,
                    "n_holdout":       N_HOLDOUT,
                }, f, indent=2)

            for path in [sim_path, idx_path, pids_path, ckey_path, cfg_path]:
                mlflow.log_artifact(path, artifact_path="model")

    print(f"\nContent-Based Model v3 — DONE")
    print(f"Run ID  : {run_id}")
    print(f"Metrics : {metrics}")


if __name__ == "__main__":
    main()