"""
hybrid_model.py — XGBoost hybrid recommendation model (Milestone 3)

Frames recommendation as a binary classification problem:
  - Positive: (patient, medication) pairs that were actually prescribed
  - Negative: (patient, medication) pairs that were never prescribed (sampled)

Features per (patient, medication) pair:
  - All patient features (age, gender, conditions, etc.)
  - Medication frequency in similar patients (from content-based similarity)
  - Medication global prevalence

Reads from ClickHouse. Logs to MLflow.

Run via spark-submit from spark-master:
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --deploy-mode client \\
        /opt/airflow/src/models/hybrid_model.py
"""

import os
import json
import tempfile
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
CH_HOST         = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CH_PORT         = os.environ.get("CLICKHOUSE_HTTP_PORT", "8123")
CH_DB           = os.environ.get("CLICKHOUSE_DB", "healthcare")
CH_USER         = os.environ.get("CLICKHOUSE_USER", "healthcare_user")
CH_PASS         = os.environ.get("CLICKHOUSE_PASSWORD", "ch_secret_2026")
MODEL_NAME      = "healthcare-hybrid-xgboost"
EXPERIMENT_NAME = "healthcare-recommendations"
TOP_K           = 10
SEED            = 42
NEG_RATIO       = 3    # negative samples per positive
N_SIMILAR       = 10   # similar patients for content-based feature

XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     SEED,
    "n_jobs":           -1,
}


def get_ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=int(CH_PORT),
        database=CH_DB, username=CH_USER, password=CH_PASS, compress=False,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(client):
    print("==> Loading patient features from ClickHouse...")
    features_df = client.query_df("""
        SELECT
            patient_id,
            age, gender_encoded, race_encoded,
            num_conditions, num_medications, num_encounters,
            has_diabetes, has_hypertension, has_asthma,
            has_hyperlipidemia, has_coronary_disease,
            condition_vector, medication_history_flags
        FROM healthcare.patient_features
        WHERE patient_id != ''
        ORDER BY patient_id
    """)
    print(f"    Patients: {len(features_df):,}")

    print("==> Loading medication history from ClickHouse...")
    meds_df = client.query_df("""
        SELECT DISTINCT patient_id, code AS medication_code
        FROM healthcare.medications
        WHERE patient_id != '' AND code != ''
    """)
    meds_df = meds_df[meds_df["patient_id"].isin(features_df["patient_id"])]
    print(f"    Patient-medication pairs: {len(meds_df):,}")

    print("==> Loading medication prevalence...")
    med_prev = client.query_df("""
        SELECT code AS medication_code,
               count(DISTINCT patient_id) AS patient_count
        FROM healthcare.medications
        GROUP BY code
        ORDER BY patient_count DESC
    """)

    return features_df, meds_df, med_prev


def build_similarity_scores(features_df, N_SIMILAR):
    """Compute content-based similarity scores for the hybrid feature."""
    from sklearn.metrics.pairwise import cosine_similarity

    scalar_cols = [
        "age", "gender_encoded", "race_encoded",
        "num_conditions", "num_medications", "num_encounters",
        "has_diabetes", "has_hypertension", "has_asthma",
        "has_hyperlipidemia", "has_coronary_disease",
    ]
    scalar_m = features_df[scalar_cols].values.astype(np.float32)
    mx = scalar_m.max(axis=0); mx[mx == 0] = 1
    scalar_m = scalar_m / mx

    cond_m = np.array(features_df["condition_vector"].tolist(), dtype=np.float32)
    med_m  = np.array(features_df["medication_history_flags"].tolist(), dtype=np.float32)
    feat_m = np.hstack([scalar_m, cond_m, med_m])

    normed = normalize(feat_m, norm="l2")
    sim    = cosine_similarity(normed)
    return sim


# ── Dataset construction ──────────────────────────────────────────────────────

def build_training_dataset(features_df, meds_df, med_prev, sim_matrix):
    """
    Build (patient, medication, label) dataset.
    Positive: actual prescriptions = 1
    Negative: randomly sampled non-prescriptions = 0 (NEG_RATIO × positives)
    """
    np.random.seed(SEED)
    print("==> Building training dataset...")

    patient_index = {pid: i for i, pid in enumerate(features_df["patient_id"])}
    all_pids      = features_df["patient_id"].values
    all_meds      = meds_df["medication_code"].unique()

    # Positive samples
    positives = meds_df.copy()
    positives["label"] = 1

    # Med prevalence lookup
    prev_map = dict(zip(med_prev["medication_code"], med_prev["patient_count"]))
    total_patients = len(features_df)

    # Per-patient medication set for negative sampling
    meds_by_patient = (
        meds_df.groupby("patient_id")["medication_code"].apply(set).to_dict()
    )

    # Content-based feature: for each (patient, med), count how many of
    # the N_SIMILAR similar patients received that medication
    def sim_med_count(pid, med):
        if pid not in patient_index:
            return 0
        idx = patient_index[pid]
        sim_scores = sim_matrix[idx].copy()
        sim_scores[idx] = -1
        sim_idxs = np.argsort(sim_scores)[::-1][:N_SIMILAR]
        count = sum(
            1 for si in sim_idxs
            if med in meds_by_patient.get(all_pids[si], set())
        )
        return count

    # Scalar feature extractor
    scalar_cols = [
        "age", "gender_encoded", "race_encoded",
        "num_conditions", "num_medications", "num_encounters",
        "has_diabetes", "has_hypertension", "has_asthma",
        "has_hyperlipidemia", "has_coronary_disease",
    ]
    patient_scalars = {
        row["patient_id"]: row[scalar_cols].values.astype(np.float32)
        for _, row in features_df[["patient_id"] + scalar_cols].iterrows()
    }

    # Build rows
    rows = []
    n_positives = len(positives)
    n_negatives = n_positives * NEG_RATIO

    # Positive rows
    for _, row in positives.iterrows():
        pid  = row["patient_id"]
        med  = row["medication_code"]
        scalars = patient_scalars.get(pid, np.zeros(len(scalar_cols)))
        sim_cnt = sim_med_count(pid, med)
        prev    = prev_map.get(med, 0) / total_patients
        rows.append({
            **dict(zip(scalar_cols, scalars)),
            "sim_med_count": sim_cnt,
            "med_prevalence": prev,
            "label": 1,
        })

    # Negative rows — sample random (patient, med) pairs not in positives
    neg_count = 0
    sampled_pids = np.random.choice(all_pids, size=n_negatives * 2, replace=True)
    sampled_meds = np.random.choice(all_meds, size=n_negatives * 2, replace=True)

    for pid, med in zip(sampled_pids, sampled_meds):
        if neg_count >= n_negatives:
            break
        if med in meds_by_patient.get(pid, set()):
            continue   # skip actual prescriptions
        scalars = patient_scalars.get(pid, np.zeros(len(scalar_cols)))
        sim_cnt = sim_med_count(pid, med)
        prev    = prev_map.get(med, 0) / total_patients
        rows.append({
            **dict(zip(scalar_cols, scalars)),
            "sim_med_count": sim_cnt,
            "med_prevalence": prev,
            "label": 0,
        })
        neg_count += 1

    dataset = pd.DataFrame(rows)
    print(f"    Dataset: {len(dataset):,} rows  "
          f"({(dataset['label']==1).sum():,} pos, "
          f"{(dataset['label']==0).sum():,} neg)")
    return dataset


# ── Evaluation metrics ────────────────────────────────────────────────────────

def precision_at_k(recommended, actual, k=TOP_K):
    if not actual: return 0.0
    return len(set(recommended[:k]) & set(actual)) / k

def recall_at_k(recommended, actual, k=TOP_K):
    if not actual: return 0.0
    return len(set(recommended[:k]) & set(actual)) / len(actual)

def ndcg_at_k(recommended, actual, k=TOP_K):
    if not actual: return 0.0
    actual_set = set(actual)
    dcg  = sum(1/np.log2(i+2) for i, m in enumerate(recommended[:k]) if m in actual_set)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(dataset, features_df, meds_df, med_prev, sim_matrix):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_cols = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_cols].values
    y = dataset["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    with mlflow.start_run(run_name="hybrid_xgboost") as run:
        run_id = run.info.run_id
        print(f"==> MLflow run ID: {run_id}")

        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("model_type",     "hybrid_xgboost")
        mlflow.log_param("top_k",          TOP_K)
        mlflow.log_param("neg_ratio",      NEG_RATIO)
        mlflow.log_param("n_features",     len(feature_cols))
        mlflow.log_param("n_train",        len(X_train))
        mlflow.log_param("n_test",         len(X_test))
        mlflow.log_param("feature_names",  str(feature_cols))

        print("==> Training XGBoost classifier...")
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Binary classification metrics
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        avg_prec = average_precision_score(y_test, y_pred_proba)

        print(f"    AUC-ROC:           {auc_roc:.4f}")
        print(f"    Average Precision: {avg_prec:.4f}")

        mlflow.log_metric("auc_roc",           auc_roc)
        mlflow.log_metric("average_precision",  avg_prec)

        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print("    Top features:", top_features)
        for fname, fimp in top_features:
            mlflow.log_metric(f"importance_{fname}", float(fimp))

        # Log model to MLflow
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        mlflow.log_metric("training_complete", 1)

        print(f"\n{'='*60}")
        print(f"Hybrid XGBoost Model — DONE")
        print(f"  AUC-ROC:           {auc_roc:.4f}")
        print(f"  Average Precision: {avg_prec:.4f}")
        print(f"  Run ID: {run_id}")
        print(f"{'='*60}\n")

        return run_id, auc_roc, avg_prec


def main():
    client = get_ch_client()

    features_df, meds_df, med_prev = load_data(client)

    print("==> Computing similarity matrix for hybrid features...")
    sim_matrix = build_similarity_scores(features_df, N_SIMILAR)

    dataset = build_training_dataset(
        features_df, meds_df, med_prev, sim_matrix
    )

    run_id, auc_roc, avg_prec = train(
        dataset, features_df, meds_df, med_prev, sim_matrix
    )


if __name__ == "__main__":
    main()