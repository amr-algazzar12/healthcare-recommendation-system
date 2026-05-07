"""
als_grid_search.py — ALS hyperparameter grid search (Milestone 3)

Runs exhaustive grid search over ALS parameters, evaluates each
combination with ranking + classification metrics, and logs every
experiment to MLflow. Produces a ranked comparison table.

Reuses all infrastructure from collaborative_filtering.py:
  load_interaction_matrix, split_interactions,
  build_ground_truth, build_predictions, compute_ranking_metrics,
  _recall_at_k, evaluate

Run via spark-submit from spark-master:
    docker exec spark-master /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --deploy-mode client \\
        --executor-memory 2g \\
        --driver-memory 2g \\
        /opt/airflow/src/models/als_grid_search.py

Or via Makefile:
    make run-als-grid-search
"""

import os
import itertools
import tempfile
import time

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
from typing import List, Dict

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

# ── Re-use existing infrastructure ────────────────────────────────────────────
# collaborative_filtering.py is mounted at the same path — import directly.
import sys
sys.path.insert(0, "/opt/airflow/src/models")

from collaborative_filtering import (
    get_spark,
    get_ch_client,
    load_interaction_matrix,
    split_interactions,
    build_ground_truth,
    build_predictions,
    compute_ranking_metrics,
    _recall_at_k,
)

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
EXPERIMENT_NAME = "healthcare-als-grid-search"
MODEL_NAME      = "healthcare-collaborative-filtering"
HDFS_MODELS     = "hdfs://namenode:9001/models/collaborative_filtering"
RESULTS_CSV     = "/opt/airflow/models/als_grid_search_results.csv"

TOP_K = 7
SEED  = 42

ALS_BASE_PARAMS = {
    "implicitPrefs":     True,
    "coldStartStrategy": "drop",
}

PARAM_GRID = {
    "rank":     [32, 64, 128],
    "maxIter":  [15, 20],
    "regParam": [0.01, 0.05, 0.1],
    "alpha":    [20, 40, 80],
}

# Total combinations: 3 × 2 × 3 × 3 = 54


# ── 1. Generate parameter combinations ───────────────────────────────────────

def generate_param_combinations(param_grid: dict)-> List[Dict]:
    """
    Returns a list of dicts, one per combination.

    Example output for a 2-key grid:
        [{"rank": 32, "maxIter": 15}, {"rank": 32, "maxIter": 20}, ...]
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ── 2. Train ALS model ────────────────────────────────────────────────────────

def train_als_model(spark: SparkSession, train_df, params: dict):
    """
    Train a single ALS model with the given hyperparameters.
    Uses checkpointing to avoid stack overflow on long lineages.

    Parameters
    ----------
    train_df : cached Spark DataFrame with columns patient_idx, med_idx, rating
    params   : dict with rank, maxIter, regParam, alpha

    Returns
    -------
    Fitted ALS model
    """
    spark.sparkContext.setCheckpointDir(
        "hdfs://namenode:9001/tmp/spark-checkpoints"
    )

    als = ALS(
        userCol="patient_idx",
        itemCol="med_idx",
        ratingCol="rating",
        seed=SEED,
        checkpointInterval=10,
        **ALS_BASE_PARAMS,
        **params,
    )

    return als.fit(train_df)


# ── 3. Extended evaluation ────────────────────────────────────────────────────

def _hit_rate_at_k(joined_df, k: int) -> float:
    """
    Hit Rate@K = fraction of patients who have at least one hit in top-K.
    """
    row = (
        joined_df
        .select(
            F.when(
                F.size(F.array_intersect(
                    F.slice("predicted_items", 1, k), "actual_items"
                )) > 0,
                1
            ).otherwise(0).alias("hit")
        )
        .agg(F.mean("hit").alias("hit_rate"))
        .collect()[0]
    )
    return float(row["hit_rate"]) if row["hit_rate"] is not None else 0.0


def _f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mrr_at_k(joined_df, k: int) -> float:
    """
    Mean Reciprocal Rank@K = mean of (1 / rank_of_first_hit) over patients.
    Returns 0 if no patient has a hit.
    """
    # Compute per patient: rank of first hit (1-indexed) within top-K
    # We need to check each recommended position sequentially.
    # Approach: slice predicted to top-K, find first index where it's in actual.
    def first_hit_rank(predicted, actual):
        actual_set = set(actual)
        for i, item in enumerate(predicted[:k], start=1):
            if item in actual_set:
                return 1.0 / i
        return 0.0

    rdd_mrr = joined_df.rdd.map(
        lambda r: first_hit_rank(
            list(r["predicted_items"]), list(r["actual_items"])
        )
    )
    vals = rdd_mrr.collect()
    return float(np.mean(vals)) if vals else 0.0


def _auc_roc_at_k(joined_df, k: int, n_items: int) -> float:
    """
    Approximated AUC-ROC for implicit recommendation.

    For each patient:
      - positives = actual_items (relevant)
      - negatives = all other items not in actual_items
      - model score for an item = (K - rank + 1) / K if in top-K else 0

    AUC = P(score(positive) > score(negative)) averaged over patients
        = fraction of (pos, neg) pairs where positive ranks higher in top-K.
    """
    def patient_auc(predicted, actual):
        actual_set    = set(actual)
        predicted_top = predicted[:k]
        ranked_pos    = [i for i, m in enumerate(predicted_top) if m in actual_set]
        ranked_neg    = [i for i, m in enumerate(predicted_top) if m not in actual_set]
        n_pos  = len(actual_set)
        n_neg  = n_items - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        wins = 0
        for rp in ranked_pos:
            # Items ranked after rp in top-K that are negative
            wins += sum(1 for rn in ranked_neg if rn > rp)
            # Items NOT in top-K are all ranked below top-K items
            n_neg_not_in_top = n_neg - len(ranked_neg)
            wins += n_neg_not_in_top
        total_pairs = n_pos * n_neg
        return wins / total_pairs

    rdd_auc = joined_df.rdd.map(
        lambda r: patient_auc(list(r["predicted_items"]), list(r["actual_items"]))
    )
    vals = [v for v in rdd_auc.collect() if v is not None]
    return float(np.mean(vals)) if vals else 0.5


def _log_loss_at_k(joined_df, k: int) -> float:
    """
    Log loss treating recommendation score as predicted probability.
    Score = (K - rank + 1) / K for top-K items, 0 otherwise.
    Positive = item in actual; negative = item in top-K but not actual.
    """
    eps = 1e-7

    def patient_log_loss(predicted, actual):
        actual_set = set(actual)
        losses = []
        for rank, item in enumerate(predicted[:k], start=1):
            score = (k - rank + 1) / k
            score = min(max(score, eps), 1 - eps)
            label = 1.0 if item in actual_set else 0.0
            losses.append(-(label * np.log(score) + (1 - label) * np.log(1 - score)))
        return float(np.mean(losses)) if losses else 0.0

    rdd_ll = joined_df.rdd.map(
        lambda r: patient_log_loss(list(r["predicted_items"]), list(r["actual_items"]))
    )
    vals = rdd_ll.collect()
    return float(np.mean(vals)) if vals else 0.0


def evaluate_model(
    model,
    train_df,
    test_df,
    k: int = TOP_K,
    n_items: int = None,
) -> dict:
    """
    Full evaluation of a trained ALS model.

    Ranking metrics  : Precision@K, Recall@K, F1@K, Hit Rate@K, NDCG@K,
                       MAP@K, MRR@K
    Classification   : AUC-ROC (approx), Log Loss (approx)

    Parameters
    ----------
    model    : fitted ALS model
    train_df : Spark DataFrame — used to exclude training items from recs
    test_df  : Spark DataFrame — ground truth
    k        : recommendation cutoff
    n_items  : total item count (for AUC-ROC denominator); inferred if None

    Returns
    -------
    dict of all metrics
    """
    if n_items is None:
        n_items = train_df.select("med_idx").distinct().count()

    ground_truth = build_ground_truth(test_df)
    predictions  = build_predictions(model, train_df, k=k)

    joined = (
        predictions
        .join(ground_truth, on="patient_idx", how="inner")
        .select("predicted_items", "actual_items")
        .cache()
    )

    n_evaluated = joined.count()

    # ── Ranking metrics (Spark MLlib) ─────────────────────────────────────────
    metrics_rdd = joined.rdd.map(
        lambda r: (list(r["predicted_items"]), list(r["actual_items"]))
    )
    rm = RankingMetrics(metrics_rdd)

    precision  = rm.precisionAt(k)
    recall     = _recall_at_k(joined, k)
    f1         = _f1_at_k(precision, recall)
    hit_rate   = _hit_rate_at_k(joined, k)
    ndcg       = rm.ndcgAt(k)
    map_k      = rm.meanAveragePrecisionAt(k)
    mrr        = _mrr_at_k(joined, k)

    # ── Classification metrics (approx) ───────────────────────────────────────
    auc_roc    = _auc_roc_at_k(joined, k, n_items)
    log_loss   = _log_loss_at_k(joined, k)

    joined.unpersist()

    return {
        "n_evaluated":     n_evaluated,
        f"precision_at_{k}": precision,
        f"recall_at_{k}":    recall,
        f"f1_at_{k}":        f1,
        f"hit_rate_at_{k}":  hit_rate,
        f"ndcg_at_{k}":      ndcg,
        f"map_at_{k}":       map_k,
        f"mrr_at_{k}":       mrr,
        "auc_roc":           auc_roc,
        "log_loss":          log_loss,
    }


# ── 4. MLflow logging ─────────────────────────────────────────────────────────

def log_to_mlflow(params: dict, metrics: dict, run_name: str) -> str:
    """
    Logs one grid search run to MLflow.
    Returns the run_id.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "top_k": TOP_K,
            "seed":  SEED,
            **params,
            **ALS_BASE_PARAMS,
        })
        mlflow.log_metrics(metrics)
        return run.info.run_id


# ── 5. Grid search orchestration ─────────────────────────────────────────────

def run_grid_search(
    spark:           SparkSession,
    interactions_df,
    param_grid:      dict = PARAM_GRID,
    k:               int  = TOP_K,
) -> pd.DataFrame:
    """
    Full grid search loop.

    1. Split data once (shared across all runs).
    2. Iterate over every parameter combination.
    3. Train, evaluate, log to MLflow.
    4. Aggregate all results into a ranked DataFrame.

    Parameters
    ----------
    spark           : active SparkSession
    interactions_df : full interaction matrix (patient_idx, med_idx, rating)
    param_grid      : dict of {param_name: [values]}
    k               : recommendation cutoff

    Returns
    -------
    pd.DataFrame sorted by ndcg_at_k descending — one row per combination
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Single shared train/test split ────────────────────────────────────────
    print("==> Splitting data (shared across all runs)...")
    train_df, test_df = split_interactions(interactions_df, seed=SEED)
    train_df.cache()
    test_df.cache()

    n_items = train_df.select("med_idx").distinct().count()
    print(f"    Unique medications (item space): {n_items:,}")

    # ── Generate combinations ─────────────────────────────────────────────────
    combos = generate_param_combinations(param_grid)
    total  = len(combos)
    print(f"\n==> Starting grid search: {total} combinations\n")
    print(f"{'─'*72}")

    results = []

    for i, params in enumerate(combos, start=1):
        param_str  = " | ".join(f"{k_}={v}" for k_, v in params.items())
        run_name   = (
            f"als_grid_r{params['rank']}"
            f"_i{params['maxIter']}"
            f"_reg{params['regParam']}"
            f"_a{params['alpha']}"
        )

        print(f"\n[{i:>2}/{total}]  {param_str}")
        t0 = time.time()

        try:
            model   = train_als_model(spark, train_df, params)
            metrics = evaluate_model(model, train_df, test_df,
                                     k=k, n_items=n_items)
            run_id  = log_to_mlflow(params, metrics, run_name)
            elapsed = time.time() - t0

            ndcg_key = f"ndcg_at_{k}"
            prec_key = f"precision_at_{k}"
            print(
                f"    NDCG@{k}={metrics[ndcg_key]:.4f}  "
                f"Prec@{k}={metrics[prec_key]:.4f}  "
                f"AUC={metrics['auc_roc']:.4f}  "
                f"({elapsed:.0f}s)  run_id={run_id[:8]}..."
            )

            results.append({
                **params,
                **metrics,
                "run_id":      run_id,
                "elapsed_sec": round(elapsed, 1),
                "status":      "ok",
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    ERROR: {exc}  ({elapsed:.0f}s)")
            results.append({
                **params,
                "run_id":      "",
                "elapsed_sec": round(elapsed, 1),
                "status":      f"error: {exc}",
            })

    train_df.unpersist()
    test_df.unpersist()

    return pd.DataFrame(results)


# ── 6. Results presentation & best model promotion ────────────────────────────

def present_results(df: pd.DataFrame, k: int = TOP_K) -> dict:
    """
    Sort results by NDCG@K, print the comparison table,
    identify the best combination, and return its params + metrics.
    """
    ndcg_col = f"ndcg_at_{k}"
    metric_cols = [
        ndcg_col,
        f"precision_at_{k}",
        f"recall_at_{k}",
        f"f1_at_{k}",
        f"hit_rate_at_{k}",
        f"map_at_{k}",
        f"mrr_at_{k}",
        "auc_roc",
        "log_loss",
    ]
    param_cols = list(PARAM_GRID.keys())

    # Filter to successful runs only
    ok_df = df[df["status"] == "ok"].copy()

    if ok_df.empty:
        print("ERROR: no successful runs to present.")
        return {}

    ok_df = ok_df.sort_values(ndcg_col, ascending=False).reset_index(drop=True)

    # ── Print table ───────────────────────────────────────────────────────────
    display_cols = param_cols + [c for c in metric_cols if c in ok_df.columns]
    float_cols   = [c for c in display_cols if c in metric_cols]

    print(f"\n{'='*72}")
    print(f"GRID SEARCH RESULTS — sorted by NDCG@{k}  "
          f"({len(ok_df)}/{len(df)} runs succeeded)")
    print(f"{'='*72}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(ok_df[display_cols].to_string(index=True))

    # ── Best combination ──────────────────────────────────────────────────────
    best = ok_df.iloc[0]
    best_params  = {p: best[p] for p in param_cols}
    best_metrics = {m: best[m] for m in metric_cols if m in ok_df.columns}

    print(f"\n{'─'*72}")
    print(f"BEST COMBINATION (NDCG@{k} = {best[ndcg_col]:.4f})")
    print(f"{'─'*72}")
    for p, v in best_params.items():
        print(f"  {p:<12} {v}")
    print()
    for m, v in best_metrics.items():
        print(f"  {m:<25} {v:.4f}")

    print(f"\n  MLflow run ID: {best['run_id']}")
    print(f"  View at:       {MLFLOW_URI}/#/experiments")
    print(f"{'='*72}\n")

    return {
        "best_params":  best_params,
        "best_metrics": best_metrics,
        "best_run_id":  best["run_id"],
        "results_df":   ok_df,
    }


def save_results(df: pd.DataFrame, run_id: str) -> None:
    """
    Save full results CSV locally and as an MLflow artifact on the best run.
    """
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"==> Results saved to {RESULTS_CSV}")

    if run_id:
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            with mlflow.start_run(run_id=run_id):
                with tempfile.TemporaryDirectory() as tmpdir:
                    csv_path = os.path.join(tmpdir, "grid_search_results.csv")
                    df.to_csv(csv_path, index=False)
                    mlflow.log_artifact(csv_path, artifact_path="grid_search")
            print(f"==> Results CSV logged to MLflow run {run_id[:8]}...")
        except Exception as e:
            print(f"    WARN: Could not log CSV to MLflow: {e}")


def promote_best_model(
    spark: SparkSession,
    interactions_df,
    best_params: dict,
) -> None:
    """
    Retrain final model with best params on ALL data (no train/test split),
    save to HDFS, and register to MLflow model registry as Production.
    """
    print("\n==> Retraining best model on full dataset for production...")
    all_data = interactions_df.cache()

    model = train_als_model(spark, all_data, best_params)

    print(f"==> Saving production model to {HDFS_MODELS}...")
    model.write().overwrite().save(HDFS_MODELS)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="als_best_production") as run:
        mlflow.log_params({
            "top_k":   TOP_K,
            "seed":    SEED,
            "retrained_on": "full_dataset",
            **best_params,
            **ALS_BASE_PARAMS,
        })
        mlflow.log_param("model_hdfs_path", HDFS_MODELS)

        mlflow.spark.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        mlflow.log_metric("training_complete", 1)
        print(f"==> Production model registered as '{MODEL_NAME}'")
        print(f"    Run ID: {run.info.run_id}")

    all_data.unpersist()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    client = get_ch_client()

    interactions_df, _ = load_interaction_matrix(spark, client)
    interactions_df.cache()

    # ── Grid search ───────────────────────────────────────────────────────────
    results_df = run_grid_search(spark, interactions_df, PARAM_GRID, k=TOP_K)

    # ── Results ───────────────────────────────────────────────────────────────
    summary = present_results(results_df, k=TOP_K)

    if not summary:
        print("ERROR: grid search produced no valid results.")
        spark.stop()
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_results(results_df, run_id=summary["best_run_id"])

    # ── Promote best model to production ─────────────────────────────────────
    promote_best_model(spark, interactions_df, summary["best_params"])

    interactions_df.unpersist()
    spark.stop()

    print("==> Grid search complete.")


if __name__ == "__main__":
    main()