"""
evaluate.py — Model evaluation and MLflow registry promotion (Milestone 3)

Compares all three models from the current MLflow experiment using their
logged metrics, picks the best performer, and promotes it to Production
in the MLflow Model Registry.

Ranking priority:
  1. NDCG@K (content-based)
  2. AUC-ROC (hybrid XGBoost)
  3. RMSE inverted (ALS — lower is better)

Run via spark-submit from spark-master (no Spark needed — pure Python):
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --deploy-mode client \\
        /opt/airflow/src/models/evaluate.py
"""

import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
EXPERIMENT_NAME = "healthcare-recommendations"

MODEL_REGISTRY = {
    "collaborative_filtering": "healthcare-collaborative-filtering",
    "content_based":           "healthcare-content-based",
    "hybrid_xgboost":          "healthcare-hybrid-xgboost",
}

# Primary metric per model (higher is better after normalisation)
# For ALS RMSE: we use 1 / (1 + rmse) so lower RMSE = higher score
PRIMARY_METRIC = {
    "collaborative_filtering": ("rmse",        "lower_better"),
    "content_based":           ("ndcg_at_k",   "higher_better"),
    "hybrid_xgboost":          ("auc_roc",     "higher_better"),
}


def get_latest_runs(client, experiment_name):
    """Get the most recent successful run per model type."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attribute.start_time DESC"],
    )

    # Take the latest run per model type
    latest = {}
    for run in runs:
        model_type = run.data.params.get("model_type", "")
        if not model_type:
            continue
        # Map model_type string to our key
        for key in MODEL_REGISTRY:
            if key in model_type or model_type in key:
                if key not in latest:
                    latest[key] = run
                break

    return latest


def score_run(model_key, run):
    """
    Compute a normalised score (higher = better) for ranking.
    """
    metrics = run.data.metrics
    metric_name, direction = PRIMARY_METRIC[model_key]

    value = metrics.get(metric_name)
    if value is None:
        return -1.0, None, None

    if direction == "lower_better":
        score = 1.0 / (1.0 + value)
    else:
        score = float(value)

    return score, metric_name, value


def promote_best_model(client, best_key, best_run):
    """
    Archive all current Production versions, then promote the best model.
    """
    registry_name = MODEL_REGISTRY[best_key]

    # Find the latest version of the best model
    try:
        versions = client.get_latest_versions(registry_name)
    except Exception as e:
        print(f"  WARN: Could not get versions for {registry_name}: {e}")
        return None

    if not versions:
        print(f"  WARN: No registered versions found for {registry_name}")
        return None

    # Archive all existing Production versions across all models
    for key, name in MODEL_REGISTRY.items():
        try:
            prod_versions = client.get_latest_versions(name, stages=["Production"])
            for v in prod_versions:
                client.transition_model_version_stage(
                    name=name,
                    version=v.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                print(f"  Archived: {name} v{v.version}")
        except Exception:
            pass

    # Promote latest version of best model to Production
    latest_version = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=registry_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"  Promoted: {registry_name} v{latest_version.version} → Production")

    # Tag the run as best
    client.set_tag(best_run.info.run_id, "best_model", "true")
    client.set_tag(best_run.info.run_id, "promoted_to_production", "true")

    return latest_version.version


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    print(f"\n{'='*60}")
    print(f"Model Evaluation & Registration")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"{'='*60}\n")

    # Get latest runs
    print("==> Fetching latest runs from MLflow...")
    latest_runs = get_latest_runs(client, EXPERIMENT_NAME)

    if not latest_runs:
        raise RuntimeError(
            "No finished runs found in MLflow. "
            "Run the three training scripts first."
        )

    print(f"    Found runs for: {list(latest_runs.keys())}")

    # Score all models
    print("\n==> Scoring models...")
    scores = {}
    report = {}

    for key, run in latest_runs.items():
        score, metric_name, metric_value = score_run(key, run)
        scores[key] = score
        report[key] = {
            "run_id":        run.info.run_id,
            "model_type":    key,
            "metric_name":   metric_name,
            "metric_value":  metric_value,
            "normalised_score": score,
        }

        # Print all metrics for this run
        print(f"\n  {key}:")
        print(f"    Run ID: {run.info.run_id}")
        for mname, mval in run.data.metrics.items():
            if mname != "training_complete":
                print(f"    {mname}: {mval:.4f}" if isinstance(mval, float) else f"    {mname}: {mval}")
        print(f"    Normalised score: {score:.4f} (via {metric_name})")

    # Pick best
    if not scores:
        raise RuntimeError("Could not score any model. Check MLflow for errors.")

    best_key   = max(scores, key=scores.get)
    best_run   = latest_runs[best_key]
    best_score = scores[best_key]

    print(f"\n==> Best model: {best_key} (score={best_score:.4f})")

    # Promote to Production
    print("==> Promoting best model to Production in MLflow registry...")
    version = promote_best_model(client, best_key, best_run)

    # Log evaluation summary as a new MLflow run
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation_summary") as eval_run:
        mlflow.log_param("best_model_type",   best_key)
        mlflow.log_param("best_model_name",   MODEL_REGISTRY[best_key])
        mlflow.log_param("best_model_version", str(version))
        mlflow.log_param("best_run_id",       best_run.info.run_id)
        mlflow.log_metric("best_normalised_score", best_score)

        for key, r in report.items():
            if r["metric_value"] is not None:
                mlflow.log_metric(f"{key}_score", r["normalised_score"])

        # Save full report as artifact
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "evaluation_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

    print(f"\n{'='*60}")
    print(f"Evaluation complete.")
    print(f"  Best model:    {best_key}")
    print(f"  Registry name: {MODEL_REGISTRY[best_key]}")
    print(f"  Version:       {version}")
    print(f"  Stage:         Production")
    print(f"{'='*60}\n")

    # Write result to a file for the DAG to read
    result = {
        "best_model": best_key,
        "registry_name": MODEL_REGISTRY[best_key],
        "version": str(version),
        "score": best_score,
    }
    result_path = "/opt/airflow/models/best_model.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result written to {result_path}")


if __name__ == "__main__":
    main()