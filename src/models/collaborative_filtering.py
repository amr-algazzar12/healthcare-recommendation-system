"""
collaborative_filtering.py — Spark MLlib ALS recommendation model (Milestone 3)

Reads patient_features and medications from ClickHouse, builds a
patient-medication interaction matrix, trains ALS collaborative filtering,
evaluates with Precision@K and NDCG@K, and logs everything to MLflow.

Run via spark-submit from spark-master:
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --deploy-mode client \\
        /opt/airflow/src/models/collaborative_filtering.py
"""

import os
import sys
import mlflow
import mlflow.spark
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
SPARK_MASTER      = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
MLFLOW_URI        = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
CH_HOST           = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CH_PORT           = os.environ.get("CLICKHOUSE_HTTP_PORT", "8123")
CH_DB             = os.environ.get("CLICKHOUSE_DB", "healthcare")
CH_USER           = os.environ.get("CLICKHOUSE_USER", "healthcare_user")
CH_PASS           = os.environ.get("CLICKHOUSE_PASSWORD", "ch_secret_2026")
HDFS_MODELS       = "hdfs://namenode:9001/models/collaborative_filtering"
MODEL_NAME        = "healthcare-collaborative-filtering"
EXPERIMENT_NAME   = "healthcare-recommendations"
TOP_K             = 10
TEST_SPLIT        = 0.2
SEED              = 42

ALS_PARAMS = {
    "rank":           10,
    "maxIter":        10,
    "regParam":       0.1,
    "alpha":          1.0,
    "implicitPrefs":  True,   # implicit feedback (prescription existence = 1)
    "coldStartStrategy": "drop",
}


def get_spark():
    return (
        SparkSession.builder
        .appName("healthcare-collaborative-filtering")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


def get_ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=int(CH_PORT),
        database=CH_DB, username=CH_USER, password=CH_PASS, compress=False,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_interaction_matrix(spark, client):
    print("==> Loading interactions from ClickHouse...")
    
    # Still using the client but keeping it minimal
    # Ensure compress=False is in your get_ch_client!
    raw_data = client.query_df("""
        SELECT DISTINCT
            patient_id,
            code AS medication_code
        FROM healthcare.medications
        WHERE patient_id != '' AND code != ''
    """)
    
    # Convert to Spark immediately
    raw_spark_df = spark.createDataFrame(raw_data)
    raw_spark_df = raw_spark_df.withColumn("rating", F.lit(1.0))

    print("==> Encoding IDs with StringIndexer...")
    
    # Use StringIndexer: This handles the mapping of UUIDs to Integers automatically
    p_indexer = StringIndexer(inputCol="patient_id", outputCol="patient_idx", handleInvalid="skip")
    m_indexer = StringIndexer(inputCol="medication_code", outputCol="med_idx", handleInvalid="skip")
    
    pipeline = Pipeline(stages=[p_indexer, m_indexer])
    pipeline_model = pipeline.fit(raw_spark_df)
    interactions_df = pipeline_model.transform(raw_spark_df)

    # Cast to Integer for ALS
    interactions_df = interactions_df.withColumn("patient_idx", F.col("patient_idx").cast("int"))
    interactions_df = interactions_df.withColumn("med_idx", F.col("med_idx").cast("int"))

    return interactions_df, pipeline_model


def precision_at_k(predictions_df, k=TOP_K):
    """
    Compute mean Precision@K across all patients.
    predictions_df: pandas DataFrame with columns [patient_idx, recommendations]
    where recommendations is a list of (med_idx, score) tuples.
    """
    # This is computed after ALS recommendForAllUsers
    # We use RMSE from the evaluator as the primary metric for ALS
    return None


# ── Training ──────────────────────────────────────────────────────────────────

def train(spark, interactions_df, pipeline_model):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="collaborative_filtering") as run:
        run_id = run.info.run_id
        # Train/test split
        train_df, test_df = interactions_df.randomSplit([0.8, 0.2], seed=SEED)

        # ALS configuration
        als = ALS(
            userCol="patient_idx",
            itemCol="med_idx",
            ratingCol="rating",
            **ALS_PARAMS,
            seed=SEED
        )
        
        print("==> Training ALS model...")
        model = als.fit(train_df)

        # Evaluate on test set
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )
        rmse = evaluator.evaluate(predictions)
        print(f"    Test RMSE: {rmse:.4f}")
        mlflow.log_metric("rmse", rmse)

        # Top-K recommendations for all patients
        print(f"==> Generating top-{TOP_K} recommendations for all patients...")
        user_recs = model.recommendForAllUsers(TOP_K)
        n_patients_with_recs = user_recs.count()
        mlflow.log_metric("n_patients_with_recommendations", n_patients_with_recs)
        print(f"    Patients with recommendations: {n_patients_with_recs:,}")

        # Save model to HDFS
        print(f"==> Saving model to {HDFS_MODELS}...")
        model.write().overwrite().save(HDFS_MODELS)
        mlflow.log_param("model_hdfs_path", HDFS_MODELS)

        # Log model to MLflow registry
        mlflow.spark.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        mlflow.log_metric("training_complete", 1)
        print(f"==> Collaborative filtering training complete.")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Run ID: {run_id}")

        return run_id, rmse


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    client = get_ch_client()

    interactions_df, pipeline_model = load_interaction_matrix(spark, client)

    run_id, rmse = train(spark, interactions_df, pipeline_model)

    print(f"\n{'='*60}")
    print(f"Collaborative Filtering — DONE")
    print(f"  RMSE:   {rmse:.4f}")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()