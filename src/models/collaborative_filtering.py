"""
collaborative_filtering.py — ALS Recommendation Model (Milestone 3)

Purpose:
    Train ALS collaborative filtering model using patient-medication interactions.

Output:
    - ALS Model → hdfs://namenode:9001/models/als_model
    - Indexers → hdfs://namenode:9001/models/als_indexers

NOTE:
    Evaluation is NOT done here.
    It will be handled centrally in evaluate.py.
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
HDFS_PROCESSED = "hdfs://namenode:9001/data/processed"
HDFS_MODELS    = "hdfs://namenode:9001/models"

ALS_MODEL_PATH      = f"{HDFS_MODELS}/als_model"
ALS_INDEXERS_PATH   = f"{HDFS_MODELS}/als_indexers"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")


# ─────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────
def get_spark():
    return (
        SparkSession.builder
        .appName("ALS-Collaborative-Filtering")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Build interactions
# ─────────────────────────────────────────────
def build_interactions(medications):
    """
    Convert medication usage into implicit interactions:
        patient_id → medication_id
    """
    interactions = (
        medications
        .select(
            F.col("patient_id"),
            F.col("code").alias("medication_id")
        )
        .dropDuplicates()
        .withColumn("interaction", F.lit(1))
    )
    return interactions


# ─────────────────────────────────────────────
# Encode categorical IDs
# ─────────────────────────────────────────────
def encode_ids(interactions):
    """
    Convert string IDs → numeric IDs for ALS
    Also return indexers for inference + evaluation
    """

    user_indexer = StringIndexer(
        inputCol="patient_id",
        outputCol="user_idx",
        handleInvalid="skip"
    )

    item_indexer = StringIndexer(
        inputCol="medication_id",
        outputCol="item_idx",
        handleInvalid="skip"
    )

    user_model = user_indexer.fit(interactions)
    item_model = item_indexer.fit(interactions)

    interactions = user_model.transform(interactions)
    interactions = item_model.transform(interactions)

    return interactions, user_model, item_model


# ─────────────────────────────────────────────
# Train ALS model
# ─────────────────────────────────────────────
def train_als(interactions):
    """
    Train ALS model using implicit feedback
    """

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="interaction",
        implicitPrefs=True,
        rank=20,
        maxIter=10,
        regParam=0.1,
        alpha=1.0,
        coldStartStrategy="drop"
    )

    model = als.fit(interactions)
    return model


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*60}")
    print(f"ALS Training Job - {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────
    medications = spark.read.parquet(f"{HDFS_PROCESSED}/medications")

    print("✔ Building interactions...")
    interactions = build_interactions(medications)

    print(f"Interactions: {interactions.count():,}")

    # ── Encode IDs ─────────────────────────────
    print("✔ Encoding IDs...")
    interactions, user_model, item_model = encode_ids(interactions)

    # ── Train model ───────────────────────────
    print("✔ Training ALS model...")
    als_model = train_als(interactions)

    # ── Save model ────────────────────────────
    print("✔ Saving ALS model...")
    als_model.write().overwrite().save(ALS_MODEL_PATH)

    # ── Save indexers (IMPORTANT) ─────────────
    print("✔ Saving indexers...")
    user_model.write().overwrite().save(f"{ALS_INDEXERS_PATH}/user_indexer")
    item_model.write().overwrite().save(f"{ALS_INDEXERS_PATH}/item_indexer")

    print(f"\n{'='*60}")
    print("✔ ALS training completed successfully")
    print(f"Model saved at: {ALS_MODEL_PATH}")
    print(f"Indexers saved at: {ALS_INDEXERS_PATH}")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()