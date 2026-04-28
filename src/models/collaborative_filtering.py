"""
collaborative_filtering.py — ALS Recommendation Model (Milestone 3)

Purpose:
    Train ALS collaborative filtering model using implicit feedback.

Output:
    - ALS Model → hdfs://namenode:9001/models/als_model
    - Indexer Pipeline → hdfs://namenode:9001/models/als_indexers

Note:
    Evaluation is handled separately in evaluate.py
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
HDFS_PROCESSED = "hdfs://namenode:9001/data/processed"
HDFS_MODELS    = "hdfs://namenode:9001/models"

ALS_MODEL_PATH    = f"{HDFS_MODELS}/als_model"
ALS_INDEXERS_PATH = f"{HDFS_MODELS}/als_indexers"

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
# Build implicit interactions
# ─────────────────────────────────────────────
def build_interactions(medications):
    return (
        medications
        .select(
            F.col("patient_id"),
            F.col("code").alias("medication_id")
        )
        .dropDuplicates()
        .withColumn("interaction", F.lit(1.0))
    )


# ─────────────────────────────────────────────
# Train ALS model
# ─────────────────────────────────────────────
def train_als(train_df):
    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="interaction",
        implicitPrefs=True,
        rank=20,
        maxIter=15,
        regParam=0.05,
        alpha=1.0,
        coldStartStrategy="drop"
    )

    return als.fit(train_df)


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

    interactions = build_interactions(medications)

    print(f" Interactions: {interactions.count():,}")

    # ─────────────────────────────────────────
    # Indexing pipeline (users + items)
    # ─────────────────────────────────────────
    indexer = Pipeline(stages=[
        StringIndexer(
            inputCol="patient_id",
            outputCol="user_idx",
            handleInvalid="skip"
        ),
        StringIndexer(
            inputCol="medication_id",
            outputCol="item_idx",
            handleInvalid="skip"
        )
    ])

    print(" Fitting indexers...")
    indexer_model = indexer.fit(interactions)
    indexed_data = indexer_model.transform(interactions)

    # ── Train ALS ─────────────────────────────
    print(" Training ALS model...")
    als_model = train_als(indexed_data)

    # ── Save model ────────────────────────────
    print(" Saving ALS model...")
    als_model.write().overwrite().save(ALS_MODEL_PATH)

    # ── Save indexers ─────────────────────────
    print(" Saving indexer pipeline...")
    indexer_model.write().overwrite().save(ALS_INDEXERS_PATH)

    print(f"\n{'='*60}")
    print(" ALS training completed successfully")
    print(f"Model saved at: {ALS_MODEL_PATH}")
    print(f"Indexers saved at: {ALS_INDEXERS_PATH}")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()
