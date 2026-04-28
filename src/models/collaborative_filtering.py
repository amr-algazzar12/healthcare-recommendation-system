"""
collaborative_filtering.py — ALS Recommendation Model (Production Milestone)

Goal:
    Train ALS collaborative filtering using implicit feedback.

Output:
    ALS Model + Indexers saved to HDFS

Note:
    No evaluation here (handled in evaluate.py)
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline


# ─────────────────────────────
# Paths
# ─────────────────────────────
HDFS_PROCESSED = "hdfs://namenode:9001/data/processed"
HDFS_MODELS    = "hdfs://namenode:9001/models"

ALS_MODEL_PATH    = f"{HDFS_MODELS}/als_model"
ALS_INDEXERS_PATH = f"{HDFS_MODELS}/als_indexers"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")


# ─────────────────────────────
def get_spark():
    return (
        SparkSession.builder
        .appName("ALS-CF")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ─────────────────────────────
def build_interactions(df):
    return (
        df.select(
            "patient_id",
            F.col("code").alias("medication_id")
        )
        .dropDuplicates()
        .withColumn("interaction", F.lit(1.0))
    )


# ─────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n=== ALS Training Started ===")

    meds = spark.read.parquet(f"{HDFS_PROCESSED}/medications")
    interactions = build_interactions(meds)

    indexer = Pipeline(stages=[
        StringIndexer(inputCol="patient_id", outputCol="user_idx", handleInvalid="skip"),
        StringIndexer(inputCol="medication_id", outputCol="item_idx", handleInvalid="skip")
    ])

    index_model = indexer.fit(interactions)
    data = index_model.transform(interactions)

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="interaction",
        implicitPrefs=True,
        rank=20,
        maxIter=15,
        regParam=0.05,
        coldStartStrategy="drop"
    )

    model = als.fit(data)

    model.write().overwrite().save(ALS_MODEL_PATH)
    index_model.write().overwrite().save(ALS_INDEXERS_PATH)

    print("ALS Model Saved")

    spark.stop()


if __name__ == "__main__":
    main()
