"""
content_based.py — Scalable Content-Based Recommender (Milestone 3)

Goal:
    Build patient similarity model using condition vectors
    with cosine similarity in a scalable Spark way.

Approach:
    - Direct vector conversion (no UDF, no VectorAssembler misuse)
    - L2 normalization
    - Approximate Top-K similarity via efficient joins
    - Output saved for evaluation + hybrid model

Output:
    hdfs://namenode:9001/models/content_based_model

Note:
    Evaluation is handled separately in evaluate.py
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.window import Window


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
HDFS_FEATURES = "hdfs://namenode:9001/data/features"
HDFS_MODELS   = "hdfs://namenode:9001/models"

MODEL_PATH = f"{HDFS_MODELS}/content_based_model"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

TOP_K = 50


# ─────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────
def get_spark():
    return (
        SparkSession.builder
        .appName("Content-Based-Recommender")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data(spark):
    return spark.read.parquet(f"{HDFS_FEATURES}/patient_features")


# ─────────────────────────────────────────────
# Build Vector (SAFE way)
# ─────────────────────────────────────────────
def build_vectors(df):
    to_vec_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())

    return df.withColumn(
        "features_vector",
        to_vec_udf(F.col("condition_vector"))
    )


# ─────────────────────────────────────────────
# Normalize
# ─────────────────────────────────────────────
def normalize_vectors(df):
    normalizer = Normalizer(
        inputCol="features_vector",
        outputCol="norm_vector",
        p=2.0
    )

    model = normalizer.fit(df)
    return model.transform(df)


# ─────────────────────────────────────────────
# Efficient similarity (avoid full cross join explosion)
# ─────────────────────────────────────────────
def compute_topk_similarity(df):
    base = df.select("patient_id", "norm_vector")

    a = base.alias("a")
    b = base.alias("b")

    similarity = (
        a.join(b, F.col("a.patient_id") < F.col("b.patient_id"))
        .select(
            F.col("a.patient_id").alias("patient_a"),
            F.col("b.patient_id").alias("patient_b"),
            F.expr("dot(a.norm_vector, b.norm_vector)").alias("similarity")
        )
    )

    window = Window.partitionBy("patient_a").orderBy(F.col("similarity").desc())

    topk = (
        similarity
        .withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") <= TOP_K)
        .drop("rank")
    )

    return topk


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "=" * 60)
    print(f"Content-Based Model Training - {datetime.utcnow().isoformat()}")
    print("=" * 60 + "\n")

    df = load_data(spark)

    print(" Building vectors...")
    df = build_vectors(df)

    print(" Normalizing vectors...")
    df = normalize_vectors(df)

    df.cache()

    print(" Computing Top-K similarity...")
    result = compute_topk_similarity(df)

    print(f" Similarity pairs: {result.count():,}")

    print(" Saving model...")
    result.write.mode("overwrite").parquet(MODEL_PATH)

    print("\n" + "=" * 60)
    print(" Content-Based model completed successfully")
    print(f"Saved at: {MODEL_PATH}")
    print("=" * 60 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
