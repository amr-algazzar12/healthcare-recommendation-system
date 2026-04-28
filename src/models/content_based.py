"""
content_based.py — Content-Based Recommendation Model (Milestone 3)

Goal:
    Build patient similarity model based on condition vectors
    using cosine similarity in a scalable Spark way.

Approach:
    - Vector normalization using Spark ML
    - Efficient similarity computation (Top-K only per patient)
    - Output stored for evaluation & hybrid model

Output:
    hdfs://namenode:9001/models/content_based_model

Note:
    Evaluation handled separately in evaluate.py
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Normalizer, VectorAssembler
from pyspark.ml.linalg import VectorUDT
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
# Load data
# ─────────────────────────────────────────────
def load_data(spark):
    return spark.read.parquet(f"{HDFS_FEATURES}/patient_features")


# ─────────────────────────────────────────────
# Build feature vector (NO UDF)
# ─────────────────────────────────────────────
def build_vectors(df):
    """
    Convert condition_vector array → Spark ML vector (efficient way)
    """

    assembler = VectorAssembler(
        inputCols=["condition_vector"],
        outputCol="features_vector"
    )

    return assembler.transform(df)


# ─────────────────────────────────────────────
# Normalize vectors
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
# Efficient Top-K similarity computation
# ─────────────────────────────────────────────
def compute_topk_similarity(df):
    """
    Avoid full O(n²) matrix by limiting output to Top-K neighbors
    """

    base = df.select("patient_id", "norm_vector")

    a = base.alias("a")
    b = base.alias("b")

    similarity = (
        a.crossJoin(b)
        .where(F.col("a.patient_id") != F.col("b.patient_id"))
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
# Main pipeline
# ─────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*60}")
    print(f"Content-Based Model Training - {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # ── Load features ─────────────────────────────
    print(" Loading features...")
    df = load_data(spark)

    # ── Vectorization ─────────────────────────────
    print(" Building feature vectors...")
    df = build_vectors(df)

    # ── Normalization ─────────────────────────────
    print(" Normalizing vectors...")
    df = normalize_vectors(df)

    # ── Similarity computation ────────────────────
    print(" Computing Top-K similarity...")
    result = compute_topk_similarity(df)

    print(f" Similarity pairs: {result.count():,}")

    # ── Save output ───────────────────────────────
    print(" Saving model artifacts...")
    result.write.mode("overwrite").parquet(MODEL_PATH)

    print(f"\n{'='*60}")
    print(" Content-Based model completed successfully")
    print(f"Saved at: {MODEL_PATH}")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()
