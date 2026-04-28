"""
content_based.py — Content-Based Recommendation Model (Milestone 3)

Idea:
    Recommend based on similarity between patients using condition vectors.

Approach:
    Cosine Similarity on condition_vector from feature engineering.

Output:
    - Similarity matrix (or nearest neighbors model)
    - Saved model artifacts in HDFS

NOTE:
    Evaluation is handled separately in evaluate.py
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.sql.types import FloatType

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
HDFS_FEATURES = "hdfs://namenode:9001/data/features"
HDFS_MODELS   = "hdfs://namenode:9001/models"

MODEL_PATH = f"{HDFS_MODELS}/content_based_model"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")


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
# Load feature data
# ─────────────────────────────────────────────
def load_data(spark):
    return spark.read.parquet(f"{HDFS_FEATURES}/patient_features")


# ─────────────────────────────────────────────
# Prepare vectors for cosine similarity
# ─────────────────────────────────────────────
def prepare_vectors(df):
    """
    Convert condition_vector array → Spark vector
    """

    def to_vector(arr):
        return Vectors.dense(arr)

    to_vector_udf = F.udf(to_vector, VectorUDT())

    df = df.withColumn(
        "features_vector",
        to_vector_udf(F.col("condition_vector"))
    )

    return df


# ─────────────────────────────────────────────
# Normalize vectors (for cosine similarity)
# ─────────────────────────────────────────────
def normalize_vectors(df):
    normalizer = Normalizer(
        inputCol="features_vector",
        outputCol="norm_vector",
        p=2.0
    )

    model = normalizer.fit(df)
    df = model.transform(df)

    return df


# ─────────────────────────────────────────────
# Build similarity model (self-join approach)
# ─────────────────────────────────────────────
def compute_similarity(df):
    """
    Compute cosine similarity between patients
    """

    df1 = df.select("patient_id", "norm_vector")
    df2 = df.select("patient_id", "norm_vector")

    similarity = (
        df1.alias("a")
        .join(df2.alias("b"))
        .where(F.col("a.patient_id") != F.col("b.patient_id"))
        .select(
            F.col("a.patient_id").alias("patient_a"),
            F.col("b.patient_id").alias("patient_b"),
            F.expr("dot(a.norm_vector, b.norm_vector)").alias("cosine_similarity")
        )
    )

    return similarity


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*60}")
    print(f"Content-Based Model Training - {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # ── Load features ─────────────────────────────
    print("✔ Loading patient features...")
    df = load_data(spark)

    # ── Prepare vectors ───────────────────────────
    print("✔ Building vectors...")
    df = prepare_vectors(df)

    # ── Normalize ─────────────────────────────────
    print("✔ Normalizing vectors...")
    df = normalize_vectors(df)

    # ── Compute similarity ────────────────────────
    print("✔ Computing cosine similarity...")
    similarity_df = compute_similarity(df)

    print(f"Similarity rows: {similarity_df.count():,}")

    # ── Save model ────────────────────────────────
    print("✔ Saving content-based model...")
    similarity_df.write.mode("overwrite").parquet(MODEL_PATH)

    print(f"\n{'='*60}")
    print("✔ Content-Based model completed successfully")
    print(f"Saved at: {MODEL_PATH}")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()