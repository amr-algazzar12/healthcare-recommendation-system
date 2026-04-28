"""
content_based.py — Scalable Content-Based Model

Goal:
    Patient similarity using cosine similarity (Top-K only)
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.window import Window


HDFS_FEATURES = "hdfs://namenode:9001/data/features"
HDFS_MODELS   = "hdfs://namenode:9001/models"

MODEL_PATH = f"{HDFS_MODELS}/content_based_model"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

TOP_K = 50


def get_spark():
    return (
        SparkSession.builder
        .appName("Content-Based")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(f"{HDFS_FEATURES}/patient_features")

    to_vec = F.udf(lambda x: Vectors.dense(x), VectorUDT())
    df = df.withColumn("features_vector", to_vec("condition_vector"))

    normalizer = Normalizer(inputCol="features_vector", outputCol="norm_vector", p=2.0)
    df = normalizer.transform(df)

    a = df.alias("a")
    b = df.alias("b")

    sim = (
        a.join(b, F.col("a.patient_id") < F.col("b.patient_id"))
        .select(
            F.col("a.patient_id").alias("patient_a"),
            F.col("b.patient_id").alias("patient_b"),
            F.expr("dot(a.norm_vector, b.norm_vector)").alias("similarity")
        )
    )

    window = Window.partitionBy("patient_a").orderBy(F.col("similarity").desc())

    topk = (
        sim.withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") <= TOP_K)
        .drop("rank")
    )

    topk.write.mode("overwrite").parquet(MODEL_PATH)

    print("Content Model Saved")

    spark.stop()


if __name__ == "__main__":
    main()
