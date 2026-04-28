"""
Content-Based — REAL cosine similarity Top-K
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.window import Window

HDFS = "hdfs://namenode:9001"
MODEL_PATH = f"{HDFS}/models/content_based_model"
TOP_K = 20

def spark_session():
    return SparkSession.builder.appName("Content").getOrCreate()

def main():
    spark = spark_session()

    df = spark.read.parquet(f"{HDFS}/data/features/patient_features")

    to_vec = F.udf(lambda x: Vectors.dense(x), VectorUDT())
    df = df.withColumn("vec", to_vec("condition_vector"))

    norm = Normalizer(inputCol="vec", outputCol="norm")
    df = norm.transform(df)

    a = df.alias("a")
    b = df.alias("b")

    sim = (
        a.join(b, F.col("a.patient_id") < F.col("b.patient_id"))
        .select(
            F.col("a.patient_id").alias("a"),
            F.col("b.patient_id").alias("b"),
            F.expr("dot(a.norm, b.norm)").alias("score")
        )
    )

    w = Window.partitionBy("a").orderBy(F.desc("score"))

    topk = sim.withColumn("rn", F.row_number().over(w)) \
              .filter(F.col("rn") <= TOP_K) \
              .drop("rn")

    topk.write.mode("overwrite").parquet(MODEL_PATH)

    spark.stop()

if __name__ == "__main__":
    main()
