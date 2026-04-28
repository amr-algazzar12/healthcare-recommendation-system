"""
Unified REAL Evaluation — NO FAKE DATA
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALSModel
import numpy as np

HDFS = "hdfs://namenode:9001"

K = 5

def spark_session():
    return SparkSession.builder.appName("Eval").getOrCreate()


# ───────── Ranking ─────────
def precision(a, p):
    return len(set(a) & set(p[:K])) / K

def recall(a, p):
    return len(set(a) & set(p[:K])) / max(len(a), 1)

def mapk(a, p):
    s, h = 0, 0
    for i, x in enumerate(p[:K]):
        if x in a:
            h += 1
            s += h / (i+1)
    return s / max(len(a), 1)


# ───────── ALS REAL ─────────
def eval_als(spark):
    model = ALSModel.load(f"{HDFS}/models/als_model")

    users = model.userFactors.select("id")

    recs = model.recommendForUserSubset(users, K)

    pred = recs.select(
        "id",
        F.expr("transform(recommendations, x -> x.item)").alias("p")
    )

    truth = spark.read.parquet(f"{HDFS}/data/processed/medications") \
        .select("patient_id", F.col("code").alias("item")) \
        .groupBy("patient_id") \
        .agg(F.collect_set("item").alias("a"))

    joined = truth.join(pred, truth.patient_id == pred.id)

    r = joined.rdd.map(lambda x: (
        precision(x.a, x.p),
        recall(x.a, x.p),
        mapk(x.a, x.p)
    )).collect()

    return np.mean(r, axis=0)


def main():
    spark = spark_session()

    p, r, m = eval_als(spark)

    print("\n=== ALS RESULTS ===")
    print("Precision@K:", p)
    print("Recall@K:", r)
    print("MAP@K:", m)

    spark.stop()

if __name__ == "__main__":
    main()
