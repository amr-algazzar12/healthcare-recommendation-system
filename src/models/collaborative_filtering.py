"""
ALS Collaborative Filtering — Production Version
"""

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline

HDFS = "hdfs://namenode:9001"
MODEL_PATH = f"{HDFS}/models/als_model"

def spark_session():
    return SparkSession.builder.appName("ALS").getOrCreate()

def main():
    spark = spark_session()

    df = spark.read.parquet(f"{HDFS}/data/processed/medications")

    interactions = df.select(
        "patient_id",
        F.col("code").alias("medication_id")
    ).dropDuplicates().withColumn("rating", F.lit(1.0))

    indexer = Pipeline(stages=[
        StringIndexer(inputCol="patient_id", outputCol="user_idx"),
        StringIndexer(inputCol="medication_id", outputCol="item_idx")
    ])

    model_index = indexer.fit(interactions)
    data = model_index.transform(interactions)

    train, test = data.randomSplit([0.8, 0.2], 42)

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        implicitPrefs=True,
        rank=20,
        maxIter=15,
        regParam=0.05,
        coldStartStrategy="drop"
    )

    model = als.fit(train)

    model.write().overwrite().save(MODEL_PATH)

    model_index.write().overwrite().save(f"{HDFS}/models/als_indexers")

    spark.stop()

if __name__ == "__main__":
    main()
