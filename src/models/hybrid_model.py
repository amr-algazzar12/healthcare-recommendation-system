"""
hybrid_model.py — Hybrid Model (RF + XGBoost)

Goal:
    Predict patient risk propensity (binary classification)
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

import pandas as pd
import xgboost as xgb


HDFS_FEATURES = "hdfs://namenode:9001/data/features"
HDFS_MODELS   = "hdfs://namenode:9001/models"

RF_MODEL_PATH  = f"{HDFS_MODELS}/random_forest_model"
XGB_MODEL_PATH = f"{HDFS_MODELS}/xgboost_model"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

SEED = 42


def get_spark():
    return (
        SparkSession.builder
        .appName("Hybrid")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_features(df):
    cols = [
        "age", "gender_encoded", "race_encoded",
        "num_conditions", "num_medications", "num_encounters",
        "has_diabetes", "has_hypertension",
        "has_asthma", "has_hyperlipidemia", "has_coronary_disease"
    ]

    df = VectorAssembler(inputCols=cols, outputCol="features").transform(df)

    df = df.withColumn(
        "label",
        (F.col("num_encounters") > 4).cast("int")
    )

    return df.select("features", "label")


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(f"{HDFS_FEATURES}/patient_features")

    df = build_features(df)

    train = df.randomSplit([0.8, 0.2], seed=SEED)[0]

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=120,
        maxDepth=8,
        seed=SEED
    )

    rf_model = rf.fit(train)
    rf_model.write().overwrite().save(RF_MODEL_PATH)

    pdf = train.toPandas()
    X = pd.DataFrame(pdf["features"].tolist())
    y = pdf["label"]

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        eval_metric="logloss"
    )

    xgb_model.fit(X, y)
    xgb_model.save_model("/tmp/xgb.json")

    os.system(f"hdfs dfs -put -f /tmp/xgb.json {XGB_MODEL_PATH}")

    print("Hybrid Models Saved")

    spark.stop()


if __name__ == "__main__":
    main()
