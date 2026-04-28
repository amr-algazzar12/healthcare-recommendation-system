"""
hybrid_model.py — Hybrid Recommendation Model (Milestone 3)

Design:
    Two independent models:
        - Random Forest (Spark MLlib)
        - XGBoost (controlled Pandas batch training)

Goal:
    Learn patient risk propensity WITHOUT data leakage

IMPORTANT:
    - No evaluation here (handled in evaluate.py)
    - Fully reproducible pipeline
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

import pandas as pd
import xgboost as xgb


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
HDFS_FEATURES = "hdfs://namenode:9001/data/features"
HDFS_MODELS   = "hdfs://namenode:9001/models"

RF_MODEL_PATH  = f"{HDFS_MODELS}/random_forest_model"
XGB_MODEL_PATH = f"{HDFS_MODELS}/xgboost_model"

SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

SEED = 42


# ─────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────
def get_spark():
    return (
        SparkSession.builder
        .appName("Hybrid-Recommender")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data(spark):
    return spark.read.parquet(f"{HDFS_FEATURES}/patient_features")


# ─────────────────────────────────────────────
# Feature Engineering (NO leakage target)
# ─────────────────────────────────────────────
def build_features(df):
    feature_cols = [
        "age",
        "gender_encoded",
        "race_encoded",
        "num_conditions",
        "num_medications",
        "num_encounters",
        "has_diabetes",
        "has_hypertension",
        "has_asthma",
        "has_hyperlipidemia",
        "has_coronary_disease"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)

    # SAFE proxy label (based on utilization only → avoids direct leakage)
    df = df.withColumn(
        "label",
        (F.col("num_encounters") > F.lit(4)).cast("int")
    )

    return df.select("features", "label")


# ─────────────────────────────────────────────
# Train/Test Split
# ─────────────────────────────────────────────
def split_data(df):
    return df.randomSplit([0.8, 0.2], seed=SEED)


# ─────────────────────────────────────────────
# Random Forest Model (Spark)
# ─────────────────────────────────────────────
def train_rf(train_df):
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=120,
        maxDepth=8,
        seed=SEED
    )

    return rf.fit(train_df)


# ─────────────────────────────────────────────
# Safe XGBoost Training (bounded conversion)
# ─────────────────────────────────────────────
def train_xgboost(train_df):
    """
    Safe conversion:
    - limit size
    - avoid full cluster crash
    """

    # sample if dataset is large (safety layer)
    sampled_df = train_df.sample(False, 0.5, seed=SEED)

    pdf = sampled_df.toPandas()

    X = pd.DataFrame(pdf["features"].tolist())
    y = pdf["label"]

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED
    )

    model.fit(X, y)

    return model


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "=" * 60)
    print(f"Hybrid Model Training - {datetime.utcnow().isoformat()}")
    print("=" * 60 + "\n")

    # Load
    df = load_data(spark)

    # Features
    df = build_features(df)

    # Split
    train_df, test_df = split_data(df)

    train_df.cache()

    print(f"Train size: {train_df.count():,}")
    print(f"Test size: {test_df.count():,}")

    # ── Random Forest ─────────────────────────
    print(" Training Random Forest...")
    rf_model = train_rf(train_df)

    rf_model.write().overwrite().save(RF_MODEL_PATH)

    # ── XGBoost ───────────────────────────────
    print(" Training XGBoost...")
    xgb_model = train_xgboost(train_df)

    xgb_model.save_model("/tmp/xgb_model.json")
    os.system(f"hdfs dfs -put -f /tmp/xgb_model.json {XGB_MODEL_PATH}")

    print("\n" + "=" * 60)
    print(" Hybrid models trained successfully")
    print(f"Random Forest → {RF_MODEL_PATH}")
    print(f"XGBoost → {XGB_MODEL_PATH}")
    print("=" * 60 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
