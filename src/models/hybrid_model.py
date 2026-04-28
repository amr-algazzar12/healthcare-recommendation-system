"""
Hybrid Model — RF + XGBoost (clean + safe)
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

import pandas as pd
import xgboost as xgb

HDFS = "hdfs://namenode:9001"
RF_PATH = f"{HDFS}/models/rf"
XGB_PATH = f"{HDFS}/models/xgb"

def spark_session():
    return SparkSession.builder.appName("Hybrid").getOrCreate()

def main():
    spark = spark_session()

    df = spark.read.parquet(f"{HDFS}/data/features/patient_features")

    cols = [
        "age","gender_encoded","race_encoded",
        "num_conditions","num_medications","num_encounters",
        "has_diabetes","has_hypertension","has_asthma",
        "has_hyperlipidemia","has_coronary_disease"
    ]

    df = VectorAssembler(inputCols=cols, outputCol="features").transform(df)

    df = df.withColumn(
        "label",
        (F.col("num_encounters") > 4).cast("int")
    )

    train, test = df.randomSplit([0.8, 0.2], 42)

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=8
    )

    rf_model = rf.fit(train)
    rf_model.write().overwrite().save(RF_PATH)

    pdf = train.toPandas()
    X = pd.DataFrame(pdf["features"].tolist())
    y = pdf["label"]

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        tree_method="hist",
        eval_metric="logloss"
    )

    xgb_model.fit(X, y)
    xgb_model.save_model("/tmp/xgb.json")

    spark._jvm.org.apache.hadoop.fs.FileSystem \
        .get(spark._jsc.hadoopConfiguration()) \
        .copyFromLocalFile(False, True,
            spark._jvm.org.apache.hadoop.fs.Path("file:/tmp/xgb.json"),
            spark._jvm.org.apache.hadoop.fs.Path(f"{XGB_PATH}")
        )

    spark.stop()

if __name__ == "__main__":
    main()
