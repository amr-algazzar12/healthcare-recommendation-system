"""
clean.py — PySpark cleaning job 
================================================
- Reads RAW Synthea CSVs (uppercase columns)
- Standardizes column names → unified schema
- Cleans data
- Writes Parquet to HDFS (/data/processed/)
"""

import sys
import os
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [clean.py] %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

HDFS_BASE = "hdfs://namenode:9001"
RAW_BASE = f"{HDFS_BASE}/data/raw"
PROCESSED_BASE = f"{HDFS_BASE}/data/processed"

SPARK_MASTER = "spark://spark-master:7077"

# ─────────────────────────────────────────────────────────────
# Spark
# ─────────────────────────────────────────────────────────────

def get_spark():
    return (
        SparkSession.builder
        .appName("healthcare_clean")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

# ─────────────────────────────────────────────────────────────
# Column Standardization (KEY FIX 🔥)
# ─────────────────────────────────────────────────────────────

def standardize_columns(df: DataFrame) -> DataFrame:
    """Convert Synthea columns → unified lowercase schema"""

    mapping = {
        "Id": "patient_id",
        "PATIENT": "patient_id",
        "ENCOUNTER": "encounter_id",
        "START": "start_date",
        "STOP": "stop_date",
        "DATE": "date",
        "BIRTHDATE": "birthdate",
        "DEATHDATE": "deathdate",
        "GENDER": "gender",
        "RACE": "race",
        "ETHNICITY": "ethnicity",
        "CITY": "city",
        "STATE": "state",
        "ZIP": "zip",
        "LAT": "lat",
        "LON": "lon",
        "CODE": "code",
        "DESCRIPTION": "description",
        "VALUE": "value",
        "UNITS": "units",
        "TYPE": "type",
    }

    for old, new in mapping.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    return df

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def trim_strings(df):
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, F.trim(F.col(c)))
    return df


def fill_nulls(df):
    string_cols = [c for c, t in df.dtypes if t == "string"]
    df = df.fillna({c: "UNKNOWN" for c in string_cols})
    return df


# ─────────────────────────────────────────────────────────────
# Cleaning Functions
# ─────────────────────────────────────────────────────────────

def clean_patients(df):
    log.info("Cleaning patients...")

    df = standardize_columns(df)

    df = df.dropDuplicates(["patient_id"])
    df = df.withColumn("birthdate", F.to_date("birthdate"))

    df = df.filter(F.col("birthdate").isNotNull())

    df = trim_strings(df)
    df = fill_nulls(df)

    df = df.withColumn("gender", F.upper(F.col("gender")))

    return df


def clean_conditions(df):
    log.info("Cleaning conditions...")

    df = standardize_columns(df)

    df = df.withColumn("start_date", F.to_date("start_date"))
    df = df.withColumn("stop_date", F.to_date("stop_date"))

    df = df.filter(F.col("start_date").isNotNull())
    df = df.dropDuplicates(["patient_id", "code", "start_date"])

    df = trim_strings(df)
    df = fill_nulls(df)

    return df


def clean_medications(df):
    log.info("Cleaning medications...")

    df = standardize_columns(df)

    df = df.withColumn("start_date", F.to_date("start_date"))
    df = df.filter(F.col("start_date").isNotNull())

    df = df.dropDuplicates(["patient_id", "code", "start_date"])

    df = trim_strings(df)
    df = fill_nulls(df)

    return df


def clean_observations(df):
    log.info("Cleaning observations...")

    df = standardize_columns(df)

    df = df.withColumn("date", F.to_timestamp("date"))

    df = df.filter(F.col("date").isNotNull() & F.col("value").isNotNull())

    df = df.dropDuplicates(["patient_id", "code", "date", "value"])

    df = trim_strings(df)
    df = fill_nulls(df)

    return df


def clean_encounters(df):
    log.info("Cleaning encounters...")

    df = standardize_columns(df)

    df = df.withColumn("start_date", F.to_timestamp("start_date"))

    df = df.filter(F.col("start_date").isNotNull())

    df = df.dropDuplicates(["encounter_id"])

    df = trim_strings(df)
    df = fill_nulls(df)

    return df


def clean_procedures(df):
    log.info("Cleaning procedures...")

    df = standardize_columns(df)

    df = df.withColumn("start_date", F.to_date("start_date"))

    df = df.filter(F.col("start_date").isNotNull())

    df = df.dropDuplicates(["patient_id", "code", "start_date"])

    df = trim_strings(df)
    df = fill_nulls(df)

    df = df.withColumn("base_cost", F.col("base_cost").cast(DoubleType()))

    return df

# ─────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────

def read_csv(spark, table):
    path = f"{RAW_BASE}/{table}.csv"
    log.info(f"Reading {table}")
    return spark.read.option("header", "true").option("inferSchema", "true").csv(path)


def write_parquet(df, table):
    path = f"{PROCESSED_BASE}/{table}"
    log.info(f"Writing {table}")
    df.coalesce(4).write.mode("overwrite").parquet(path)

# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

PIPELINE = [
    ("patients", clean_patients),
    ("conditions", clean_conditions),
    ("medications", clean_medications),
    ("observations", clean_observations),
    ("encounters", clean_encounters),
    ("procedures", clean_procedures),
]


def run():
    spark = get_spark()

    failed = []

    for table, fn in PIPELINE:
        try:
            df = read_csv(spark, table)
            df = fn(df)
            write_parquet(df, table)
        except Exception as e:
            log.error(f"FAILED {table}: {e}")
            failed.append(table)

    spark.stop()

    if failed:
        sys.exit(1)

    log.info("ALL CLEANING DONE ✅")


if __name__ == "__main__":
    run()