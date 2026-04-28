"""
clean.py — PySpark data cleaning job (Milestone 2)

Reads raw Synthea CSVs from HDFS, renames columns to match the
ClickHouse schema exactly, applies cleaning rules, and writes
cleaned Parquet to hdfs://namenode:9001/data/processed/

Column mapping verified against:
  - HDFS CSV headers (head -2 of each file)
  - ClickHouse system.columns for database=healthcare

Run via:
    make run-clean
    (docker exec spark-master /opt/spark/bin/spark-submit ...)
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

HDFS_RAW       = "hdfs://namenode:9001/data/raw"
HDFS_PROCESSED = "hdfs://namenode:9001/data/processed"
SPARK_MASTER   = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")


def get_spark():
    return (
        SparkSession.builder
        .appName("healthcare-clean")
        .master(SPARK_MASTER)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ── PATIENTS ──────────────────────────────────────────────────────────────────
# CSV:  Id, BIRTHDATE, DEATHDATE, SSN, DRIVERS, PASSPORT, PREFIX, FIRST,
#       LAST, SUFFIX, MAIDEN, MARITAL, RACE, ETHNICITY, GENDER, BIRTHPLACE,
#       ADDRESS, CITY, STATE, COUNTY, FIPS, ZIP, LAT, LON,
#       HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, INCOME
# CH:   patient_id, birthdate, deathdate, ssn, first, last, gender, race,
#       ethnicity, city, state, zip, lat, lon,
#       healthcare_expenses, healthcare_coverage
def clean_patients(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("Id",                  "patient_id")
        .withColumnRenamed("BIRTHDATE",           "birthdate")
        .withColumnRenamed("DEATHDATE",           "deathdate")
        .withColumnRenamed("SSN",                 "ssn")
        .withColumnRenamed("FIRST",               "first")
        .withColumnRenamed("LAST",                "last")
        .withColumnRenamed("GENDER",              "gender")
        .withColumnRenamed("RACE",                "race")
        .withColumnRenamed("ETHNICITY",           "ethnicity")
        .withColumnRenamed("CITY",                "city")
        .withColumnRenamed("STATE",               "state")
        .withColumnRenamed("ZIP",                 "zip")
        .withColumnRenamed("LAT",                 "lat")
        .withColumnRenamed("LON",                 "lon")
        .withColumnRenamed("HEALTHCARE_EXPENSES", "healthcare_expenses")
        .withColumnRenamed("HEALTHCARE_COVERAGE", "healthcare_coverage")
    )

    keep = ["patient_id", "birthdate", "deathdate", "ssn", "first", "last",
            "gender", "race", "ethnicity", "city", "state", "zip",
            "lat", "lon", "healthcare_expenses", "healthcare_coverage"]
    df = df.select(keep)

    df = df.dropna(subset=["patient_id", "birthdate", "gender"])
    df = df.filter(F.col("patient_id") != "")

    df = (
        df
        .withColumn("birthdate",           F.to_date("birthdate", "yyyy-MM-dd"))
        .withColumn("deathdate",           F.to_date("deathdate", "yyyy-MM-dd"))
        .withColumn("lat",                 F.col("lat").cast(DoubleType()))
        .withColumn("lon",                 F.col("lon").cast(DoubleType()))
        .withColumn("healthcare_expenses", F.col("healthcare_expenses").cast(DoubleType()))
        .withColumn("healthcare_coverage", F.col("healthcare_coverage").cast(DoubleType()))
        .withColumn("gender",              F.upper(F.col("gender")))
    )

    df = df.filter(F.col("birthdate") > F.lit("1900-01-01").cast("date"))
    df = df.dropDuplicates(["patient_id"])
    return df


# ── CONDITIONS ────────────────────────────────────────────────────────────────
# CSV:  START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION
# CH:   start_date, stop_date, patient_id, encounter_id, code, description
# Dates: plain yyyy-MM-dd
def clean_conditions(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("START",       "start_date")
        .withColumnRenamed("STOP",        "stop_date")
        .withColumnRenamed("PATIENT",     "patient_id")
        .withColumnRenamed("ENCOUNTER",   "encounter_id")
        .withColumnRenamed("CODE",        "code")
        .withColumnRenamed("DESCRIPTION", "description")
    )

    df = df.dropna(subset=["patient_id", "code", "start_date"])
    df = df.filter((F.col("patient_id") != "") & (F.col("code") != ""))

    df = (
        df
        .withColumn("start_date", F.to_date("start_date", "yyyy-MM-dd"))
        .withColumn("stop_date",  F.to_date("stop_date",  "yyyy-MM-dd"))
    )

    df = df.filter(F.col("start_date") > F.lit("1900-01-01").cast("date"))
    df = df.dropDuplicates(["patient_id", "code", "start_date"])
    return df


# ── MEDICATIONS ───────────────────────────────────────────────────────────────
# CSV:  START, STOP, PATIENT, PAYER, ENCOUNTER, CODE, DESCRIPTION,
#       BASE_COST, PAYER_COVERAGE, DISPENSES, TOTALCOST,
#       REASONCODE, REASONDESCRIPTION
# CH:   start_date, stop_date, patient_id, encounter_id, code, description,
#       reasoncode, reasondescription
# Dates: ISO timestamp "2020-09-23T09:46:43Z" → extract date only
def clean_medications(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("START",             "start_date")
        .withColumnRenamed("STOP",              "stop_date")
        .withColumnRenamed("PATIENT",           "patient_id")
        .withColumnRenamed("ENCOUNTER",         "encounter_id")
        .withColumnRenamed("CODE",              "code")
        .withColumnRenamed("DESCRIPTION",       "description")
        .withColumnRenamed("REASONCODE",        "reasoncode")
        .withColumnRenamed("REASONDESCRIPTION", "reasondescription")
    )

    keep = ["start_date", "stop_date", "patient_id", "encounter_id",
            "code", "description", "reasoncode", "reasondescription"]
    df = df.select(keep)

    df = df.dropna(subset=["patient_id", "code", "start_date"])
    df = df.filter((F.col("patient_id") != "") & (F.col("code") != ""))

    df = (
        df
        .withColumn("start_date",
                    F.to_date(F.to_timestamp("start_date",
                                             "yyyy-MM-dd'T'HH:mm:ss'Z'")))
        .withColumn("stop_date",
                    F.to_date(F.to_timestamp("stop_date",
                                             "yyyy-MM-dd'T'HH:mm:ss'Z'")))
    )

    df = df.dropDuplicates(["patient_id", "code", "start_date"])
    return df


# ── OBSERVATIONS ──────────────────────────────────────────────────────────────
# CSV:  DATE, PATIENT, ENCOUNTER, CATEGORY, CODE, DESCRIPTION,
#       VALUE, UNITS, TYPE
# CH:   date, patient_id, encounter_id(Nullable), code, description,
#       value, units(Nullable), type
# CATEGORY column is NOT in CH schema — dropped
# DATE format: "2019-09-11T09:19:53Z"
def clean_observations(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("DATE",        "date")
        .withColumnRenamed("PATIENT",     "patient_id")
        .withColumnRenamed("ENCOUNTER",   "encounter_id")
        .withColumnRenamed("CODE",        "code")
        .withColumnRenamed("DESCRIPTION", "description")
        .withColumnRenamed("VALUE",       "value")
        .withColumnRenamed("UNITS",       "units")
        .withColumnRenamed("TYPE",        "type")
        # CATEGORY is dropped by not selecting it
    )

    keep = ["date", "patient_id", "encounter_id", "code",
            "description", "value", "units", "type"]
    df = df.select(keep)

    df = df.dropna(subset=["patient_id", "code", "value", "date"])
    df = df.filter(
        (F.col("patient_id") != "") &
        (F.col("code")       != "") &
        (F.col("value")      != "")
    )

    df = df.withColumn(
        "date",
        F.to_timestamp("date", "yyyy-MM-dd'T'HH:mm:ss'Z'")
    )

    # Normalise empty encounter_id → null (CH column is Nullable)
    df = df.withColumn(
        "encounter_id",
        F.when(
            F.col("encounter_id").isNull() | (F.col("encounter_id") == ""),
            F.lit(None)
        ).otherwise(F.col("encounter_id"))
    )

    df = df.dropDuplicates(["patient_id", "code", "date"])
    return df


# ── ENCOUNTERS ────────────────────────────────────────────────────────────────
# CSV:  Id, START, STOP, PATIENT, ORGANIZATION, PROVIDER, PAYER,
#       ENCOUNTERCLASS, CODE, DESCRIPTION, BASE_ENCOUNTER_COST,
#       TOTAL_CLAIM_COST, PAYER_COVERAGE, REASONCODE, REASONDESCRIPTION
# CH:   encounter_id, start_dt, stop_dt, patient_id, encounterclass, code,
#       description, reasoncode, reasondescription,
#       base_encounter_cost, total_claim_cost, payer_coverage
# Timestamps: "2019-09-11T09:19:53Z"
def clean_encounters(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("Id",                  "encounter_id")
        .withColumnRenamed("START",               "start_dt")
        .withColumnRenamed("STOP",                "stop_dt")
        .withColumnRenamed("PATIENT",             "patient_id")
        .withColumnRenamed("ENCOUNTERCLASS",      "encounterclass")
        .withColumnRenamed("CODE",                "code")
        .withColumnRenamed("DESCRIPTION",         "description")
        .withColumnRenamed("REASONCODE",          "reasoncode")
        .withColumnRenamed("REASONDESCRIPTION",   "reasondescription")
        .withColumnRenamed("BASE_ENCOUNTER_COST", "base_encounter_cost")
        .withColumnRenamed("TOTAL_CLAIM_COST",    "total_claim_cost")
        .withColumnRenamed("PAYER_COVERAGE",      "payer_coverage")
    )

    keep = ["encounter_id", "start_dt", "stop_dt", "patient_id",
            "encounterclass", "code", "description", "reasoncode",
            "reasondescription", "base_encounter_cost",
            "total_claim_cost", "payer_coverage"]
    df = df.select(keep)

    df = df.dropna(subset=["encounter_id", "patient_id", "start_dt"])
    df = df.filter(
        (F.col("encounter_id") != "") & (F.col("patient_id") != "")
    )

    df = (
        df
        .withColumn("start_dt",
                    F.to_timestamp("start_dt", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
        .withColumn("stop_dt",
                    F.to_timestamp("stop_dt",  "yyyy-MM-dd'T'HH:mm:ss'Z'"))
        .withColumn("base_encounter_cost",
                    F.col("base_encounter_cost").cast(DoubleType()))
        .withColumn("total_claim_cost",
                    F.col("total_claim_cost").cast(DoubleType()))
        .withColumn("payer_coverage",
                    F.col("payer_coverage").cast(DoubleType()))
    )

    df = df.dropDuplicates(["encounter_id"])
    return df


# ── PROCEDURES ────────────────────────────────────────────────────────────────
# CSV:  START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION,
#       BASE_COST, REASONCODE, REASONDESCRIPTION
# CH:   date, patient_id, encounter_id, code, description,
#       reasoncode, reasondescription, base_cost
# START (timestamp) maps to CH date (Date32) — extract date only
def clean_procedures(df: DataFrame) -> DataFrame:
    df = (
        df
        .withColumnRenamed("START",             "date")
        .withColumnRenamed("PATIENT",           "patient_id")
        .withColumnRenamed("ENCOUNTER",         "encounter_id")
        .withColumnRenamed("CODE",              "code")
        .withColumnRenamed("DESCRIPTION",       "description")
        .withColumnRenamed("BASE_COST",         "base_cost")
        .withColumnRenamed("REASONCODE",        "reasoncode")
        .withColumnRenamed("REASONDESCRIPTION", "reasondescription")
    )

    keep = ["date", "patient_id", "encounter_id", "code",
            "description", "reasoncode", "reasondescription", "base_cost"]
    df = df.select(keep)

    df = df.dropna(subset=["patient_id", "code", "date"])
    df = df.filter((F.col("patient_id") != "") & (F.col("code") != ""))

    df = (
        df
        .withColumn("date",
                    F.to_date(
                        F.to_timestamp("date", "yyyy-MM-dd'T'HH:mm:ss'Z'")
                    ))
        .withColumn("base_cost", F.col("base_cost").cast(DoubleType()))
    )

    df = df.dropDuplicates(["patient_id", "code", "date", "encounter_id"])
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
TABLES = {
    "patients":     clean_patients,
    "conditions":   clean_conditions,
    "medications":  clean_medications,
    "observations": clean_observations,
    "encounters":   clean_encounters,
    "procedures":   clean_procedures,
}


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*60}")
    print(f"Healthcare Clean Job — {datetime.utcnow().isoformat()}")
    print(f"Input:  {HDFS_RAW}")
    print(f"Output: {HDFS_PROCESSED}")
    print(f"{'='*60}\n")

    results = {}
    for table, cleaner in TABLES.items():
        print(f"==> {table}")
        df_raw    = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .csv(f"{HDFS_RAW}/{table}.csv")
        )
        raw_count   = df_raw.count()
        df_clean    = cleaner(df_raw)
        clean_count = df_clean.count()

        df_clean.write.mode("overwrite").parquet(
            f"{HDFS_PROCESSED}/{table}"
        )

        results[table] = (raw_count, clean_count, raw_count - clean_count)
        print(f"    raw={raw_count:,}  clean={clean_count:,}  "
              f"dropped={raw_count - clean_count:,}")

    print(f"\n{'='*60}")
    print("Summary:")
    for t, (r, c, d) in results.items():
        print(f"  {t:<20} raw={r:>10,}  clean={c:>10,}  dropped={d:>6,}")
    print(f"{'='*60}\n")
    spark.stop()


if __name__ == "__main__":
    main()