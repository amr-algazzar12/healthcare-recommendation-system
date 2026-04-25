"""
feature_engineering.py — PySpark feature engineering job (Milestone 2)

Reads cleaned Parquet from hdfs://namenode:9001/data/processed/
Builds patient-level feature vectors and writes to:
    hdfs://namenode:9001/data/features/patient_features

Output schema matches healthcare.patient_features exactly:
    patient_id, age, gender_encoded, race_encoded,
    num_conditions, num_medications, num_encounters,
    has_diabetes, has_hypertension, has_asthma,
    has_hyperlipidemia, has_coronary_disease,
    condition_vector, medication_history_flags,
    feature_version, created_at

Run via:
    make run-features
    (docker exec spark-master /opt/spark/bin/spark-submit ...)
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, FloatType, IntegerType,
    ShortType, ByteType, TimestampType
)

# ── Paths ──────────────────────────────────────────────────────────────────
HDFS_PROCESSED = "hdfs://namenode:9001/data/processed"
HDFS_FEATURES  = "hdfs://namenode:9001/data/features"
SPARK_MASTER   = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

# ── Feature version tag ────────────────────────────────────────────────────
FEATURE_VERSION = "v1.0"

# ── Chronic disease condition codes (SNOMED-CT) ────────────────────────────
# These are the actual codes Synthea generates — verified against Synthea source
DIABETES_CODES       = {"44054006", "73211009", "314771006"}
HYPERTENSION_CODES   = {"59621000"}
ASTHMA_CODES         = {"195967001"}
HYPERLIPIDEMIA_CODES = {"55822004"}
CORONARY_CODES       = {"53741008", "414545008", "22298006"}

# ── Top-N conditions for multi-hot vector ─────────────────────────────────
# We build a fixed-size vector over the most common condition codes.
# 50 dimensions keeps the array small while covering ~95% of Synthea output.
TOP_N_CONDITIONS = 50

# ── Top-N medication codes for history flags array ────────────────────────
TOP_N_MEDICATIONS = 30


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("healthcare-feature-engineering")
        .master(SPARK_MASTER)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ── DEMOGRAPHICS ──────────────────────────────────────────────────────────
def build_demographics(patients: DataFrame) -> DataFrame:
    """
    From clean patients Parquet produce:
        patient_id, age (Int32), gender_encoded (Int8), race_encoded (Int8)

    gender:  M → 1, F → 0
    race encoding (matches Synthea categories):
        white → 0, black → 1, asian → 2,
        native → 3, hispanic → 4, other → 5
    """
    today = F.current_date()

    race_map = {
        "white":    0,
        "black":    1,
        "asian":    2,
        "native":   3,
        "hispanic": 4,
    }

    race_expr = (
        F.when(F.lower(F.col("race")) == "white",    F.lit(0))
         .when(F.lower(F.col("race")) == "black",    F.lit(1))
         .when(F.lower(F.col("race")) == "asian",    F.lit(2))
         .when(F.lower(F.col("race")) == "native",   F.lit(3))
         .when(F.lower(F.col("race")) == "hispanic", F.lit(4))
         .otherwise(F.lit(5))  # other / unknown
    ).cast(ByteType())

    gender_expr = (
        F.when(F.col("gender") == "M", F.lit(1))
         .otherwise(F.lit(0))
    ).cast(ByteType())

    # Use deathdate if patient is deceased, else today — gives correct age
    age_expr = (
        F.datediff(
            F.coalesce(F.col("deathdate"), today),
            F.col("birthdate")
        ) / 365.25
    ).cast(IntegerType())

    return patients.select(
        F.col("patient_id"),
        age_expr.alias("age"),
        gender_expr.alias("gender_encoded"),
        race_expr.alias("race_encoded"),
    )


# ── CONDITION FEATURES ────────────────────────────────────────────────────
def build_condition_features(
    conditions: DataFrame,
    patients_ids: DataFrame,
) -> DataFrame:
    """
    Produces per-patient:
        patient_id,
        num_conditions       (Int32)  — distinct condition codes
        has_diabetes         (UInt8)
        has_hypertension     (UInt8)
        has_asthma           (UInt8)
        has_hyperlipidemia   (UInt8)
        has_coronary_disease (UInt8)
        condition_vector     (Array(Float32)) — multi-hot over top-50 codes
    """

    # ── chronic disease binary flags ──────────────────────────────────────
    def flag(codes: set) -> F.Column:
        return F.max(
            F.when(F.col("code").isin(codes), F.lit(1)).otherwise(F.lit(0))
        ).cast(ByteType())

    agg_df = conditions.groupBy("patient_id").agg(
        F.countDistinct("code").cast(IntegerType()).alias("num_conditions"),
        flag(DIABETES_CODES).alias("has_diabetes"),
        flag(HYPERTENSION_CODES).alias("has_hypertension"),
        flag(ASTHMA_CODES).alias("has_asthma"),
        flag(HYPERLIPIDEMIA_CODES).alias("has_hyperlipidemia"),
        flag(CORONARY_CODES).alias("has_coronary_disease"),
        F.collect_set("code").alias("patient_codes"),
    )

    # ── multi-hot condition vector ─────────────────────────────────────────
    # Pick top-N codes by frequency across all patients
    top_codes = (
        conditions
        .groupBy("code")
        .agg(F.countDistinct("patient_id").alias("patient_count"))
        .orderBy(F.col("patient_count").desc())
        .limit(TOP_N_CONDITIONS)
        .select("code")
        .rdd.flatMap(lambda r: [r[0]])
        .collect()
    )

    # Build a UDF that converts a patient's code set → fixed-length Float32 array
    top_codes_bc = conditions.sparkSession.sparkContext.broadcast(top_codes)

    @F.udf(ArrayType(FloatType()))
    def make_condition_vector(patient_code_set):
        vocab = top_codes_bc.value
        if patient_code_set is None:
            return [0.0] * len(vocab)
        code_set = set(patient_code_set)
        return [1.0 if c in code_set else 0.0 for c in vocab]

    agg_df = agg_df.withColumn(
        "condition_vector", make_condition_vector(F.col("patient_codes"))
    ).drop("patient_codes")

    # Left join so patients with zero conditions still get a row
    result = patients_ids.join(agg_df, on="patient_id", how="left")

    # Fill nulls for patients with no conditions at all
    zero_vector = [0.0] * TOP_N_CONDITIONS
    result = result.fillna(
        {
            "num_conditions":     0,
            "has_diabetes":       0,
            "has_hypertension":   0,
            "has_asthma":         0,
            "has_hyperlipidemia": 0,
            "has_coronary_disease": 0,
        }
    ).withColumn(
        "condition_vector",
        F.coalesce(F.col("condition_vector"), F.array(*[F.lit(0.0)] * TOP_N_CONDITIONS))
    )

    return result


# ── MEDICATION FEATURES ───────────────────────────────────────────────────
def build_medication_features(
    medications: DataFrame,
    patients_ids: DataFrame,
) -> DataFrame:
    """
    Produces per-patient:
        patient_id,
        num_medications          (Int32)
        medication_history_flags (Array(UInt8)) — multi-hot over top-30 codes
    """

    num_meds = (
        medications
        .groupBy("patient_id")
        .agg(F.countDistinct("code").cast(IntegerType()).alias("num_medications"),
             F.collect_set("code").alias("patient_med_codes"))
    )

    # Top-N medication codes by frequency
    top_med_codes = (
        medications
        .groupBy("code")
        .agg(F.countDistinct("patient_id").alias("patient_count"))
        .orderBy(F.col("patient_count").desc())
        .limit(TOP_N_MEDICATIONS)
        .select("code")
        .rdd.flatMap(lambda r: [r[0]])
        .collect()
    )

    top_med_bc = medications.sparkSession.sparkContext.broadcast(top_med_codes)

    @F.udf(ArrayType(ByteType()))
    def make_med_flags(med_code_set):
        vocab = top_med_bc.value
        if med_code_set is None:
            return [0] * len(vocab)
        code_set = set(med_code_set)
        return [1 if c in code_set else 0 for c in vocab]

    num_meds = num_meds.withColumn(
        "medication_history_flags", make_med_flags(F.col("patient_med_codes"))
    ).drop("patient_med_codes")

    result = patients_ids.join(num_meds, on="patient_id", how="left")

    result = result.fillna({"num_medications": 0}).withColumn(
        "medication_history_flags",
        F.coalesce(
            F.col("medication_history_flags"),
            F.array(*[F.lit(0).cast(ByteType())] * TOP_N_MEDICATIONS)
        )
    )

    return result


# ── ENCOUNTER FEATURES ────────────────────────────────────────────────────
def build_encounter_features(
    encounters: DataFrame,
    patients_ids: DataFrame,
) -> DataFrame:
    """
    Produces per-patient:
        patient_id,
        num_encounters (Int32)
    """
    num_enc = (
        encounters
        .groupBy("patient_id")
        .agg(F.count("encounter_id").cast(IntegerType()).alias("num_encounters"))
    )

    result = patients_ids.join(num_enc, on="patient_id", how="left")
    return result.fillna({"num_encounters": 0})


# ── ASSEMBLE FINAL FEATURE TABLE ──────────────────────────────────────────
def build_patient_features(spark: SparkSession) -> DataFrame:
    """
    Joins all feature groups on patient_id.
    Output schema matches healthcare.patient_features exactly.
    """

    # ── read cleaned Parquet ──────────────────────────────────────────────
    patients   = spark.read.parquet(f"{HDFS_PROCESSED}/patients")
    conditions = spark.read.parquet(f"{HDFS_PROCESSED}/conditions")
    medications = spark.read.parquet(f"{HDFS_PROCESSED}/medications")
    encounters = spark.read.parquet(f"{HDFS_PROCESSED}/encounters")

    # Spine — one row per patient
    patient_ids = patients.select("patient_id")

    # ── build each feature group ──────────────────────────────────────────
    demo   = build_demographics(patients)                                     # patient_id, age, gender_encoded, race_encoded
    cond   = build_condition_features(conditions, patient_ids)               # patient_id, num_conditions, has_*, condition_vector
    meds   = build_medication_features(medications, patient_ids)             # patient_id, num_medications, medication_history_flags
    enc    = build_encounter_features(encounters, patient_ids)               # patient_id, num_encounters

    # ── join on patient_id ────────────────────────────────────────────────
    features = (
        demo
        .join(cond.drop("patient_id"), demo.patient_id == cond.patient_id, how="left")
        .drop(cond.patient_id)
        .join(meds.drop("patient_id"), demo.patient_id == meds.patient_id, how="left")
        .drop(meds.patient_id)
        .join(enc.drop("patient_id"),  demo.patient_id == enc.patient_id,  how="left")
        .drop(enc.patient_id)
    )

    # ── add metadata columns ──────────────────────────────────────────────
    features = features.withColumn(
        "feature_version", F.lit(FEATURE_VERSION)
    ).withColumn(
        "created_at", F.current_timestamp().cast(TimestampType())
    )

    # ── enforce final column order (matches CH DDL exactly) ───────────────
    final_cols = [
        "patient_id",
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
        "has_coronary_disease",
        "condition_vector",
        "medication_history_flags",
        "feature_version",
        "created_at",
    ]

    return features.select(final_cols)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*60}")
    print(f"Healthcare Feature Engineering Job — {datetime.utcnow().isoformat()}")
    print(f"Input:   {HDFS_PROCESSED}")
    print(f"Output:  {HDFS_FEATURES}/patient_features")
    print(f"Version: {FEATURE_VERSION}")
    print(f"{'='*60}\n")

    df = build_patient_features(spark)

    # Cache before count to avoid double computation
    df.cache()
    row_count = df.count()

    print(f"==> patient_features: {row_count:,} rows")

    df.write.mode("overwrite").parquet(
        f"{HDFS_FEATURES}/patient_features"
    )

    print(f"\n{'='*60}")
    print(f"Feature engineering complete.")
    print(f"  patient_features: {row_count:,} rows written")
    print(f"  Output: {HDFS_FEATURES}/patient_features")
    print(f"{'='*60}\n")

    spark.stop()


if __name__ == "__main__":
    main()
