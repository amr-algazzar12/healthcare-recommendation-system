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
TOP_N_MEDICATIONS = 50


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

    n_conds = len(top_codes)
    _cond_vocab = list(top_codes)

    @F.udf(ArrayType(FloatType()))
    def make_condition_vector(patient_code_set, vocab=_cond_vocab):
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
    result = result.fillna(
        {
            "num_conditions":       0,
            "has_diabetes":         0,
            "has_hypertension":     0,
            "has_asthma":           0,
            "has_hyperlipidemia":   0,
            "has_coronary_disease": 0,
        }
    ).withColumn(
        "condition_vector",
        F.coalesce(
            F.col("condition_vector"),
            F.array(*[F.lit(0.0).cast(FloatType())] * n_conds)
        )
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
        medication_history_flags (Array(Float32)) — TF-IDF weighted, L2-normalized

    Weighting strategy:
        TF  = log(1 + count of medication occurrences for this patient)
              — log dampening prevents high-frequency drugs from dominating
        IDF = log((total_patients + 1) / (patients_who_took_this_med + 1))
              — sklearn-style smoothed IDF; reduces weight of ubiquitous drugs
        TF-IDF = TF * IDF
        Final vector = L2-normalize(TF-IDF) with epsilon smoothing

    Epsilon smoothing (1e-4 added before normalisation):
        Prevents fully-zero vectors for patients whose medications fall
        entirely outside the top-N vocabulary, so cosine similarity never
        collapses to zero for such patients.

    Implementation uses only Spark built-in functions (pivot + array expr)
    — no Python UDFs — for full executor-side parallelism and scalability.
    """

    # ── 1. Total patient count (for IDF denominator) ──────────────────────
    total_patients = patients_ids.count()

    # ── 2. Per-patient, per-medication raw occurrence count ───────────────
    # Each row in medications = one prescription/dispensing event.
    # We count raw rows (not distinct) so repeated prescriptions raise TF.
    patient_med_counts = (
        medications
        .groupBy("patient_id", "code")
        .agg(F.count("*").cast(IntegerType()).alias("raw_count"))
    )

    # ── 3. Per-medication distinct patient count (for IDF) ────────────────
    med_patient_counts = (
        medications
        .groupBy("code")
        .agg(F.countDistinct("patient_id").alias("patient_count"))
    )

    # ── 4. Select top-N medication codes by patient coverage ──────────────
    # Ordering by patient_count (breadth) rather than total occurrences
    # keeps the vocabulary representative of the population.
    top_med_codes_df = (
        med_patient_counts
        .orderBy(F.col("patient_count").desc())
        .limit(TOP_N_MEDICATIONS)
    )

    # Collect as an ordered Python list — order is fixed here and reused
    # for both the pivot and the final array construction so columns always
    # align with the same position in every patient's vector.
    top_med_codes = (
        top_med_codes_df
        .select("code")
        .rdd.flatMap(lambda r: [r[0]])
        .collect()
    )
    n_meds = len(top_med_codes)

    # ── 5. Compute smoothed IDF for top-N codes ───────────────────────────
    # IDF = log((N + 1) / (df + 1))   — sklearn BM25-compatible smoothing
    # Joining with top_med_codes_df restricts IDF table to vocabulary only.
    idf_df = (
        top_med_codes_df
        .withColumn(
            "idf",
            F.log(F.lit(float(total_patients + 1)) / (F.col("patient_count") + 1))
        )
        .select("code", "idf")
    )

    # ── 6. Compute per-patient TF-IDF scores for top-N codes ──────────────
    # inner join: keeps only (patient, code) pairs where code is in vocab
    tfidf_df = (
        patient_med_counts
        .join(idf_df, on="code", how="inner")
        .withColumn(
            "tf",
            F.log(F.lit(1.0) + F.col("raw_count").cast(FloatType()))
        )
        .withColumn("tfidf", F.col("tf") * F.col("idf"))
        .select("patient_id", "code", "tfidf")
    )

    # ── 7. Pivot to wide format: one column per vocabulary code ───────────
    # Passing explicit values list guarantees consistent column ordering
    # regardless of which codes happen to appear in a given data partition.
    tfidf_wide = (
        tfidf_df
        .groupBy("patient_id")
        .pivot("code", top_med_codes)
        .agg(F.first("tfidf"))
        .fillna(0.0)
    )

    # ── 8. num_medications: distinct codes across ALL medications ─────────
    # Intentionally counts the full medication history, not just top-N,
    # so the count column retains its original semantics.
    num_meds_df = (
        medications
        .groupBy("patient_id")
        .agg(F.countDistinct("code").cast(IntegerType()).alias("num_medications"))
    )

    # ── 9. Build L2-normalised, epsilon-smoothed vector ───────────────────
    #
    # For each position i in the vocabulary:
    #   smoothed_i  = tfidf_i + EPSILON
    #   l2_norm     = sqrt( sum_i( smoothed_i ^ 2 ) )
    #   final_i     = smoothed_i / l2_norm
    #
    # EPSILON (1e-4) is small enough not to distort real signal but large
    # enough to guarantee a non-zero denominator and a non-zero vector even
    # for patients with no top-N medication history.
    EPSILON = 1e-4

    # Smoothed raw expressions (before normalisation)
    smoothed_exprs = [
        (F.coalesce(F.col(f"`{c}`"), F.lit(0.0)) + F.lit(EPSILON)).cast(FloatType())
        for c in top_med_codes
    ]

    # L2 norm computed entirely in Spark SQL (no UDF)
    squared_sum_expr = sum(e ** 2 for e in smoothed_exprs)
    l2_norm_expr = F.sqrt(squared_sum_expr)

    # Final normalised array expression
    normalized_exprs = [
        (e / l2_norm_expr).cast(FloatType())
        for e in smoothed_exprs
    ]

    tfidf_wide = (
        tfidf_wide
        .withColumn("medication_history_flags", F.array(*normalized_exprs))
        .select("patient_id", "medication_history_flags")
    )

    # ── 10. Join back to patient spine ────────────────────────────────────
    result = patients_ids.join(num_meds_df, on="patient_id", how="left")
    result = result.join(tfidf_wide,   on="patient_id", how="left")

    # Patients entirely absent from medications table get:
    #   num_medications = 0
    #   medication_history_flags = uniform epsilon-normalised vector
    # The uniform vector is the L2-normalised all-epsilon array:
    #   each element = ε / sqrt(n * ε²) = 1 / sqrt(n)
    uniform_val = float(1.0 / (n_meds ** 0.5))
    zero_fallback = F.array(*[F.lit(uniform_val).cast(FloatType())] * n_meds)

    result = (
        result
        .fillna({"num_medications": 0})
        .withColumn(
            "medication_history_flags",
            F.coalesce(F.col("medication_history_flags"), zero_fallback)
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
    demo   = build_demographics(patients)
    cond   = build_condition_features(conditions, patient_ids)
    meds   = build_medication_features(medications, patient_ids)
    enc    = build_encounter_features(encounters, patient_ids)

    # ── join on patient_id ────────────────────────────────────────────────
    features = (
        demo
        .join(cond,  on="patient_id", how="left")
        .join(meds,  on="patient_id", how="left")
        .join(enc,   on="patient_id", how="left")
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