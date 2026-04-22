"""
spark_session.py — Centralized SparkSession factory.
Connects to Spark Standalone master by default.
Switch SPARK_MASTER env var to 'yarn' for YARN cluster mode.
"""
from pyspark.sql import SparkSession
from src.utils.config import SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY


def get_spark(app_name: str = "HealthcareRec") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
