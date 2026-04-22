"""
config.py — Central configuration loaded from environment variables.
All secrets come from .env via Docker Compose env_file.
"""
import os

# ClickHouse
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_HTTP_PORT = int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123"))
CLICKHOUSE_DB = os.environ.get("CLICKHOUSE_DB", "healthcare")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "healthcare_user")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "ch_secret_2026")

# HDFS
HDFS_NAMENODE = os.environ.get("HDFS_NAMENODE_HOST", "namenode")
HDFS_PORT = os.environ.get("HDFS_NAMENODE_PORT", "9001")
HDFS_BASE_URL = f"hdfs://{HDFS_NAMENODE}:{HDFS_PORT}"
HDFS_RAW_PATH = f"{HDFS_BASE_URL}/data/raw"
HDFS_PROCESSED_PATH = f"{HDFS_BASE_URL}/data/processed"
HDFS_FEATURES_PATH = f"{HDFS_BASE_URL}/data/features"

# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", "/mlflow/artifacts")

# Spark
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "2g")
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "1g")
