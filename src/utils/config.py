"""
Central configuration — all values read from environment variables.
Never hard-code credentials here; set them in .env / docker-compose.yml.
"""

import os

# ---------------------------------------------------------------------------
# ClickHouse
# ---------------------------------------------------------------------------
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "healthcare_user")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "ch_secret_2026")
CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "healthcare")

# ---------------------------------------------------------------------------
# HDFS / WebHDFS
# ---------------------------------------------------------------------------
HDFS_NAMENODE_HOST = os.environ.get("HDFS_NAMENODE_HOST", "namenode")
HDFS_WEBHDFS_PORT = int(os.environ.get("HDFS_WEBHDFS_PORT", "9870"))
HDFS_RPC_PORT = int(os.environ.get("HDFS_RPC_PORT", "9001"))

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data/raw")
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")