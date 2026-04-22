"""
clickhouse_client.py — Shared ClickHouse client factory.
Replaces the Hive beeline / SSHOperator pattern from v4.
"""
import clickhouse_connect
from src.utils.config import (
    CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT,
    CLICKHOUSE_DB, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
)


def get_client():
    """Return a connected clickhouse_connect client."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_HTTP_PORT,
        database=CLICKHOUSE_DB,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )


def query_df(sql: str):
    """Execute a SELECT and return a pandas DataFrame."""
    client = get_client()
    return client.query_df(sql)


def execute(sql: str):
    """Execute a non-SELECT statement (INSERT, CREATE, etc.)."""
    client = get_client()
    client.command(sql)
