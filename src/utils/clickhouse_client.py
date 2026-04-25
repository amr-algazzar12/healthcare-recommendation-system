"""
Thin wrapper around clickhouse-connect.

Usage
-----
    from utils.clickhouse_client import get_client, query_df

    client = get_client()
    df = query_df("SELECT * FROM healthcare.patients LIMIT 10")
"""

from __future__ import annotations

import os
from functools import lru_cache

import clickhouse_connect
from clickhouse_connect.driver.client import Client


def get_client() -> Client:
    """
    Return a clickhouse-connect Client configured from environment variables.

    Environment variables (with defaults matching docker-compose / .env):
        CLICKHOUSE_HOST       clickhouse
        CLICKHOUSE_PORT       8123          ← HTTP interface (not 9000/9900)
        CLICKHOUSE_USER       healthcare_user
        CLICKHOUSE_PASSWORD   ch_secret_2026
        CLICKHOUSE_DATABASE   healthcare
    """
    host = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
    port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
    user = os.environ.get("CLICKHOUSE_USER", "healthcare_user")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "ch_secret_2026")
    database = os.environ.get("CLICKHOUSE_DATABASE", "healthcare")

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
        # Increase timeout for large INSERT batches
        connect_timeout=30,
        send_receive_timeout=300,
    )


def query_df(sql: str) -> "pd.DataFrame":
    """
    Execute a SELECT and return results as a pandas DataFrame.
    Convenience wrapper used by notebooks and ad-hoc scripts.
    """
    client = get_client()
    return client.query_df(sql)