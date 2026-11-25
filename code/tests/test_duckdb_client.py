"""Unit tests for DuckDB client helper."""

import pytest
from polymarket_gnn.data.duckdb_client import get_duckdb_connection, DEFAULT_DB_PATH


def test_duckdb_read_only_connection():
    con = get_duckdb_connection(DEFAULT_DB_PATH, read_only=True)
    assert con is not None
    
    # Test querying universe.markets
    res = con.sql("SELECT COUNT(*) FROM universe.markets").fetchone()
    assert res is not None
    assert res[0] > 0
    con.close()
