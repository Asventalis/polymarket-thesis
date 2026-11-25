"""Script to build the 1-hour development fixture."""

from pathlib import Path
import duckdb
from polymarket_gnn.data.duckdb_client import get_duckdb_connection, execute_sql_file

def build_dev_fixture():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sql_dir = project_root / "sql"
    dev_fixture_dir = project_root / "data" / "derived" / "dev_fixture"
    dev_fixture_dir.mkdir(parents=True, exist_ok=True)
    
    # Needs a read-write connection to create temp tables and tables
    # However, since we don't want to modify preprocessing.duckdb, we'll attach it read-only
    # to an in-memory database!
    db_path = project_root / "data" / "preprocessing.duckdb"
    con = duckdb.connect(':memory:')
    con.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")
    
    # We create active_markets to significantly reduce the cross-join grid size
    print("Finding active markets for the 1-hour window...")
    con.execute("""
        CREATE TEMP TABLE active_markets AS
        SELECT DISTINCT market_id 
        FROM db.orderbook
        WHERE 
            to_timestamp(timestamp_received / 1000.0) >= TIMESTAMP '2026-03-10 13:00:00' 
            AND to_timestamp(timestamp_received / 1000.0) <= TIMESTAMP '2026-03-10 15:00:00'
            AND market_id IN (SELECT market_id FROM db.markets_universe);
    """)
    
    params = {
        "start_time": "2026-03-10 14:00:00",
        "end_time": "2026-03-10 15:00:00",
        "lookback_start_time": "2026-03-10 13:59:40",
        "memory_seconds": 20
    }
    
    print("Executing 03_orderbook_features_2s.sql...")
    execute_sql_file(con, sql_dir / "03_orderbook_features_2s.sql", params)
    
    print("Executing 04_transaction_features_2s.sql...")
    execute_sql_file(con, sql_dir / "04_transaction_features_2s.sql", params)
    
    print("Executing 05_targets_2s.sql...")
    execute_sql_file(con, sql_dir / "05_targets_2s.sql", params)
    
    print("Executing 06_market_edges_2s.sql...")
    execute_sql_file(con, sql_dir / "06_market_edges_2s.sql", params)
    
    print("Exporting orderbook_features_2s to Parquet...")
    con.execute(f"COPY orderbook_features_2s TO '{dev_fixture_dir}/orderbook_features_2s.parquet' (FORMAT PARQUET);")
    
    print("Exporting transaction_features_2s to Parquet...")
    con.execute(f"COPY transaction_features_2s TO '{dev_fixture_dir}/transaction_features_2s.parquet' (FORMAT PARQUET);")
    
    print("Exporting targets_2s to Parquet...")
    con.execute(f"COPY targets_2s TO '{dev_fixture_dir}/targets_2s.parquet' (FORMAT PARQUET);")
    
    print("Exporting market_edges_2s to Parquet...")
    con.execute(f"COPY market_edges_2s TO '{dev_fixture_dir}/market_edges_2s.parquet' (FORMAT PARQUET);")
    
    print("Successfully built dev fixture in data/derived/dev_fixture/")

if __name__ == "__main__":
    build_dev_fixture()
