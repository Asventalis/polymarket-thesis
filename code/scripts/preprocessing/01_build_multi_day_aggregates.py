"""Script to build the full multi-day dataset (March 6 - March 26)."""

import os
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
from polymarket_gnn.data.duckdb_client import get_duckdb_connection, execute_sql_file

def build_multi_day():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sql_dir = project_root / "sql"
    out_dir = project_root / "data" / "derived" / "multi_day"
    
    # Create subdirectories for tables
    for table in ["orderbook", "transactions", "targets", "edges"]:
        (out_dir / table).mkdir(parents=True, exist_ok=True)
        
    db_path = project_root / "data" / "preprocessing.duckdb"
    
    start_date = datetime(2026, 3, 6)
    end_date = datetime(2026, 3, 26)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n======================================")
        print(f"Processing date: {date_str}")
        print(f"======================================")
        
        # Start of day and end of day
        t_start = f"{date_str} 00:00:00"
        t_end = f"{date_str} 23:59:59"
        
        # Open a fresh memory DB to avoid Temp table collisions and memory bloat
        con = duckdb.connect(':memory:')
        con.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")
        
        params = {
            "start_time": t_start,
            "end_time": t_end,
            "lookback_start_time": (current_date - timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("Finding active markets for the day...")
        con.execute(f"""
            CREATE TEMP TABLE active_markets AS
            SELECT DISTINCT market_id 
            FROM db.orderbook
            WHERE 
                to_timestamp(timestamp_received / 1000.0) >= TIMESTAMP '{params['start_time']}' 
                AND to_timestamp(timestamp_received / 1000.0) <= TIMESTAMP '{params['end_time']}'
                AND market_id IN (SELECT market_id FROM db.markets_universe);
        """)
        
        print("Executing orderbook features...")
        execute_sql_file(con, sql_dir / "03_orderbook_features_2s.sql", params)
        print(f"Exporting orderbook -> {out_dir}/orderbook/{date_str}.parquet")
        con.execute(f"COPY orderbook_features_2s TO '{out_dir}/orderbook/{date_str}.parquet' (FORMAT PARQUET);")
        
        print("Executing transaction features...")
        execute_sql_file(con, sql_dir / "04_transaction_features_2s.sql", params)
        print(f"Exporting transactions -> {out_dir}/transactions/{date_str}.parquet")
        con.execute(f"COPY transaction_features_2s TO '{out_dir}/transactions/{date_str}.parquet' (FORMAT PARQUET);")
        
        print("Executing targets...")
        execute_sql_file(con, sql_dir / "05_targets_2s.sql", params)
        print(f"Exporting targets -> {out_dir}/targets/{date_str}.parquet")
        con.execute(f"COPY targets_2s TO '{out_dir}/targets/{date_str}.parquet' (FORMAT PARQUET);")
        
        for mem in [2, 10, 20, 40]:
            (out_dir / f"edges_{mem}s").mkdir(parents=True, exist_ok=True)
            params["lookback_start_time"] = (current_date - timedelta(seconds=mem)).strftime("%Y-%m-%d %H:%M:%S")
            params["memory_seconds"] = mem
            print(f"Executing edges for {mem}s memory...")
            execute_sql_file(con, sql_dir / "06_market_edges_2s.sql", params)
            print(f"Exporting edges -> {out_dir}/edges_{mem}s/{date_str}.parquet")
            con.execute(f"COPY market_edges_2s TO '{out_dir}/edges_{mem}s/{date_str}.parquet' (FORMAT PARQUET);")
        
        con.close()
        current_date += timedelta(days=1)

if __name__ == "__main__":
    build_multi_day()
