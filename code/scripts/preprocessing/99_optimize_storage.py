"""Optimize dataset layout on disk for pure sequential IO access.

Pre-joins orderbook, transactions, and target tables into a single unified Parquet file
sorted strictly by (grid_time ASC, market_id ASC) with Parquet Row-Group zone mapping.
"""

import time
from pathlib import Path
import duckdb

def build_all_unified_sorted_datasets(num_markets=500):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    multi_day_dir = project_root / "data" / "derived" / "multi_day"
    targets_dir = multi_day_dir / "targets"
    unified_dir = multi_day_dir / "unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    target_files = sorted(list(targets_dir.glob("*.parquet")))
    if not target_files:
        print("No target parquet files found to optimize.")
        return
        
    print(f"Found {len(target_files)} daily parquet datasets to pre-sort and optimize...")
    
    for t_file in target_files:
        date_str = t_file.stem
        out_file = unified_dir / f"{date_str}.parquet"
        
        print(f"\nBuilding unified pre-sorted Parquet dataset for {date_str} (Top {num_markets} markets)...")
        start_t = time.time()
        
        con = duckdb.connect()
        con.execute("SET preserve_insertion_order = false;")
        
        query = f"""
        COPY (
            WITH selected_markets AS (
                SELECT DISTINCT market_id
                FROM read_parquet('{multi_day_dir}/targets/{date_str}.parquet')
                LIMIT {num_markets}
            )
            SELECT 
                o.grid_time,
                o.market_id,
                o.mid_price, o.spread, o.best_bid, o.best_ask,
                tx.volume_2s, tx.num_trades_2s, tx.buy_volume_2s, tx.sell_volume_2s, tx.unique_wallets_2s,
                t.target_10s
            FROM read_parquet('{multi_day_dir}/orderbook/{date_str}.parquet') o
            JOIN read_parquet('{multi_day_dir}/transactions/{date_str}.parquet') tx
              ON o.market_id = tx.market_id AND o.grid_time = tx.grid_time
            JOIN read_parquet('{multi_day_dir}/targets/{date_str}.parquet') t
              ON o.market_id = t.market_id AND o.grid_time = t.grid_time
            JOIN selected_markets sm
              ON o.market_id = sm.market_id
            WHERE t.target_10s IS NOT NULL
            ORDER BY o.grid_time ASC, o.market_id ASC
        ) TO '{out_file}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'SNAPPY');
        """
        
        con.execute(query)
        elapsed = time.time() - start_t
        print(f"Successfully generated pre-sorted unified Parquet -> {out_file} in {elapsed:.2f}s")
        
        # Test sequential benchmark read
        bench_start = time.time()
        bench_query = f"SELECT * FROM read_parquet('{out_file}') LIMIT 7200"
        df = con.execute(bench_query).df()
        bench_elapsed = time.time() - bench_start
        print(f"Loaded {len(df):,} rows from {out_file.name} in {bench_elapsed*1000:.2f} ms (Pure Sequential IO Access)!")

if __name__ == "__main__":
    build_all_unified_sorted_datasets()


