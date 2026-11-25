import argparse
import duckdb
import os
import glob
import time
from datetime import datetime

def parse_partition_path(path_str: str) -> dict:
    parts = path_str.split('/')
    result = {}
    for p in parts:
        if '=' in p:
            k, v = p.split('=')
            if k in ['year', 'month', 'day', 'hour']:
                result[k] = int(v)
    return result

def get_files_in_range(base_dir: str, category: str, start_dt: datetime, end_dt: datetime) -> list:
    pattern = os.path.join(base_dir, category, "**", "*.parquet")
    all_files = glob.glob(pattern, recursive=True)
    
    valid_files = []
    for f in all_files:
        p = parse_partition_path(f)
        if 'year' in p and 'month' in p and 'day' in p and 'hour' in p:
            file_dt = datetime(p['year'], p['month'], p['day'], p['hour'])
            if start_dt <= file_dt <= end_dt:
                valid_files.append(f)
    return valid_files

def main():
    parser = argparse.ArgumentParser(description="Use DuckDB to count orderbook events that occur before their first snapshot.")
    parser.add_argument("--base_dir", type=str, default="/home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/partitioned")
    parser.add_argument("--start", type=str, required=True, help="Start time in format 'YYYY-MM-DD HH' (e.g. '2026-03-06 00')")
    parser.add_argument("--end", type=str, required=True, help="End time in format 'YYYY-MM-DD HH' (e.g. '2026-03-06 12')")
    args = parser.parse_args()

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d %H")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d %H")
    except ValueError:
        print("Error: --start and --end must be in the exact format 'YYYY-MM-DD HH'")
        return

    snap_files = get_files_in_range(args.base_dir, "snapshots", start_dt, end_dt)
    ob_files = get_files_in_range(args.base_dir, "orderbook", start_dt, end_dt)

    print(f"================================================================================")
    print(f" DUCKDB ORPHANED TICKS DIAGNOSTIC")
    print(f"================================================================================")
    print(f"Time Span: {start_dt} to {end_dt}")
    print(f"Found {len(snap_files)} snapshot files and {len(ob_files)} orderbook files.")
    
    if not ob_files:
        print("No orderbook files found in this time span. Exiting.")
        return
        
    print("Running query... (this may take a moment on large datasets)")

    # DuckDB accepts lists of files as strings formatted like ['file1', 'file2']
    snap_list_str = "[" + ", ".join([f"'{f}'" for f in snap_files]) + "]"
    ob_list_str = "[" + ", ".join([f"'{f}'" for f in ob_files]) + "]"

    query = f"""
    WITH first_snaps AS (
        SELECT market_id, MIN(timestamp_received) as absolute_first_snap
        FROM read_parquet({snap_list_str}, union_by_name=true)
        GROUP BY market_id
    ),
    ob_events AS (
        SELECT market_id, timestamp_received
        FROM read_parquet({ob_list_str}, union_by_name=true)
    )
    SELECT 
        COUNT(*) as total_ticks,
        SUM(CASE WHEN m.absolute_first_snap IS NULL OR o.timestamp_received < m.absolute_first_snap THEN 1 ELSE 0 END) as ticks_before_first_snap,
        SUM(CASE WHEN m.absolute_first_snap IS NOT NULL AND o.timestamp_received >= m.absolute_first_snap THEN 1 ELSE 0 END) as valid_ticks
    FROM ob_events o
    LEFT JOIN first_snaps m ON o.market_id = m.market_id
    """
    
    start_time = time.time()
    try:
        con = duckdb.connect()
        df = con.execute(query).df()
        
        print("\n--- RESULTS ---")
        print(df.to_string(index=False))
        print(f"\nQuery completed in {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"\nError running query: {e}")

if __name__ == "__main__":
    main()
