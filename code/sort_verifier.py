import os
import time
import datetime
from pathlib import Path
import duckdb

class DeltaLogger:
    def __init__(self):
        self.last_time = time.time()
        
    def log(self, msg, reset=False):
        now = time.time()
        if reset:
            self.last_time = now
        delta = now - self.last_time
        self.last_time = now
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp_str}] [+{delta:7.3f}s] {msg}", flush=True)

logger = DeltaLogger()

def check_and_sort_directory(h_dir: Path, con: duckdb.DuckDBPyConnection):
    # If a previous run left a temp file, clean it up
    temp_file = h_dir / "temp_sorted.parquet"
    if temp_file.exists():
        temp_file.unlink()

    # Find all parquet files
    parquet_files = sorted(list(h_dir.glob("*.parquet")))
    if not parquet_files:
        return
        
    logger.log(f"Checking out-of-core: {h_dir}")
    
    # We must quote the glob pattern correctly for DuckDB
    files_glob = f"{str(h_dir)}/*.parquet"
    
    # 1. Check if sorted
    query_check = f"""
    SELECT count(*) FROM (
        SELECT timestamp_received, lag(timestamp_received) OVER () as prev_ts
        FROM read_parquet('{files_glob}')
    ) WHERE timestamp_received < prev_ts
    """
    
    try:
        t0 = time.time()
        res = con.execute(query_check).fetchone()
        unsorted_count = res[0]
        check_time = time.time() - t0
        
        if unsorted_count == 0:
            logger.log(f"  -> OK. File is strictly sorted. (Check took {check_time:.3f}s)")
        else:
            logger.log(f"  -> FAILED! Found {unsorted_count} unordered rows. (Check took {check_time:.3f}s)")
            logger.log(f"  -> Initiating Out-of-Core Merge-Sort and Rewrite to Disk...")
            
            t1 = time.time()
            temp_file = h_dir / "temp_sorted.tmp"
            if temp_file.exists():
                temp_file.unlink()
                
            query_sort = f"""
            COPY (
                SELECT * FROM read_parquet('{files_glob}') 
                ORDER BY timestamp_received
            ) TO '{str(temp_file)}' (FORMAT PARQUET)
            """
            con.execute(query_sort)
            sort_time = time.time() - t1
            
            logger.log(f"  -> Sort completed in {sort_time:.3f}s. Replacing old files...")
            
            # Delete old un-sorted parquet files
            for p_file in parquet_files:
                p_file.unlink()
                
            # Rename temp to the final name
            final_name = h_dir / "000000_sorted.parquet"
            temp_file.rename(final_name)
            logger.log(f"  -> Replacement successful.")
            
    except Exception as e:
        logger.log(f"  -> ERROR processing {h_dir}: {e}")

def process_table_type(base_dir: Path, table_type: str):
    logger.log(f"Scanning for {table_type} directories in {base_dir}...")
    
    # Find all hour directories dynamically
    hour_dirs = []
    for year_dir in base_dir.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            for day_dir in month_dir.glob("day=*"):
                for hour_dir in day_dir.glob("hour=*"):
                    hour_dirs.append(hour_dir)
                    
    hour_dirs = sorted(hour_dirs)
    logger.log(f"Found {len(hour_dirs)} hour directories to process for {table_type}.")
    
    # Use a single in-memory connection for the whole batch
    con = duckdb.connect(database=':memory:')
    
    for h_dir in hour_dirs:
        check_and_sort_directory(h_dir, con)

def main():
    logger.log("Starting Out-of-Core Parquet Sort Verifier", reset=True)
    base_data = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    
    ob_dir = base_data / "orderbook"
    if ob_dir.exists():
        process_table_type(ob_dir, "orderbook")
        
    snap_dir = base_data / "snapshots"
    if snap_dir.exists():
        process_table_type(snap_dir, "snapshots")
        
    logger.log("=== ALL DIRECTORIES VERIFIED AND SORTED ===")

if __name__ == "__main__":
    main()
