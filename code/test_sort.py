import time
import pandas as pd
import duckdb
from pathlib import Path
import random

def main():
    base_dir = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    
    # Pick a random hour from snapshots (since they were the ones that needed sorting)
    snap_base = base_dir / "snapshots"
    hours = list(snap_base.glob("year=*/month=*/day=*/hour=*"))
    
    if not hours:
        print("No hour directories found!")
        return
        
    random_hour_dir = random.choice(hours)
    
    # Extract the relative path parts (year=..., month=..., day=..., hour=...)
    rel_parts = random_hour_dir.relative_to(snap_base)
    
    ob_dir = base_dir / "orderbook" / rel_parts
    snap_dir = random_hour_dir
    
    print(f"--- Testing Random Hour ---")
    print(f"Path: {rel_parts}")
    
    # Check if both directories exist
    if not ob_dir.exists():
        print(f"Warning: Corresponding orderbook dir {ob_dir} does not exist.")
        return
        
    # Orderbook
    t0 = time.time()
    ob_files = sorted(list(ob_dir.glob("*.parquet")))
    print(f"\n[Orderbook] Found {len(ob_files)} parquet files. Loading into RAM...")
    ob_df = duckdb.query(f"SELECT * FROM read_parquet('{str(ob_dir)}/**/*.parquet')").df()
    t_load_ob = time.time() - t0
    print(f"[Orderbook] Loaded {len(ob_df)} events in {t_load_ob:.3f} seconds.")
    
    t0 = time.time()
    ob_sorted = ob_df['timestamp_received'].is_monotonic_increasing
    t_check_ob = time.time() - t0
    print(f"[Orderbook] Is sorted chronologically? {ob_sorted} (Checked {len(ob_df)} rows in {t_check_ob:.3f}s)")
    
    # Snapshots
    t0 = time.time()
    snap_files = sorted(list(snap_dir.glob("*.parquet")))
    print(f"\n[Snapshots] Found {len(snap_files)} parquet files. Loading into RAM...")
    snap_df = duckdb.query(f"SELECT * FROM read_parquet('{str(snap_dir)}/**/*.parquet')").df()
    t_load_snap = time.time() - t0
    print(f"[Snapshots] Loaded {len(snap_df)} snapshots in {t_load_snap:.3f} seconds.")
    
    t0 = time.time()
    snap_sorted = snap_df['timestamp_received'].is_monotonic_increasing
    t_check_snap = time.time() - t0
    print(f"[Snapshots] Is sorted chronologically? {snap_sorted} (Checked {len(snap_df)} rows in {t_check_snap:.3f}s)")
    
    if ob_sorted and snap_sorted:
        print("\nSUCCESS! Both Orderbook and Snapshots are perfectly sorted in RAM!")
    else:
        print("\nWARNING: Some data is still unsorted!")

if __name__ == "__main__":
    main()
