import time
import pandas as pd
import duckdb
from pathlib import Path
import random

def main():
    base_dir = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    snap_base = base_dir / "snapshots"
    hours = list(snap_base.glob("year=*/month=*/day=*/hour=*"))
    
    if not hours:
        print("No hour directories found!")
        return
        
    # random_hour_dir = random.choice(hours)
    # Let's pick a fixed one just in case, or random
    random_hour_dir = random.choice(hours)
    rel_parts = random_hour_dir.relative_to(snap_base)
    ob_dir = base_dir / "orderbook" / rel_parts
    snap_dir = random_hour_dir
    
    if not ob_dir.exists() or not snap_dir.exists():
        print("Mismatched directories. Rerun.")
        return

    print(f"--- Testing Grouping on Random Hour ---")
    print(f"Path: {rel_parts}")
    
    # 1. Load Data with a 'type' flag to distinguish them
    t0 = time.time()
    ob_df = duckdb.query(f"SELECT *, 'event' as type FROM read_parquet('{str(ob_dir)}/**/*.parquet')").df()
    snap_df = duckdb.query(f"SELECT *, 'snapshot' as type FROM read_parquet('{str(snap_dir)}/**/*.parquet')").df()
    t_load = time.time() - t0
    print(f"1. Loaded {len(ob_df)} events and {len(snap_df)} snapshots in {t_load:.3f}s")
    
    # 2. Combine and Interleave Chronologically
    t0 = time.time()
    combined_df = pd.concat([ob_df, snap_df], ignore_index=True)
    combined_df.sort_values("timestamp_received", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    t_combine = time.time() - t0
    print(f"2. Combined and interleaved {len(combined_df)} rows globally in {t_combine:.3f}s")
    
    # 3. Group by Market ID
    t0 = time.time()
    grouped = combined_df.groupby("market_id")
    
    # By default Pandas groupby is lazy. To measure true extraction time, 
    # we convert it into a concrete dictionary of DataFrames.
    groups_dict = {market_id: group for market_id, group in grouped}
    t_group = time.time() - t0
    print(f"3. Grouped into {len(groups_dict)} unique markets in {t_group:.3f}s")
    
    # Print some stats
    num_events = [len(g) for g in groups_dict.values()]
    avg_events = sum(num_events) / len(num_events)
    max_events = max(num_events)
    
    print(f"\nStats:")
    print(f"- Average events/snapshots per market: {avg_events:.1f}")
    print(f"- Max events in a single market: {max_events}")

if __name__ == "__main__":
    main()
