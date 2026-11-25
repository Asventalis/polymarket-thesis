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
        return
        
    random_hour_dir = random.choice(hours)
    rel_parts = random_hour_dir.relative_to(snap_base)
    ob_dir = base_dir / "orderbook" / rel_parts
    snap_dir = random_hour_dir
    
    if not ob_dir.exists():
        print("Missing dir")
        return

    print(f"--- Testing Stable Sort vs Hash Groupby on {rel_parts} ---")
    
    # Load Data
    t0 = time.time()
    ob_df = duckdb.query(f"SELECT *, 'event' as type FROM read_parquet('{str(ob_dir)}/**/*.parquet')").df()
    snap_df = duckdb.query(f"SELECT *, 'snapshot' as type FROM read_parquet('{str(snap_dir)}/**/*.parquet')").df()
    combined_df = pd.concat([ob_df, snap_df], ignore_index=True)
    print(f"Loaded and concatenated {len(combined_df)} rows in {time.time() - t0:.3f}s")
    
    # ========================================================
    # METHOD A: Interleave (Time Sort) + Stable Sort by Market
    # ========================================================
    t0 = time.time()
    df_a = combined_df.copy()
    df_a.sort_values("timestamp_received", inplace=True)
    df_a.sort_values("market_id", kind="mergesort", inplace=True) # mergesort is stable in pandas
    df_a.reset_index(drop=True, inplace=True)
    t_a = time.time() - t0
    print(f"[A] Stable Sort (Time -> Market) took: {t_a:.3f}s")

    # ========================================================
    # METHOD B: Direct Multi-Column Sort
    # ========================================================
    t0 = time.time()
    df_b = combined_df.copy()
    df_b.sort_values(["market_id", "timestamp_received"], inplace=True)
    df_b.reset_index(drop=True, inplace=True)
    t_b = time.time() - t0
    print(f"[B] Direct Multi-Sort (Market, Time) took: {t_b:.3f}s")
    
    # ========================================================
    # METHOD C: Hash Table Groupby (Dict slicing)
    # ========================================================
    t0 = time.time()
    df_c = combined_df.copy()
    df_c.sort_values("timestamp_received", inplace=True)
    grouped = df_c.groupby("market_id")
    groups_dict = {m: g for m, g in grouped}
    t_c = time.time() - t0
    print(f"[C] Hash Table Groupby (Slicing) took: {t_c:.3f}s")

if __name__ == "__main__":
    main()
