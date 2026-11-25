import duckdb
import pandas as pd
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

def load_and_interleave_hour(ob_dir: Path | str, snap_dir: Path | str) -> pd.DataFrame:
    """
    Loads orderbook events and snapshots for a specific hour, interleaves them, 
    and sorts using Method B (Direct Multi-Sort by market_id and timestamp_received) 
    to create a single, memory-contiguous, cache-friendly DataFrame.
    
    Returns:
        pd.DataFrame: The chronologically interleaved and market-grouped events.
    """
    t0 = time.time()
    
    # Load into memory using DuckDB for blazing fast Parquet parsing
    ob_query = f"SELECT *, 'event' as type FROM read_parquet('{str(ob_dir)}/**/*.parquet')"
    snap_query = f"SELECT *, 'snapshot' as type FROM read_parquet('{str(snap_dir)}/**/*.parquet')"
    
    ob_df = duckdb.query(ob_query).df()
    snap_df = duckdb.query(snap_query).df()
    
    # Combine both datasets vertically
    combined_df = pd.concat([ob_df, snap_df], ignore_index=True)
    
    # Method B: Direct Multi-Sort
    # This guarantees that the events are grouped by market_id AND ordered chronologically within each market.
    # The underlying memory becomes contiguous, allowing ultra-fast sequential iteration.
    combined_df.sort_values(["market_id", "timestamp_received"], inplace=True)
    
    # Reset index to physically reorganize memory pointers
    combined_df.reset_index(drop=True, inplace=True)
    
    t_total = time.time() - t0
    logger.info(f"Loaded and interleaved {len(combined_df)} rows in {t_total:.3f}s")
    
    return combined_df

# Quick test if script is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    # We use the same hour from the last test for validation
    ob = base / "orderbook/year=2026/month=3/day=23/hour=13"
    snap = base / "snapshots/year=2026/month=3/day=23/hour=13"
    
    if ob.exists() and snap.exists():
        df = load_and_interleave_hour(ob, snap)
        print("\n--- Struttura DataFrame Pronta ---")
        print(f"Shape: {df.shape}")
        print("\nPrimi 5 elementi (primo mercato in ordine alfabetico):")
        print(df[['market_id', 'timestamp_received', 'type']].head())
    else:
        print("Cartelle dell'ora non trovate sul server.")
