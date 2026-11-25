import time
import pandas as pd
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    # Using the exact same hour we just processed for consistency
    snap_dir = base_dir / "snapshots/year=2026/month=3/day=13/hour=1"
    ob_dir = base_dir / "orderbook/year=2026/month=3/day=13/hour=1"
    
    print("Loading data...")
    ob_df = duckdb.query(f"SELECT market_id FROM read_parquet('{str(ob_dir)}/**/*.parquet')").df()
    snap_df = duckdb.query(f"SELECT market_id FROM read_parquet('{str(snap_dir)}/**/*.parquet')").df()
    
    print("Aggregating counts...")
    combined_df = pd.concat([ob_df, snap_df], ignore_index=True)
    counts = combined_df['market_id'].value_counts().values
    
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    
    # Generate custom bins: 1, 2..9, 10, 20..90, 100, 200..
    max_val = counts.max()
    bins = []
    magnitude = 1
    while magnitude <= max_val * 10:
        for i in range(1, 10):
            bins.append(i * magnitude)
        magnitude *= 10
    bins = [b for b in bins if b <= max_val * 10]
    
    plt.hist(counts, bins=bins, color='#3498db', edgecolor='black', alpha=0.8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Distribution of Events per Market (Log-Log Scale)')
    plt.xlabel('Number of Events in 1 Hour (Log Scale)')
    plt.ylabel('Number of Markets (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    output_path = '/data/v.pampinella/agy_project/events_distribution.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
