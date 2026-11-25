import duckdb
import pandas as pd
import numpy as np
import time

print("Loading data from DuckDB (10-minute slice)...")
con = duckdb.connect("data/preprocessing.duckdb", read_only=True)

# 10-minute window for testing
start_time_ob = pd.to_datetime("2026-03-06 00:00:00").timestamp() * 1000
end_time_ob = pd.to_datetime("2026-03-06 00:10:00").timestamp() * 1000

print("Executing Vectorized SQL Query to compute Orderbook Deltas...")
t0 = time.time()
ob_query = f"""
SELECT * FROM (
    SELECT 
        market_id, 
        token_id,
        change_side, 
        change_price, 
        timestamp_received, 
        change_size,
        change_size - LAG(change_size, 1, 0) OVER (PARTITION BY market_id, change_side, change_price ORDER BY timestamp_received) as delta
    FROM orderbook 
    WHERE timestamp_received BETWEEN {start_time_ob} AND {end_time_ob}
) WHERE delta < -1e-6
"""
ob_df = con.execute(ob_query).df()

tx_query = f"""
SELECT market_id, token_id, price, tokens as size, side as tx_side, CAST(EXTRACT(EPOCH FROM timestamp)*1000 AS BIGINT) as timestamp_ms
FROM transactions_with_market 
WHERE timestamp BETWEEN '2026-03-06 00:00:00' AND '2026-03-06 00:10:10'
"""
tx_df = con.execute(tx_query).df()
print(f"Data Loaded in {time.time()-t0:.2f}s")

ob_df["rem_size"] = -ob_df["delta"]
print(f"Total Negative Delta Events (Removed Liquidity): {len(ob_df)}")
print(f"Total Transactions in window: {len(tx_df)}")

print("Matching Orderbook against Blockchain Transactions (6-second window)...")
t1 = time.time()

# Convert price/size to rounded strings/integers for exact fast matching
# We use rounding to avoid float precision issues during merges
ob_df["match_price"] = ob_df["change_price"].round(3).astype('float64')
ob_df["match_size"] = ob_df["rem_size"].round(3).astype('float64')

tx_df["match_price"] = tx_df["price"].round(3).astype('float64')
tx_df["match_size"] = tx_df["size"].round(3).astype('float64')

# Filter transactions for YES token Buys if user prefers, but actually let's match any execution
# that perfectly hits the token_id and market_id
# Sort for merge_asof
ob_df = ob_df.sort_values("timestamp_received")
tx_df = tx_df.sort_values("timestamp_ms")

# Perform a rolling asof merge: For each Orderbook event, find the next Transaction within 6000ms
# matching exact market, token, price, and size
matched = pd.merge_asof(
    ob_df, 
    tx_df, 
    left_on="timestamp_received", 
    right_on="timestamp_ms", 
    by=["market_id", "token_id", "match_price", "match_size"],
    direction="forward",
    tolerance=6000 # 6 seconds in milliseconds
)

# If timestamp_ms is not null, a match was found!
successful_matches = matched["timestamp_ms"].notnull().sum()
total_removals = len(ob_df)

print(f"Matching Completed in {time.time()-t1:.2f}s")
print("\n--- RESULTS ---")
print(f"Total Negative Deltas:           {total_removals}")
print(f"Matched with TX (Executions):    {successful_matches}")
print(f"Unmatched (True Cancellations):  {total_removals - successful_matches}")
print(f"True Cancel / Total Removal:     {(total_removals - successful_matches) / total_removals * 100:.2f}%")
