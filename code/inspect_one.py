import duckdb
import pandas as pd
import numpy as np

print("Querying the natively partitioned parquet dataset for a negative delta execution match...")

# Hunt for a True Execution among the pre-computed deltas in the partitioned dataset
query_ob = """
SELECT market_id, token_id, change_side, change_price, timestamp_received, change_size, delta
FROM read_parquet('data/derived/partitioned/orderbook/year=2026/month=3/day=6/hour=0/*.parquet')
WHERE delta < -50
LIMIT 5000
"""
ob_events = duckdb.query(query_ob).df()

for idx, ob_event in ob_events.iterrows():
    m_id = ob_event["market_id"]
    p = float(ob_event["change_price"])
    s = float(abs(ob_event["delta"]))
    
    start_ts = ob_event["timestamp_received"]
    end_ts = start_ts + pd.Timedelta(seconds=6)
    
    query_tx = f"""
    SELECT timestamp, side as tx_side, price, tokens as size, tx_hash
    FROM read_parquet('data/derived/partitioned/transactions/year=2026/month=3/day=6/hour=0/*.parquet')
    WHERE market_id = '{m_id}'
    AND timestamp >= '{start_ts}' 
    AND timestamp <= '{end_ts}'
    """
    try:
        tx_events = duckdb.query(query_tx).df()
    except Exception:
        # Just in case the transactions parquet doesn't have the exact hourly partition
        continue
    
    if not tx_events.empty:
        p_match = np.isclose(tx_events["price"], p, atol=1e-3)
        s_match = np.isclose(tx_events["size"], s, atol=1e-3)
        
        if (p_match & s_match).any():
            print("\n==== 1. THE ORDERBOOK EVENT (NEGATIVE DELTA) ====")
            print(ob_event.to_frame().T.to_string(index=False))
            print(f"\nTime Window: {start_ts}  -->  {end_ts}")
            print("\n==== 2. MATCHING BLOCKCHAIN TRANSACTIONS IN 6s ====")
            print(tx_events.to_string(index=False))
            print("\n--> PERFECT MATCH FOUND! This was a True Execution.")
            break
