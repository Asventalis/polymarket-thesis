import duckdb
import time

print("Building Market Lifecycle Event Queue using last_transaction_timestamp...")
t0 = time.time()
con = duckdb.connect("data/preprocessing.duckdb", read_only=True)

query = """
    COPY (
        WITH open_events AS (
            SELECT 
                deploying_timestamp AS event_timestamp, 
                market_id, 
                'OPEN' AS event_type
            FROM universe.markets
            WHERE deploying_timestamp IS NOT NULL
        ),
        close_events AS (
            SELECT 
                last_transaction_timestamp AS event_timestamp, 
                market_id, 
                'CLOSE' AS event_type
            FROM universe.markets
            WHERE last_transaction_timestamp IS NOT NULL
        ),
        all_events AS (
            SELECT * FROM open_events
            UNION ALL
            SELECT * FROM close_events
        )
        SELECT * FROM all_events 
        ORDER BY event_timestamp ASC
    ) TO 'data/derived/market_lifecycle_events.parquet' (FORMAT PARQUET);
"""
con.execute(query)
print(f"Lifecycle Event Queue built and saved to data/derived/market_lifecycle_events.parquet in {time.time()-t0:.2f}s!")
