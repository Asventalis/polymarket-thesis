import duckdb
import time

print("Connecting to preprocessing.duckdb...")
# Must connect with read_only=False to alter the table
con = duckdb.connect("data/preprocessing.duckdb", read_only=False)

print("Scanning orderbook, snapshots, and transactions datasets...")
print("This may take a minute or two as it scans the entire partitioned storage...")
t0 = time.time()

# We union the timestamp columns from all 3 massive datasets and find the global min and max per market
query = """
    CREATE OR REPLACE TABLE temp_market_bounds AS
    WITH all_events AS (
        SELECT market_id, timestamp_received AS ts FROM read_parquet('data/derived/partitioned/orderbook/*/*/*/*/*.parquet')
        UNION ALL
        SELECT market_id, timestamp_received AS ts FROM read_parquet('data/derived/partitioned/snapshots/*/*/*/*/*.parquet')
        UNION ALL
        SELECT market_id, timestamp AS ts FROM read_parquet('data/derived/partitioned/transactions/*/*/*/*/*.parquet')
    )
    SELECT market_id, MIN(ts) AS first_event_timestamp, MAX(ts) AS last_event_timestamp
    FROM all_events
    GROUP BY market_id;
"""
con.execute(query)
print(f"Aggregation completed in {time.time()-t0:.2f}s!")

print("Augmenting universe.markets...")

# Safely add the columns (ignoring if they already exist from a previous run)
try:
    con.execute("ALTER TABLE universe.markets ADD COLUMN first_event_timestamp TIMESTAMP;")
except duckdb.CatalogException:
    pass

try:
    con.execute("ALTER TABLE universe.markets ADD COLUMN last_event_timestamp TIMESTAMP;")
except duckdb.CatalogException:
    pass

# Update the universe.markets table using the bounds we just calculated
update_query = """
    UPDATE universe.markets 
    SET 
        first_event_timestamp = temp_market_bounds.first_event_timestamp,
        last_event_timestamp = temp_market_bounds.last_event_timestamp
    FROM temp_market_bounds 
    WHERE universe.markets.market_id = temp_market_bounds.market_id;
"""
con.execute(update_query)

# Clean up
con.execute("DROP TABLE temp_market_bounds;")
con.close()

print("Successfully augmented universe.markets with 'first_event_timestamp' and 'last_event_timestamp'!")
