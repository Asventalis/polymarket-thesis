import duckdb
import time

print("Connecting to preprocessing.duckdb...")
con = duckdb.connect("data/preprocessing.duckdb", read_only=False)

print("Scanning transactions dataset for last transaction timestamps...")
t0 = time.time()

query = """
    CREATE OR REPLACE TABLE temp_last_tx AS
    SELECT market_id, MAX(timestamp) AS last_tx_timestamp
    FROM read_parquet('data/derived/partitioned/transactions/*/*/*/*/*.parquet')
    GROUP BY market_id;
"""
con.execute(query)
print(f"Aggregation completed in {time.time()-t0:.2f}s!")

print("Augmenting main.markets_cleaned_full...")
try:
    con.execute("ALTER TABLE main.markets_cleaned_full ADD COLUMN last_transaction_timestamp TIMESTAMP;")
except duckdb.CatalogException:
    pass

update_query = """
    UPDATE main.markets_cleaned_full 
    SET last_transaction_timestamp = temp_last_tx.last_tx_timestamp
    FROM temp_last_tx 
    WHERE main.markets_cleaned_full.market_id = temp_last_tx.market_id;
"""
con.execute(update_query)
con.execute("DROP TABLE temp_last_tx;")
con.close()

print("Successfully augmented markets with 'last_transaction_timestamp'!")
