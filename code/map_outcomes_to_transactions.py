import duckdb
import os
import shutil
import time

print("Starting Transaction Partition Mapping...")
t0 = time.time()

# We connect to preprocessing.duckdb to access universe.markets
con = duckdb.connect("data/preprocessing.duckdb", read_only=True)

input_dir = "data/derived/partitioned/transactions"
output_dir = "data/derived/partitioned/transactions_mapped"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("Executing distributed DuckDB COPY query to append token_outcome...")

# DuckDB's COPY command can read all partitioned parquets, perform the JOIN, 
# and stream the results directly back out into a new hive-partitioned directory tree
query = f"""
COPY (
    SELECT 
        t.block_number,
        t.timestamp,
        t.tx_hash,
        t.order_index,
        t.wallet,
        t.role,
        t.token_id,
        CASE 
            WHEN t.token_id = m.token_id_1 THEN 1
            WHEN t.token_id = m.token_id_2 THEN -1
            ELSE 0
        END AS token_direction,
        t.side,
        t.tokens,
        t.price,
        t.total_usdc,
        t.fee_rate,
        t.market_id,
        t.year,
        t.month,
        t.day,
        t.hour
    FROM read_parquet('{input_dir}/*/*/*/*/*.parquet', hive_partitioning=true) t
    LEFT JOIN universe.markets m ON t.market_id = m.market_id
) TO '{output_dir}' (FORMAT PARQUET, PARTITION_BY (year, month, day, hour))
"""

con.execute(query)

print(f"Dataset successfully mapped and written to {output_dir} in {time.time()-t0:.2f}s!")

# Replace the old directory with the new one
print("Swapping directories...")
shutil.rmtree(input_dir)
os.rename(output_dir, input_dir)

print("Done! All 1-hour transaction parquet files now have the 'token_outcome' column.")
