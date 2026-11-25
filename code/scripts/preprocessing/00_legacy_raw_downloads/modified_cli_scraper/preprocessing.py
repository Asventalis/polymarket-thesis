#!/usr/bin/env python3

import duckdb
from datetime import datetime, timedelta

conn = duckdb.connect(":memory:")

DATA_PATH = "/home/vitopampinella/Desktop/Magistrale/Tesi/Project/modified_cli_scraper/data"

conn.execute(f"""
CREATE TABLE transactions AS
SELECT
    CAST(block_number AS BIGINT) AS block_number,
    CAST(timestamp AS TIMESTAMP) AS timestamp,
    CAST(tx_hash AS VARCHAR) AS tx_hash,
    CAST(order_index AS INTEGER) AS order_index,
    CAST(wallet AS VARCHAR) AS wallet,
    CAST(role AS VARCHAR) AS role,
    CAST(token_id AS VARCHAR) AS token_id,
    CAST(side AS VARCHAR) AS side,
    CAST(tokens AS DOUBLE) AS tokens,
    CAST(price AS DOUBLE) AS price,
    CAST(total_usdc AS DOUBLE) AS total_usdc,
    CAST(fee_rate AS DOUBLE) AS fee_rate
FROM read_csv(
    '{DATA_PATH}/*.csv',
    header = true,
    all_varchar = true
)
""")

current_date = datetime(2026, 3, 6)
end_date = datetime(2026, 3, 26)

while current_date <= end_date:
    print(f"Processing {current_date.strftime('%Y-%m-%d')}...")
    date_str = current_date.strftime("%Y-%m-%d")
    next_date = current_date + timedelta(days=1)

    output_file = f"{DATA_PATH}/transactions_{date_str}.parquet"

    row_count = conn.execute(f"""
        SELECT COUNT(*)
        FROM transactions
        WHERE timestamp >= TIMESTAMP '{date_str}'
          AND timestamp <  TIMESTAMP '{next_date.strftime("%Y-%m-%d")}'
    """).fetchone()[0]

    if row_count > 0:
        conn.execute(f"""
            COPY (
                SELECT *
                FROM transactions
                WHERE timestamp >= TIMESTAMP '{date_str}'
                  AND timestamp <  TIMESTAMP '{next_date.strftime("%Y-%m-%d")}'
                ORDER BY block_number, tx_hash, order_index
            )
            TO '{output_file}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        print(f"✓ {date_str}: {row_count:,} transactions")

    current_date = next_date

conn.close()