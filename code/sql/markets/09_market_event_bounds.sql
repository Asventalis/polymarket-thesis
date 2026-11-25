-- 09_market_event_bounds.sql: three per-market timestamps added to markets_cleaned_full
-- after the hourly partitions exist (scripts/preprocessing/02_build_hourly_partitions.py):
--
--   first_event_timestamp      : earliest order-book, snapshot or fill timestamp
--   last_event_timestamp       : latest of the same
--   last_transaction_timestamp : latest fill timestamp
--
-- This is the SQL of the two root-level scripts add_market_bounds.py and
-- add_last_tx_timestamp.py (August 2026), which scan data/derived/partitioned. The
-- three columns are the last three of markets_cleaned_full in the final database and
-- feed build_lifecycle_events.py (market_lifecycle_events.parquet: OPEN at
-- deploying_timestamp, CLOSE at last_transaction_timestamp). Not re-run on 29 Oct 2026:
-- the partitions live only on the server.

CREATE OR REPLACE TABLE temp_market_bounds AS
WITH all_events AS (
    SELECT market_id, timestamp_received AS ts
    FROM read_parquet('$PARTITIONED_DIR/orderbook/*/*/*/*/*.parquet')
    UNION ALL
    SELECT market_id, timestamp_received AS ts
    FROM read_parquet('$PARTITIONED_DIR/snapshots/*/*/*/*/*.parquet')
    UNION ALL
    SELECT market_id, timestamp AS ts
    FROM read_parquet('$PARTITIONED_DIR/transactions/*/*/*/*/*.parquet')
)
SELECT market_id, MIN(ts) AS first_event_timestamp, MAX(ts) AS last_event_timestamp
FROM all_events
GROUP BY market_id;

CREATE OR REPLACE TABLE temp_last_tx AS
SELECT market_id, MAX(timestamp) AS last_tx_timestamp
FROM read_parquet('$PARTITIONED_DIR/transactions/*/*/*/*/*.parquet')
GROUP BY market_id;

ALTER TABLE main.markets_cleaned_full ADD COLUMN IF NOT EXISTS first_event_timestamp TIMESTAMP;
ALTER TABLE main.markets_cleaned_full ADD COLUMN IF NOT EXISTS last_event_timestamp TIMESTAMP;
ALTER TABLE main.markets_cleaned_full ADD COLUMN IF NOT EXISTS last_transaction_timestamp TIMESTAMP;

UPDATE main.markets_cleaned_full AS m
SET first_event_timestamp = b.first_event_timestamp,
    last_event_timestamp = b.last_event_timestamp
FROM temp_market_bounds AS b
WHERE m.market_id = b.market_id;

UPDATE main.markets_cleaned_full AS m
SET last_transaction_timestamp = t.last_tx_timestamp
FROM temp_last_tx AS t
WHERE m.market_id = t.market_id;

DROP TABLE temp_market_bounds;
DROP TABLE temp_last_tx;
