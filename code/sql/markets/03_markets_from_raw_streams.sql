-- 03_markets_from_raw_streams.sql: the set of market ids seen in each raw stream,
-- with per-market aggregates, plus the Kaggle label file and the per-token
-- transaction counts.
--
--   markets_orderbook : one row per market_id in the Kaggle order-book stream (158,538)
--   markets_snapshots : one row per market_id in the Kaggle snapshot stream   (158,248)
--   markets_kaggle    : the Kaggle file labels/market_targets.parquet as is    (123,895)
--   token_ids         : one row per token_id in the scraped transactions (diagnostic)
--
-- Sources: DuckDB UI notebooks "Dates Comparison" (19 Jun 2026, cells 2, 4, 6) and
-- "Market_scraping_queries" (25 Jun 2026, cell 3); the first two also exist as
-- legacy_code/old_thesis_code/SQL/create_markets_from_raw_{orderbook,snapshots}.sql.
-- Each of these scans a whole raw stream once.
--
-- token_ids is not read by any later step. In the final database it has 378,887 rows
-- and covers 6-26 April 2026 (173.4 M fills): it was computed on 25 Jun 2026 from an
-- earlier transaction scrape, before the 6-26 March 2026 set that the rest of the
-- database uses, so its row count is not reproducible from the March files.

CREATE OR REPLACE TABLE markets_orderbook AS
WITH grouped AS (
    SELECT
        market_id,

        COUNT(*) AS n_rows,
        COUNT(DISTINCT token_id) AS n_tokens,

        MIN(timestamp_received) AS min_timestamp_received_ms,
        MAX(timestamp_received) AS max_timestamp_received_ms,

        MIN(timestamp_created_at) AS min_timestamp_created_at_ms,
        MAX(timestamp_created_at) AS max_timestamp_created_at_ms,

        MIN(best_bid) AS min_best_bid,
        MAX(best_bid) AS max_best_bid,
        AVG(best_bid) AS avg_best_bid,

        MIN(best_ask) AS min_best_ask,
        MAX(best_ask) AS max_best_ask,
        AVG(best_ask) AS avg_best_ask,

        MIN(spread) AS min_spread,
        MAX(spread) AS max_spread,
        AVG(spread) AS avg_spread,

        MIN(mid_price) AS min_mid_price,
        MAX(mid_price) AS max_mid_price,
        AVG(mid_price) AS avg_mid_price,

        MIN(change_price) AS min_change_price,
        MAX(change_price) AS max_change_price,
        AVG(change_price) AS avg_change_price,

        MIN(change_size) AS min_change_size,
        MAX(change_size) AS max_change_size,
        AVG(change_size) AS avg_change_size,
        SUM(change_size) AS sum_change_size,

        COUNT(*) FILTER (WHERE change_side = 'BUY') AS n_buy_changes,
        COUNT(*) FILTER (WHERE change_side = 'SELL') AS n_sell_changes,
        COUNT(*) FILTER (WHERE change_size = 0) AS n_zero_size_changes

    FROM orderbook
    GROUP BY market_id
)
SELECT
    market_id,

    n_rows,
    n_tokens,

    min_timestamp_received_ms,
    max_timestamp_received_ms,
    make_timestamp_ms(min_timestamp_received_ms) AS min_timestamp_received,
    make_timestamp_ms(max_timestamp_received_ms) AS max_timestamp_received,

    min_timestamp_created_at_ms,
    max_timestamp_created_at_ms,
    make_timestamp_ms(min_timestamp_created_at_ms) AS min_timestamp_created_at,
    make_timestamp_ms(max_timestamp_created_at_ms) AS max_timestamp_created_at,

    max_timestamp_received_ms - min_timestamp_received_ms AS duration_received_ms,
    make_timestamp_ms(max_timestamp_received_ms)
        - make_timestamp_ms(min_timestamp_received_ms) AS duration_received,

    max_timestamp_created_at_ms - min_timestamp_created_at_ms AS duration_created_at_ms,
    make_timestamp_ms(max_timestamp_created_at_ms)
        - make_timestamp_ms(min_timestamp_created_at_ms) AS duration_created_at,

    min_best_bid,
    max_best_bid,
    avg_best_bid,

    min_best_ask,
    max_best_ask,
    avg_best_ask,

    min_spread,
    max_spread,
    avg_spread,

    min_mid_price,
    max_mid_price,
    avg_mid_price,

    min_change_price,
    max_change_price,
    avg_change_price,

    min_change_size,
    max_change_size,
    avg_change_size,
    sum_change_size,

    n_buy_changes,
    n_sell_changes,
    n_zero_size_changes

FROM grouped;

CREATE OR REPLACE TABLE markets_snapshots AS
WITH grouped AS (
    SELECT
        market_id,

        COUNT(*) AS n_rows,

        MIN(timestamp_received) AS min_timestamp_received_ms,
        MAX(timestamp_received) AS max_timestamp_received_ms,

        MIN(timestamp_created_at) AS min_timestamp_created_at_ms,
        MAX(timestamp_created_at) AS max_timestamp_created_at_ms

    FROM snapshots
    GROUP BY market_id
)
SELECT
    market_id,

    n_rows,

    min_timestamp_received_ms,
    max_timestamp_received_ms,
    make_timestamp_ms(min_timestamp_received_ms) AS min_timestamp_received,
    make_timestamp_ms(max_timestamp_received_ms) AS max_timestamp_received,

    min_timestamp_created_at_ms,
    max_timestamp_created_at_ms,
    make_timestamp_ms(min_timestamp_created_at_ms) AS min_timestamp_created_at,
    make_timestamp_ms(max_timestamp_created_at_ms) AS max_timestamp_created_at,

    max_timestamp_received_ms - min_timestamp_received_ms AS duration_received_ms,
    make_timestamp_ms(max_timestamp_received_ms)
        - make_timestamp_ms(min_timestamp_received_ms) AS duration_received,

    max_timestamp_created_at_ms - min_timestamp_created_at_ms AS duration_created_at_ms,
    make_timestamp_ms(max_timestamp_created_at_ms)
        - make_timestamp_ms(min_timestamp_created_at_ms) AS duration_created_at

FROM grouped;

CREATE OR REPLACE TABLE markets_kaggle AS
SELECT *
FROM read_parquet('$KAGGLE_LABELS_PARQUET');

CREATE OR REPLACE TABLE token_ids AS
SELECT
    token_id,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT tx_hash) AS n_txs,
    COUNT(DISTINCT wallet) AS n_wallets,
    MIN(timestamp) AS first_seen,
    MAX(timestamp) AS last_seen,
    SUM(tokens) AS gross_tokens,
    SUM(total_usdc) AS gross_usdc
FROM transactions
WHERE token_id IS NOT NULL
GROUP BY token_id
ORDER BY n_rows DESC;
