-- 06_markets_universe.sql: the union of the market ids seen in the five sources, with
-- one flag per source, and the subset that the Gamma download does not cover.
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cell 18.
-- Expected: markets_universe 175,476 rows, markets_missing 59,617 rows.
--
-- After this file, run
--   scripts/preprocessing/00_legacy_raw_downloads/scrape_missing_markets_by_condition_id.py --db-path <db>
-- which asks gamma-api.polymarket.com/markets?condition_ids=... (closed=false, then
-- closed=true) for every row of markets_missing, falls back to
-- clob.polymarket.com/clob-markets for the ids Gamma does not know, and fills
-- missing_fetched_markets (59,433 rows), missing_fetched_clob_markets (184 rows) and
-- missing_fetched_markets_status (59,617 rows). The raw answers are kept as JSONL under
-- <db_dir>/missing_fetched_markets_jsonl and <db_dir>/missing_fetched_clob_markets_jsonl.
-- Then continue with 07.

BEGIN TRANSACTION;

DROP TABLE IF EXISTS markets_universe;
DROP TABLE IF EXISTS markets_missing;

CREATE TABLE markets_universe AS
WITH all_markets AS (
    SELECT DISTINCT TRIM(CAST(market_id AS VARCHAR)) AS market_id, 'own_cleaned' AS source_table
    FROM markets_own_cleaned
    WHERE market_id IS NOT NULL AND TRIM(CAST(market_id AS VARCHAR)) <> ''

    UNION ALL

    SELECT DISTINCT TRIM(CAST(market_id AS VARCHAR)) AS market_id, 'transactions' AS source_table
    FROM markets_transactions
    WHERE market_id IS NOT NULL AND TRIM(CAST(market_id AS VARCHAR)) <> ''

    UNION ALL

    SELECT DISTINCT TRIM(CAST(market_id AS VARCHAR)) AS market_id, 'orderbook' AS source_table
    FROM markets_orderbook
    WHERE market_id IS NOT NULL AND TRIM(CAST(market_id AS VARCHAR)) <> ''

    UNION ALL

    SELECT DISTINCT TRIM(CAST(market_id AS VARCHAR)) AS market_id, 'snapshots' AS source_table
    FROM markets_snapshots
    WHERE market_id IS NOT NULL AND TRIM(CAST(market_id AS VARCHAR)) <> ''

    UNION ALL

    SELECT DISTINCT TRIM(CAST(market_id AS VARCHAR)) AS market_id, 'kaggle' AS source_table
    FROM markets_kaggle
    WHERE market_id IS NOT NULL AND TRIM(CAST(market_id AS VARCHAR)) <> ''
)
SELECT
    market_id,

    BOOL_OR(source_table = 'own_cleaned') AS in_own_cleaned,
    BOOL_OR(source_table = 'transactions') AS in_transactions,
    BOOL_OR(source_table = 'orderbook') AS in_orderbook,
    BOOL_OR(source_table = 'snapshots') AS in_snapshots,
    BOOL_OR(source_table = 'kaggle') AS in_kaggle,

    COUNT(DISTINCT source_table) AS source_count,

    CURRENT_TIMESTAMP AS created_at
FROM all_markets
GROUP BY market_id;

CREATE TABLE markets_missing AS
SELECT *
FROM markets_universe
WHERE in_own_cleaned = FALSE;

CREATE INDEX IF NOT EXISTS idx_markets_universe_market_id
ON markets_universe(market_id);

CREATE INDEX IF NOT EXISTS idx_markets_missing_market_id
ON markets_missing(market_id);

COMMIT;
