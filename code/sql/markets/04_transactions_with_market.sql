-- 04_transactions_with_market.sql: attach a market_id to every scraped fill through
-- the two CLOB token ids of each Gamma market, and list the token ids that no Gamma
-- market explains.
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cells 11 and 13.
-- transactions_with_market materialises the whole transaction stream (218,882,492
-- rows in the final database, about 11 GB on disk).
--
-- After this file, run
--   scripts/preprocessing/00_legacy_raw_downloads/fetch_polymarket_markets_by_token.py --db-path <db>
-- which asks clob.polymarket.com/markets-by-token/<token_id> for every row of
-- unmatched_token_ids, fills markets_fetched_from_token_transactions (condition_id,
-- primary_token_id, secondary_token_id; 22,858 rows in the final database) and deletes
-- the resolved tokens from unmatched_token_ids. Then continue with 05.

CREATE OR REPLACE TABLE transactions_with_market AS
WITH token_to_market AS (
    SELECT
        market_id,
        token_id_1 AS token_id
    FROM markets_own_cleaned
    WHERE token_id_1 IS NOT NULL

    UNION ALL

    SELECT
        market_id,
        token_id_2 AS token_id
    FROM markets_own_cleaned
    WHERE token_id_2 IS NOT NULL
)
SELECT
    t.*,
    m.market_id
FROM transactions AS t
LEFT JOIN token_to_market AS m
    ON t.token_id = m.token_id;

CREATE OR REPLACE TABLE unmatched_token_ids AS
SELECT DISTINCT token_id
FROM transactions_with_market
WHERE market_id IS NULL;
