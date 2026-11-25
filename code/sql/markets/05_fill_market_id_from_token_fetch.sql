-- 05_fill_market_id_from_token_fetch.sql: fill the market_id of the fills whose token
-- was resolved by fetch_polymarket_markets_by_token.py, then list the distinct markets
-- that have at least one fill.
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cells 14 and 17.
-- Expected: markets_transactions has 157,776 rows in the final database.

WITH token_to_market AS (
    SELECT
        condition_id AS market_id,
        primary_token_id AS token_id
    FROM markets_fetched_from_token_transactions
    WHERE primary_token_id IS NOT NULL

    UNION ALL

    SELECT
        condition_id AS market_id,
        secondary_token_id AS token_id
    FROM markets_fetched_from_token_transactions
    WHERE secondary_token_id IS NOT NULL
)
UPDATE transactions_with_market AS t
SET market_id = m.market_id
FROM token_to_market AS m
WHERE t.market_id IS NULL
  AND t.token_id = m.token_id;

CREATE OR REPLACE TABLE markets_transactions AS
SELECT DISTINCT market_id
FROM transactions_with_market;
