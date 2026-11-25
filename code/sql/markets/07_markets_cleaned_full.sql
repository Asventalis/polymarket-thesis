-- 07_markets_cleaned_full.sql: clean the markets fetched by condition id into the same
-- schema as markets_own_cleaned, append the ones the Gamma download did not have, and
-- measure how much data the 184 CLOB-only markets carry.
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cells 21, 24, 26.
-- Expected: missing_fetched_markets_cleaned 59,433 rows, markets_cleaned_full 175,292
-- rows, clob_only_market_coverage 184 rows.
--
-- The tags of these markets come embedded in the Gamma answer (a list of structs),
-- not from the /tags endpoint.

CREATE OR REPLACE TABLE missing_fetched_markets_cleaned AS
WITH tags_distinct AS (
    SELECT DISTINCT
        CAST(m.id AS VARCHAR) AS id,
        tag.slug AS slug
    FROM missing_fetched_markets AS m
    CROSS JOIN UNNEST(m.tags) AS u(tag)
    WHERE tag.slug IS NOT NULL
),
tags_agg AS (
    SELECT
        id,
        string_agg(slug, ',' ORDER BY slug) AS tags
    FROM tags_distinct
    GROUP BY id
)
SELECT
    -- id, conditionId, questionID
    m.id,
    m.conditionId AS market_id,
    m.questionID AS question_id,

    -- tags
    COALESCE(t.tags, '') AS tags,

    -- question, slug, description, resolutionSource
    m.question,
    m.slug,
    m.description,
    m.resolutionSource AS resolution_source,

    -- dates
    CAST(m.endDate AS TIMESTAMP) AS end_date,
    CAST(m.startDate AS TIMESTAMP) AS start_date,
    CAST(m.createdAt AS TIMESTAMP) AS created_at,
    CAST(m.closedTime AS TIMESTAMP) AS closed_time,
    CAST(m.umaEndDate AS TIMESTAMP) AS uma_end_date,
    CAST(m.endDateIso AS TIMESTAMP) AS end_date_iso,
    CAST(m.startDateIso AS TIMESTAMP) AS start_date_iso,
    CAST(m.acceptingOrdersTimestamp AS TIMESTAMP) AS accepting_orders_timestamp,
    CAST(m.deployingTimestamp AS TIMESTAMP) AS deploying_timestamp,

    -- outcomes, outcomePrices, clobTokenIds
    m.outcomes ->> '$[0]' AS outcome_1,
    m.outcomes ->> '$[1]' AS outcome_2,

    CAST(m.outcomePrices ->> '$[0]' AS FLOAT) AS outcome_price_1,
    CAST(m.outcomePrices ->> '$[1]' AS FLOAT) AS outcome_price_2,

    m.clobTokenIds ->> '$[0]' AS token_id_1,
    m.clobTokenIds ->> '$[1]' AS token_id_2,

    -- gameStartTime, sportsMarketType, gameId
    CAST(m.gameStartTime AS TIMESTAMP) AS game_start_time,
    m.sportsMarketType AS sports_market_type,
    m.gameId AS game_id

FROM missing_fetched_markets AS m
LEFT JOIN tags_agg AS t
    ON CAST(m.id AS VARCHAR) = t.id;

CREATE OR REPLACE TABLE markets_cleaned_full AS
SELECT *
FROM markets_own_cleaned

UNION ALL

SELECT m.*
FROM missing_fetched_markets_cleaned AS m
ANTI JOIN markets_own_cleaned AS o
    ON LOWER(CAST(m.market_id AS VARCHAR)) = LOWER(CAST(o.market_id AS VARCHAR));

CREATE OR REPLACE TABLE clob_only_market_coverage AS
WITH clob_markets AS (
    SELECT DISTINCT LOWER(CAST(c AS VARCHAR)) AS market_id
    FROM missing_fetched_clob_markets
),

tx_counts AS (
    SELECT
        LOWER(CAST(tx.market_id AS VARCHAR)) AS market_id,
        COUNT(*) AS transaction_rows
    FROM transactions_with_market AS tx
    INNER JOIN clob_markets AS cm
        ON LOWER(CAST(tx.market_id AS VARCHAR)) = cm.market_id
    GROUP BY LOWER(CAST(tx.market_id AS VARCHAR))
),

ob_counts AS (
    SELECT
        LOWER(CAST(ob.market_id AS VARCHAR)) AS market_id,
        COUNT(*) AS orderbook_rows
    FROM markets_orderbook AS ob
    INNER JOIN clob_markets AS cm
        ON LOWER(CAST(ob.market_id AS VARCHAR)) = cm.market_id
    GROUP BY LOWER(CAST(ob.market_id AS VARCHAR))
),

snap_counts AS (
    SELECT
        LOWER(CAST(s.market_id AS VARCHAR)) AS market_id,
        COUNT(*) AS snapshot_rows
    FROM markets_snapshots AS s
    INNER JOIN clob_markets AS cm
        ON LOWER(CAST(s.market_id AS VARCHAR)) = cm.market_id
    GROUP BY LOWER(CAST(s.market_id AS VARCHAR))
)

SELECT
    cm.market_id,
    COALESCE(tx.transaction_rows, 0) AS transaction_rows,
    COALESCE(ob.orderbook_rows, 0) AS orderbook_rows,
    COALESCE(sn.snapshot_rows, 0) AS snapshot_rows
FROM clob_markets AS cm
LEFT JOIN tx_counts AS tx
    ON cm.market_id = tx.market_id
LEFT JOIN ob_counts AS ob
    ON cm.market_id = ob.market_id
LEFT JOIN snap_counts AS sn
    ON cm.market_id = sn.market_id;
