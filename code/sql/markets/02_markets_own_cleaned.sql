-- 02_markets_own_cleaned.sql: flatten the Gamma rows into the market schema used
-- everywhere else (one row per market, snake_case names, timestamps cast, outcome and
-- token ids extracted from the JSON arrays, tags as a comma-separated string).
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cell 8. The notebook
-- read the tags from a table named market_tags; the final database calls the same
-- table closed_market_tags (verified on 29 Oct 2026: re-aggregating closed_market_tags
-- reproduces the tags column of all 115,859 rows).
--
-- Expected row count: 115,859 (same as closed_markets).

CREATE OR REPLACE TABLE markets_own_cleaned AS
WITH tags_distinct AS (
    SELECT DISTINCT
        market_id,
        slug
    FROM closed_market_tags
    WHERE slug IS NOT NULL
),
tags_agg AS (
    SELECT
        market_id,
        string_agg(slug, ',' ORDER BY slug) AS tags
    FROM tags_distinct
    GROUP BY market_id
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

FROM closed_markets AS m
LEFT JOIN tags_agg AS t
    ON CAST(m.id AS VARCHAR) = t.market_id;
