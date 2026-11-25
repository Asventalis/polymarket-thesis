-- 01_load_closed_markets.sql: load the Gamma API pages of closed markets.
--
-- Input: the JSONL pages written by
--   scripts/preprocessing/00_legacy_raw_downloads/download_polymarket_2025_own_all.py
-- (endpoint gamma-api.polymarket.com/markets/keyset, closed=true, include_tag=true,
--  end_date_min=2026-03-06, start_date_max=2026-03-26; one file per page,
--  closed_markets_NNNNN.jsonl). The final database holds 115,859 rows.
--
-- Source: legacy notebook gpt_preprocessing.ipynb, cell "1. Load closed market JSONL
-- files into DuckDB" (the same read_json_auto options are used by
-- scrape_missing_markets_by_condition_id.py).
--
-- After this file, run
--   scripts/preprocessing/00_legacy_raw_downloads/fetch_tags.py --db-path <db>
-- which fills closed_market_tags (one row per market and tag; 655,762 rows in the
-- final database) from gamma-api.polymarket.com/markets/<id>/tags.

CREATE OR REPLACE TABLE closed_markets AS
SELECT *
FROM read_json_auto(
    '$CLOSED_MARKETS_GLOB',
    format = 'newline_delimited',
    union_by_name = true,
    sample_size = -1,
    timestampformat = 'iso'
);
