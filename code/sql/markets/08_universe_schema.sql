-- 08_universe_schema.sql: the "universe" schema read by the rest of the pipeline
-- (scripts/preprocessing/01_build_multi_day_aggregates.py and
-- 02_build_hourly_partitions.py read universe.transactions and universe.markets).
-- The two stream views convert the millisecond epochs to timestamps.
--
-- Source: DuckDB UI notebook "Clean market queries" (21 Jul 2026), cell 28.

CREATE SCHEMA IF NOT EXISTS universe;

CREATE OR REPLACE VIEW universe.markets AS
SELECT *
FROM main.markets_cleaned_full;

CREATE OR REPLACE VIEW universe.transactions AS
SELECT *
FROM main.transactions_with_market;

CREATE OR REPLACE VIEW universe.snapshots AS
SELECT * REPLACE (
  make_timestamp_ms(timestamp_received) AS timestamp_received,
  make_timestamp_ms(timestamp_created_at) AS timestamp_created_at
)
FROM main.snapshots;

CREATE OR REPLACE VIEW universe.orderbook AS
SELECT * REPLACE (
  make_timestamp_ms(timestamp_received) AS timestamp_received,
  make_timestamp_ms(timestamp_created_at) AS timestamp_created_at
)
FROM main.orderbook;
