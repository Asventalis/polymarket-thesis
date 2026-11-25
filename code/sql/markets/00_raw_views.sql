-- 00_raw_views.sql: views over the raw inputs. Nothing is copied into the database.
--
--   orderbook    : Kaggle archive, archive/orderbook/orderbook_YYYY-MM-DD.parquet
--   snapshots    : Kaggle archive, archive/snapshots/snapshots_YYYY-MM-DD.parquet
--   transactions : daily parquet files written by
--                  scripts/preprocessing/00_legacy_raw_downloads/modified_cli_scraper/preprocessing.py
--                  from the CSV output of the modified poly-trade-scan fork.
--
-- Source: view definitions stored in data/preprocessing.duckdb (duckdb_views()) and the
-- DuckDB UI notebook "Users_checks" (16 Jun 2026). The notebooks of June 2026 called the
-- first two views raw_orderbook / raw_snapshots; the final database calls them
-- orderbook / snapshots, and so do the files that follow.
--
-- Placeholders ($NAME) are replaced by scripts/preprocessing/00_build_markets_tables.py.

CREATE OR REPLACE VIEW orderbook AS
SELECT *
FROM read_parquet('$ORDERBOOK_GLOB', union_by_name = true);

CREATE OR REPLACE VIEW snapshots AS
SELECT *
FROM read_parquet('$SNAPSHOTS_GLOB', union_by_name = true);

CREATE OR REPLACE VIEW transactions AS
SELECT *
FROM read_parquet('$TRANSACTIONS_GLOB', union_by_name = true);
