# sql/markets: the market tables of `data/preprocessing.duckdb`

Recovered on 28–29 October 2026 from the DuckDB UI notebooks (`docs/duckdb_ui_notebooks_export_2026-10-28.md`),
the legacy notebooks and the view definitions stored in the database. Each file names its source cell.
Run them in order with `scripts/preprocessing/00_build_markets_tables.py`, which replaces the `$PLACEHOLDERS`
with the paths given on the command line and runs each file in one DuckDB connection.

| step | file | creates | rows (4 Aug 2026 db) | then run |
|---|---|---|---|---|
| 00 | `00_raw_views.sql` | views `orderbook`, `snapshots`, `transactions` | | |
| 01 | `01_load_closed_markets.sql` | `closed_markets` | 115,859 | `00_legacy_raw_downloads/fetch_tags.py` → `closed_market_tags` (655,762) |
| 02 | `02_markets_own_cleaned.sql` | `markets_own_cleaned` | 115,859 | |
| 03 | `03_markets_from_raw_streams.sql` | `markets_orderbook`, `markets_snapshots`, `markets_kaggle`, `token_ids` | 158,538 / 158,248 / 123,895 / see note | |
| 04 | `04_transactions_with_market.sql` | `transactions_with_market`, `unmatched_token_ids` | 218,882,492 | `00_legacy_raw_downloads/fetch_polymarket_markets_by_token.py` → `markets_fetched_from_token_transactions` (22,858) |
| 05 | `05_fill_market_id_from_token_fetch.sql` | fills `market_id`, `markets_transactions` | 157,776 | |
| 06 | `06_markets_universe.sql` | `markets_universe`, `markets_missing` | 175,476 / 59,617 | `00_legacy_raw_downloads/scrape_missing_markets_by_condition_id.py` → `missing_fetched_markets` (59,433), `missing_fetched_clob_markets` (184), `missing_fetched_markets_status` (59,617) |
| 07 | `07_markets_cleaned_full.sql` | `missing_fetched_markets_cleaned`, `markets_cleaned_full`, `clob_only_market_coverage` | 59,433 / 175,292 / 184 | |
| 08 | `08_universe_schema.sql` | schema `universe` (4 views) | | |
| 09 | `09_market_event_bounds.sql` | 3 timestamp columns on `markets_cleaned_full` | | needs `data/derived/partitioned` (after `02_build_hourly_partitions.py`) |

`markets_cleaned_full` (= `universe.markets`) is exported as `data/universe_markets.parquet` (175,292 rows, 6 Aug 2026).

## Inputs

* Kaggle archive (version 3 of 27 March 2026): `archive/orderbook/*.parquet`, `archive/snapshots/*.parquet`,
  `labels/market_targets.parquet`.
* Scraped fills: the daily `transactions_YYYY-MM-DD.parquet` files written by
  `00_legacy_raw_downloads/modified_cli_scraper/preprocessing.py` (6–26 March 2026).
* Gamma pages of closed markets: `closed_markets_NNNNN.jsonl` written by
  `00_legacy_raw_downloads/download_polymarket_2025_own_all.py`.
* The three API scrapers above, which write straight into the database.

The raw JSONL pages (Gamma closed markets, missing markets, CLOB markets) are no longer on disk; their content
survives only as the tables `closed_markets`, `closed_market_tags`, `missing_fetched_markets`,
`missing_fetched_clob_markets`, `markets_fetched_from_token_transactions` of `data/preprocessing.duckdb`.
A fresh download would return today's state of the Gamma API, not the state of June–July 2026.

## Verification (29 October 2026)

On a scratch copy of the input tables (`closed_markets`, `closed_market_tags`, the scraper outputs,
`markets_orderbook`, `markets_snapshots`, `markets_kaggle`, `markets_transactions`), steps 02, 06, 07 and 08
were re-run with the runner and compared with the 4 August database: `markets_own_cleaned`, `markets_universe`,
`markets_missing`, `missing_fetched_markets_cleaned`, `markets_cleaned_full` and `clob_only_market_coverage`
are identical row by row (`EXCEPT` in both directions returns 0; `created_at` of the universe tables and the three
columns of step 09 excluded). Steps 00, 01, 03, 04, 05 and 09 could not be re-run locally (the raw files are
gone from this machine); their SQL is the notebook text, and the token logic of 04–05 was checked on the final
table: all 218,882,492 fills have a `market_id`, 157,776 distinct markets, 314,656 distinct token ids.

Note on `token_ids` (step 03): the table in the 4 August database (378,887 rows) was built on 25 June 2026 from an
earlier scrape covering 6–26 April 2026 (173.4 M fills), not the March set used by everything else. It is
diagnostic only; no later step reads it.

## Notebook names vs database names

* `raw_orderbook` / `raw_snapshots` in the June notebooks are the views `orderbook` / `snapshots`.
* `market_tags` in the July notebook is `closed_market_tags` (checked: re-aggregating it reproduces the
  `tags` column of all 115,859 rows of `markets_own_cleaned`).
* `markets_with_tags` (cell 9 of "Clean market queries") was an intermediate that the final database does not keep.
