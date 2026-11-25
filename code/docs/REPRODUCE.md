# Reproducing the data pipeline

State on 29 October 2026. Paths are relative to this repository (`agy_project/`); on the UniPi server the
repository is `/data/v.pampinella/agy_project` and GPU scripts run through `./run.sh` / `./run_dev.sh`
(Apptainer + uv). Row counts and file sizes are those of the author's copies.

## 0. Public inputs

| input | where | what |
|---|---|---|
| Polymarket Tick-Level Orderbook Dataset, version 3 (27 Mar 2026), CC BY-NC 4.0 | kaggle.com/datasets/marvingozo/polymarket-tick-level-orderbook-dataset | `archive/orderbook/orderbook_YYYY-MM-DD.parquet`, `archive/snapshots/snapshots_YYYY-MM-DD.parquet` (Yes side only), `labels/market_targets.parquet` |
| Polygon blockchain, CTF Exchange `matchOrders` calls, 6–26 March 2026 | public node `wss://polygon.drpc.org` | read by the fork in `scripts/preprocessing/00_legacy_raw_downloads/modified_cli_scraper/` (base: github.com/martkir/poly-trade-scan) |
| Polymarket Gamma API and CLOB API | gamma-api.polymarket.com, clob.polymarket.com | market metadata, tags, token → market |
| GDELT 2.0 GKG (news features only) | data.gdeltproject.org | 15-minute GKG files, `data/external/gdelt` (13 GB) |

## 1. Raw fills (`data/raw`-level, immutable)

1. `modified_cli_scraper` (CLI of the fork, `POLYGON_WSS_URL` in `.env`): decodes every `matchOrders` call of the
   block range, one CSV row per filled order (taker and makers, `role`, `order_index`, `fee_raw`).
2. `modified_cli_scraper/preprocessing.py`: CSV → `transactions_YYYY-MM-DD.parquet` (DuckDB). The `DATA_PATH`
   constant at its top is still an absolute path and must be edited.

## 2. Market tables (`data/preprocessing.duckdb`, 15 GB)

`sql/markets/` + `scripts/preprocessing/00_build_markets_tables.py`; see `sql/markets/README.md` for the order,
the three API scrapers that run between the SQL steps, the expected row counts and what was re-verified.
Result: `universe.markets` (175,292 markets; also `data/universe_markets.parquet`), `universe.transactions`
(218,882,492 fills with `market_id`), `universe.orderbook`, `universe.snapshots`.

## 3. Derived streams (`data/derived/`, server)

| step | script | reads | writes |
|---|---|---|---|
| 3.1 | `scripts/preprocessing/01_build_multi_day_aggregates.py` (runs `sql/03`–`06`) | `universe.*` | `derived/multi_day/{orderbook,transactions,targets,edges_2s,unified}` (2 s grid, 13 GB) |
| 3.2 | `scripts/preprocessing/02_build_hourly_partitions.py` | `universe.transactions`, raw order book and snapshots | `derived/partitioned/{orderbook,snapshots,transactions}/year=/month=/day=/hour=/data_0.parquet` (188 GB) |
| 3.3 | `scripts/preprocessing/03_compute_deltas.py` | partitioned order book | adds the `delta` column (book changes between consecutive events) |
| 3.4 | `map_outcomes_to_transactions.py` (repository root) | partitioned transactions + `universe.markets` | `token_direction` (Yes/No) on the fills |
| 3.5 | `sql/markets/09_market_event_bounds.sql` (= `add_market_bounds.py`, `add_last_tx_timestamp.py`) | partitioned streams | `first/last_event_timestamp`, `last_transaction_timestamp` on `markets_cleaned_full`; then `build_lifecycle_events.py` → `derived/market_lifecycle_events.parquet` |
| 3.6 | `src/polymarket_gnn/data/build_id_mappings.py` | partitioned transactions, `derived/market_node_mapping.csv` | `derived/id_mappings/{market,wallet}_to_idx.parquet` |
| 3.7 | `scripts/preprocessing/04_build_text_embeddings.py`, `05_extract_node_features.py`, `06_build_wallet_features.py` | `universe.markets`, multi_day | `derived/market_embeddings.pt`, node features, wallet-flow parquet |

## 4. Tensors and features (server)

| step | script | writes |
|---|---|---|
| 4.1 | `src/polymarket_gnn/data/lob_builder.py` | `derived/tensors/year=/month=/day=/hour=/lob_features.parquet`: one row per market every 20 book events, 52-feature book vector (27 GB) |
| 4.2 | `hand_features.py --days ...` | `derived/hand_features/day=*.parquet`: the 61 causal hand features at every tensor row (7.7 GB) |
| 4.3 | `tree_features.py --horizon H` | `derived/tree_logprobs_<H>`: cross-fitted LightGBM log-probabilities at every row (2.2 GB per horizon) |
| 4.4 | `tsfm_cache.py` | `derived/tsfm/chronos2_1h`: zero-shot Chronos-2 forecasts on a 1-minute grid (745 MB) |
| 4.5 | `news_features.py`, `sibling_features.py`, `build_wallet_skill.py` | `derived/news_features`, `derived/sibling_features`, wallet track records |

The 3.2 TB of `data/processed/2026-03-06/chunk_*.pt` (18–19 Aug 2026) are read only by
`src/polymarket_gnn/models/train_fast.py`; they are probably obsolete and have been left untouched.

## 5. Models and studies (server, one run per GPU)

* Control and trees: `scripts/data_audit/feature_audit.py` (61 features at the validation samples, LightGBM G5),
  `scripts/data_audit/control_proba.py`, `scripts/data_audit/datamining.py` (PCA, clustering, separability).
* Deep model: `train_full.py` (CT-DGNN; `--hand-dir`, `--tree-dir`, `--tsfm-dir`, `--news-dir`, `--tree-offset`),
  driven by the `run_*.sh` scripts at the repository root; checkpoints `model_<run>_day_<dd>.pt` (1.4 MB each).
* Baselines and evaluation: `baseline_evaluation.py`, `src/polymarket_gnn/evaluation/`, the `*_report.py`
  scripts; outputs under `reports/`.
* Graph study (descriptive, all 21 days): `scripts/data_audit/graph_*.py`.

The experiments use 6–13 March (13 March is the validation day); the test set 24–26 March is locked (not evaluated).

## 6. Known absolute paths

`src/polymarket_gnn/data/{dataset_utils,lob_builder,lob_window_buffer,recover_hour,build_id_mappings}.py` and
`scripts/data_audit/*.py` still name `/data/v.pampinella/...` (the server). `modified_cli_scraper/preprocessing.py`
names the author's old laptop folder. The market scripts and `duckdb_client.py` resolve paths from the repository
root since 29 October 2026.

## 7. Where the data is

* Laptop: `data/preprocessing.duckdb` (15 GB, the only copy), `data/universe_markets.parquet`.
* Server `/data/v.pampinella/agy_project/data`: `derived/` (partitioned 188 GB, tensors 27 GB, hand features
  7.7 GB, tree log-probabilities 6.6 GB, tsfm 0.7 GB, multi_day 13 GB), `external/gdelt` 13 GB, checkpoints,
  `reports/`. No copy of `preprocessing.duckdb`.
* Nowhere any more: the Kaggle archive files, the daily `transactions_*.parquet`, the Gamma/CLOB JSONL pages
  (the first can be downloaded again from Kaggle; the fills can be re-scraped from Polygon; the API pages cannot be
  reproduced as of June–July 2026 and survive only inside the database).
