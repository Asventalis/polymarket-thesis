# Code of the thesis

This directory holds the code that produced every number, table and figure of the thesis
*Data Analysis and Data Mining on Decentralised Prediction Markets: the Price Process of
Polymarket from One Minute to Twelve Hours* (University of Pisa, 2026). It is a snapshot of
the working copy used for the final experiments, taken in November 2026.

**Only the code is here.** The data are not in the repository:

- the raw sources are public and are listed in `docs/REPRODUCE.md` (the Polymarket
  tick-level order-book archive on Kaggle, the Polygon blockchain, the Polymarket Gamma and
  CLOB APIs, the GDELT news files);
- the derived datasets (market tables, hourly transaction partitions, order-book tensors,
  engineered features, cross-fitted tree log-probabilities, foundation-model forecasts,
  about 55 GB), the trained checkpoints and the reports with the raw numbers are not
  distributed, because of their size and of the licence of the order-book archive.

The paths inside the scripts are those of the working copies (`data/raw/`, `data/derived/`
and the server directories under `/data/`), as described in appendix C of the thesis.

## Layout

| path | content |
|---|---|
| `src/polymarket_gnn/` | the Python package: data pipeline (DuckDB, NumPy), feature indices, models (PyTorch), training and evaluation |
| `tests/` | the test suite (label construction, feature indices, state engine, encoders, objectives) |
| `sql/` | the SQL steps that build the market tables and the first derived streams |
| `scripts/preprocessing/` | collection and preprocessing: the fork of `poly-trade-scan` that reads the fills from Polygon, the metadata scrapers, the market tables, the hourly partitions |
| `scripts/data_audit/` | the stateless control and its feature audit, the calibration offsets, the test report, the gap experiment, the graph maps, the news and text audits, the tabular network, the economic simulation |
| `analysis/scripts/` | the order-book animation |
| `train_full.py`, `eval_checkpoint.py` | training of the stateful models (one run per label) and their evaluation on the test days |
| `hand_features.py`, `tree_features.py`, `sibling_features.py`, `news_features.py`, `tsfm_zero_shot.py`, `tsfm_cache.py`, `build_wallet_skill.py` | the engineered features, the cross-fitted trees, the sibling, news, foundation-model and wallet-skill inputs |
| `run_*.sh`, `queue_*.sh`, `test_queue_*.sh`, `trees_test.sh`, `lag_controls.sh` | the drivers that launched the runs on the GPU server, kept as the record of every configuration |
| `*_report.py`, `compare_runs.py`, `quick_ba_h.py`, `lag_bootstrap.py` | the scripts that turn predictions into the tables of the thesis |
| `docs/REPRODUCE.md` | the data pipeline step by step, with row counts and sizes; `docs/model_architecture.md` describes the stateful model |

The run labels, the configurations and the steps that reproduce each number are in
appendices B and C of the thesis.

## Environment

The project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`,
Python version in `.python-version`): `uv sync` creates the virtual environment. The GPU
runs were executed on a shared server inside an Apptainer image through `run.sh` and
`run_dev.sh`; `Dockerfile`, `docker-compose.yml` and `install_rocm.sh` are the equivalent
local setups.
