#!/usr/bin/env python3
"""Rebuild the market tables of data/preprocessing.duckdb from the raw inputs.

Runs the files of sql/markets/ in order, substituting the $PLACEHOLDERS with the paths
given on the command line. Two steps need an API scraper between them:

    00  raw views                 (Kaggle parquet + scraped transaction parquet)
    01  closed_markets            (Gamma JSONL pages)
        -> 00_legacy_raw_downloads/fetch_tags.py                    (closed_market_tags)
    02  markets_own_cleaned
    03  markets_orderbook, markets_snapshots, markets_kaggle, token_ids
    04  transactions_with_market, unmatched_token_ids
        -> 00_legacy_raw_downloads/fetch_polymarket_markets_by_token.py
    05  fill market_id from the token fetch, markets_transactions
    06  markets_universe, markets_missing
        -> 00_legacy_raw_downloads/scrape_missing_markets_by_condition_id.py
    07  missing_fetched_markets_cleaned, markets_cleaned_full, clob_only_market_coverage
    08  schema universe (views)
    09  first/last event timestamps      (needs data/derived/partitioned, i.e. after
                                           02_build_hourly_partitions.py)

Examples
    uv run python scripts/preprocessing/00_build_markets_tables.py --steps 00,01
    uv run python scripts/preprocessing/00_build_markets_tables.py --steps 02,03,04
    uv run python scripts/preprocessing/00_build_markets_tables.py --check

Expected row counts (the database of 4 Aug 2026) are checked with --check.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql" / "markets"

STEPS = {
    "00": "00_raw_views.sql",
    "01": "01_load_closed_markets.sql",
    "02": "02_markets_own_cleaned.sql",
    "03": "03_markets_from_raw_streams.sql",
    "04": "04_transactions_with_market.sql",
    "05": "05_fill_market_id_from_token_fetch.sql",
    "06": "06_markets_universe.sql",
    "07": "07_markets_cleaned_full.sql",
    "08": "08_universe_schema.sql",
    "09": "09_market_event_bounds.sql",
}

# Tables a step needs that are produced by a scraper, not by SQL.
REQUIRED_BEFORE = {
    "02": ("closed_market_tags", "00_legacy_raw_downloads/fetch_tags.py"),
    "05": ("markets_fetched_from_token_transactions",
           "00_legacy_raw_downloads/fetch_polymarket_markets_by_token.py"),
    "07": ("missing_fetched_markets",
           "00_legacy_raw_downloads/scrape_missing_markets_by_condition_id.py"),
}

# Row counts of data/preprocessing.duckdb as of 4 Aug 2026. token_ids is left out: in that
# database it comes from an April 2026 scrape (see sql/markets/03_markets_from_raw_streams.sql).
EXPECTED_COUNTS = {
    "closed_markets": 115_859,
    "closed_market_tags": 655_762,
    "markets_own_cleaned": 115_859,
    "markets_orderbook": 158_538,
    "markets_snapshots": 158_248,
    "markets_kaggle": 123_895,
    "transactions_with_market": 218_882_492,
    "markets_fetched_from_token_transactions": 22_858,
    "markets_transactions": 157_776,
    "markets_universe": 175_476,
    "markets_missing": 59_617,
    "missing_fetched_markets": 59_433,
    "missing_fetched_clob_markets": 184,
    "missing_fetched_markets_status": 59_617,
    "missing_fetched_markets_cleaned": 59_433,
    "markets_cleaned_full": 175_292,
    "clob_only_market_coverage": 184,
}

PLACEHOLDER = re.compile(r"\$([A-Z][A-Z0-9_]*)")


def render(sql: str, params: dict[str, str]) -> str:
    def sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in params or params[key] is None:
            raise KeyError(f"placeholder ${key} has no value; pass the matching option")
        return str(params[key]).replace("'", "''")
    return PLACEHOLDER.sub(sub, sql)


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = 'main' AND table_name = ?",
        [name],
    ).fetchone()
    return bool(row and row[0])


def run_step(con: duckdb.DuckDBPyConnection, step: str, params: dict, dry_run: bool) -> None:
    path = SQL_DIR / STEPS[step]
    sql = render(path.read_text(encoding="utf-8"), params)
    if step in REQUIRED_BEFORE and not dry_run:
        table, script = REQUIRED_BEFORE[step]
        if not table_exists(con, table):
            sys.exit(f"step {step} needs table {table}: run scripts/preprocessing/{script} first")
    print(f"== step {step}: {path.name}")
    if dry_run:
        print(sql)
        return
    t0 = time.time()
    con.execute(sql)
    print(f"   done in {time.time() - t0:.1f} s")


def check(con: duckdb.DuckDBPyConnection) -> int:
    bad = 0
    for table, expected in EXPECTED_COUNTS.items():
        if not table_exists(con, table):
            print(f"  {table:42s} missing        (expected {expected:>11,})")
            bad += 1
            continue
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        flag = "ok " if n == expected else "DIFF"
        bad += n != expected
        print(f"  {table:42s} {n:>11,}  {flag}  (expected {expected:>11,})")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(PROJECT_ROOT / "data" / "preprocessing.duckdb"))
    p.add_argument("--steps", default="", help="comma-separated step numbers, e.g. 02,03; empty = none")
    p.add_argument("--check", action="store_true", help="compare row counts with the 4 Aug 2026 database")
    p.add_argument("--dry-run", action="store_true", help="print the rendered SQL instead of running it")
    p.add_argument("--orderbook-glob", default=None, help="Kaggle archive/orderbook/*.parquet (step 00)")
    p.add_argument("--snapshots-glob", default=None, help="Kaggle archive/snapshots/*.parquet (step 00)")
    p.add_argument("--transactions-glob", default=None, help="scraped transactions_*.parquet (step 00)")
    p.add_argument("--closed-markets-glob", default=None, help="Gamma pages closed_markets_*.jsonl (step 01)")
    p.add_argument("--kaggle-labels", default=None, help="Kaggle labels/market_targets.parquet (step 03)")
    p.add_argument("--partitioned-dir", default=str(PROJECT_ROOT / "data" / "derived" / "partitioned"),
                   help="data/derived/partitioned (step 09)")
    args = p.parse_args()

    params = {
        "ORDERBOOK_GLOB": args.orderbook_glob,
        "SNAPSHOTS_GLOB": args.snapshots_glob,
        "TRANSACTIONS_GLOB": args.transactions_glob,
        "CLOSED_MARKETS_GLOB": args.closed_markets_glob,
        "KAGGLE_LABELS_PARQUET": args.kaggle_labels,
        "PARTITIONED_DIR": args.partitioned_dir,
    }

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        if s not in STEPS:
            sys.exit(f"unknown step {s}; known: {', '.join(STEPS)}")

    con = duckdb.connect(args.db_path, read_only=(not steps))
    print(f"database: {args.db_path}")
    for s in steps:
        run_step(con, s, params, args.dry_run)
    if args.check:
        print("== row counts")
        bad = check(con)
        print(f"   {bad} table(s) differ from the 4 Aug 2026 database")
    con.close()


if __name__ == "__main__":
    main()
