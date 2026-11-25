"""
One-time script: partition Polymarket data into hourly Parquet files.

Produces:
    data/derived/partitioned/
        transactions/
            year=2026/month=3/day=6/hour=8/
                data_0.parquet   ← one file per partition
        orderbook/
            year=2026/month=3/day=6/hour=8/
                data_0.parquet

Source for transactions : universe.transactions (raw, 218M rows)
                          DuckDB's COPY TO handles the full scan once.
Source for orderbook    : data/derived/multi_day/orderbook/*.parquet
                          (already pre-aggregated 2s-grid files, split in memory)

Why hourly?
  - The window extractor needs only O(window_hours) file reads.
  - DuckDB can read a 1-hour parquet in <100ms regardless of the full dataset size.
  - Files are small enough to fit in RAM individually (~10-50 MB each).

Usage:
    uv run python -m polymarket_gnn.data.build_hourly_partitions
    uv run python -m polymarket_gnn.data.build_hourly_partitions --source transactions
    uv run python -m polymarket_gnn.data.build_hourly_partitions --source orderbook

DuckDB PARTITION_BY note:
    DuckDB writes Hive-style directories:
        year=2026/month=3/day=6/hour=8/data_0.parquet
    Reading back is as simple as:
        read_parquet('partitioned/transactions/**/*.parquet', hive_partitioning=true)
    Or for a specific hour:
        read_parquet('partitioned/transactions/year=2026/month=3/day=6/hour=8/*.parquet')
"""

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "preprocessing.duckdb").exists():
            return parent
    raise FileNotFoundError("Could not find project root (preprocessing.duckdb not found)")


def partition_transactions(
    db_path: str,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Partition universe.transactions into hourly Hive parquets using DuckDB's
    native COPY TO ... PARTITION_BY.

    DuckDB does the full scan once and fans out into directories automatically.
    218M rows @ ~10 MB/s ≈ 3-5 minutes on a local SSD.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(out_dir.glob("year=*/**/*.parquet"))
    if existing and not overwrite:
        print(f"  [transactions] {len(existing)} partition files already exist. "
              f"Pass --overwrite to redo. Skipping.")
        return

    print(f"  [transactions] Partitioning from universe.transactions → {out_dir}")
    print(f"  This is a one-time operation (~3-5 min for 218M rows)...")

    t0 = time.time()
    db = duckdb.connect(db_path, read_only=True)
    try:
        db.execute(f"""
            COPY (
                SELECT
                    *,
                    EXTRACT(year  FROM timestamp)::INTEGER AS year,
                    EXTRACT(month FROM timestamp)::INTEGER AS month,
                    EXTRACT(day   FROM timestamp)::INTEGER AS day,
                    EXTRACT(hour  FROM timestamp)::INTEGER AS hour
                FROM universe.transactions
            )
            TO '{out_dir}'
            (FORMAT PARQUET, PARTITION_BY (year, month, day, hour), OVERWRITE_OR_IGNORE TRUE)
        """)
    finally:
        db.close()

    n_files = len(list(out_dir.glob("year=*/**/*.parquet")))
    print(f"  [transactions] Done: {n_files} partition files in {time.time()-t0:.0f}s")


def partition_orderbook(
    db_path: str,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Partition universe.orderbook into hourly Hive parquets using DuckDB's
    native COPY TO ... PARTITION_BY directly from the master DuckDB file.
    Runs day-by-day (March 6 to March 26) to maximize IO throughput and enable instant checkpointing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp("2026-03-06")
    end_date   = pd.Timestamp("2026-03-26")

    print(f"  [orderbook] Partitioning all dates ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) → {out_dir}")
    print(f"  Running day-by-day streaming scan (~55 seconds per day, ~20 min total)...")
    t0_total = time.time()

    db = duckdb.connect(db_path, read_only=True)
    try:
        cur_date = start_date
        while cur_date <= end_date:
            year, month, day = cur_date.year, cur_date.month, cur_date.day
            day_dir = out_dir / f"year={year}" / f"month={month}" / f"day={day}"
            
            # Check if this day is already completely partitioned (at least 20 hours present)
            if day_dir.exists() and not overwrite:
                existing_hours = list(day_dir.glob("hour=*/*.parquet"))
                if len(existing_hours) >= 20:
                    print(f"    {cur_date.strftime('%Y-%m-%d')}: already done ({len(existing_hours)} hours found). Skipping.")
                    cur_date += pd.Timedelta(days=1)
                    continue

            t0 = time.time()
            next_date = cur_date + pd.Timedelta(days=1)
            t_start_sql = cur_date.strftime("%Y-%m-%d 00:00:00")
            t_end_sql   = next_date.strftime("%Y-%m-%d 00:00:00")
            print(f"    Processing {cur_date.strftime('%Y-%m-%d')}...", end=" ", flush=True)

            db.execute(f"""
                COPY (
                    SELECT
                        *,
                        {year}  AS year,
                        {month} AS month,
                        {day}   AS day,
                        EXTRACT(hour FROM timestamp_received)::INTEGER AS hour
                    FROM universe.orderbook
                    WHERE timestamp_received >= TIMESTAMP '{t_start_sql}'
                      AND timestamp_received <  TIMESTAMP '{t_end_sql}'
                )
                TO '{out_dir}'
                (FORMAT PARQUET, PARTITION_BY (year, month, day, hour), OVERWRITE_OR_IGNORE TRUE)
            """)
            n_hours = len(list(day_dir.glob("hour=*/*.parquet"))) if day_dir.exists() else 0
            print(f"done in {time.time()-t0:.1f}s ({n_hours} hourly files)")
            cur_date += pd.Timedelta(days=1)
    finally:
        db.close()

    total_files = len(list(out_dir.glob("year=*/**/*.parquet")))
    print(f"  [orderbook] Done: {total_files} partition files across 21 days in {time.time()-t0_total:.0f}s")


def partition_snapshots(
    db_path: str,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Partition universe.snapshots into hourly Hive parquets directly from the master DB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("year=*/**/*.parquet"))
    if existing and not overwrite:
        print(f"  [snapshots] {len(existing)} partition files already exist. "
              f"Pass --overwrite to redo. Skipping.")
        return

    print(f"  [snapshots] Partitioning from universe.snapshots → {out_dir}")
    print(f"  This is a one-time operation (~15-30 seconds for 51M rows)...")
    t0 = time.time()

    db = duckdb.connect(db_path, read_only=True)
    try:
        db.execute(f"""
            COPY (
                SELECT
                    *,
                    EXTRACT(year  FROM timestamp_received)::INTEGER AS year,
                    EXTRACT(month FROM timestamp_received)::INTEGER AS month,
                    EXTRACT(day   FROM timestamp_received)::INTEGER AS day,
                    EXTRACT(hour  FROM timestamp_received)::INTEGER AS hour
                FROM universe.snapshots
            )
            TO '{out_dir}'
            (FORMAT PARQUET, PARTITION_BY (year, month, day, hour), OVERWRITE_OR_IGNORE TRUE)
        """)
    finally:
        db.close()

    n_files = len(list(out_dir.glob("year=*/**/*.parquet")))
    print(f"  [snapshots] Done: {n_files} partition files in {time.time()-t0:.0f}s")




# ─────────────────────────────────────────────────────────────────
# Convenience: read a single hourly partition back
# ─────────────────────────────────────────────────────────────────

def hourly_parquet_path(base_dir: Path, ts: pd.Timestamp) -> Path:
    """Return the Hive partition directory for a given timestamp."""
    return (base_dir /
            f"year={ts.year}" /
            f"month={ts.month}" /
            f"day={ts.day}" /
            f"hour={ts.hour}")


def hours_in_window(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return all hours (floor) that overlap with [start, end]."""
    h = start.floor("h")
    result = []
    while h <= end:
        result.append(h)
        h += pd.Timedelta(hours=1)
    return result


def collect_hourly_parquets(
    base_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    """
    Return all hourly parquet files that overlap with [start, end].
    Missing hours (no trades / no orderbook / no snapshots) are silently skipped.
    """
    files = []
    for h in hours_in_window(start, end):
        part_dir = hourly_parquet_path(base_dir, h)
        for f in sorted(part_dir.glob("*.parquet")):
            files.append(f)
    return files


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition Polymarket data into hourly Parquet files (one-time operation)."
    )
    parser.add_argument(
        "--source",
        choices=["transactions", "orderbook", "snapshots", "all"],
        default="all",
        help="Which data source to partition (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing partition files",
    )
    args = parser.parse_args()

    root      = find_project_root()
    db_path   = str(root / "data" / "preprocessing.duckdb")
    part_root = root / "data" / "derived" / "partitioned"

    print(f"Project root : {root}")
    print(f"Output root  : {part_root}")
    print()

    if args.source in ("transactions", "all"):
        partition_transactions(
            db_path   = db_path,
            out_dir   = part_root / "transactions",
            overwrite = args.overwrite,
        )
        print()

    if args.source in ("orderbook", "all"):
        partition_orderbook(
            db_path   = db_path,
            out_dir   = part_root / "orderbook",
            overwrite = args.overwrite,
        )
        print()

    if args.source in ("snapshots", "all"):
        partition_snapshots(
            db_path   = db_path,
            out_dir   = part_root / "snapshots",
            overwrite = args.overwrite,
        )
        print()

    print("All done. Update window_extractor.py to use:")
    print(f"  PARTITIONED_TX_DIR   = {part_root / 'transactions'}")
    print(f"  PARTITIONED_OB_DIR   = {part_root / 'orderbook'}")
    print(f"  PARTITIONED_SNAP_DIR = {part_root / 'snapshots'}")

