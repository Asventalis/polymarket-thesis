"""
Window-based raw data extraction for the Polymarket GNN pipeline.

Given a time window (start, end), this module extracts:
  1. All markets whose ACTIVE PERIOD intersects the window
     - Market active period: [accepting_orders_timestamp, market_end)
       where market_end = game_start_time (sports) or closed_time (others)
       or NULL (still open / unknown close time)
  2. All orderbook events in [start, end]
  3. All transactions in [start, end]
  4. All snapshots in [start, end]

The result is a WindowData dataclass holding all raw DataFrames,
ready to be consumed by graph / embedding builders downstream.

Design notes:
  - Markets with NULL closed_time and NULL game_start_time are included if
    their accepting_orders_timestamp <= end  (they may still be open).
  - Markets with NULL accepting_orders_timestamp are excluded (we cannot
    know when they started accepting orders).
  - All timestamps are returned timezone-aware (UTC).
  - Queries are pushed down to DuckDB — nothing is loaded into Python until
    needed, keeping memory usage proportional to the window size.

Usage:
    from polymarket_gnn.data.window_extractor import extract_window, WindowData

    data = extract_window(
        db_path   = "data/preprocessing.duckdb",
        start     = "2026-03-06 08:00:00",
        end       = "2026-03-06 10:00:00",
    )

    print(f"Markets: {len(data.markets)}")
    print(f"Transactions: {len(data.transactions)}")
    print(f"Orderbook events: {len(data.orderbook)}")
    print(f"Snapshots: {len(data.snapshots)}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


# Hourly-partitioned parquet directories (Hive-style: year=/month=/day=/hour=/).
# Built once by:  uv run python -m polymarket_gnn.data.build_hourly_partitions
# Reading a 2-hour window = reading 2 hourly files ≈ <100ms total.
_PART_TX_DIR   = Path("data") / "derived" / "partitioned" / "transactions"
_PART_OB_DIR   = Path("data") / "derived" / "partitioned" / "orderbook"
_PART_SNAP_DIR = Path("data") / "derived" / "partitioned" / "snapshots"

# Fallback: day-level aggregated parquets (built by preprocessing pipeline).
# Used when hourly partitions haven't been built yet.
# NOTE: transactions day-parquets are aggregated (no wallet-level data) →
#       only orderbook day-parquets are used as a fallback for OB features.
_DAY_OB_DIR  = Path("data") / "derived" / "multi_day" / "orderbook"


# ─────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────

@dataclass
class WindowData:
    """
    All raw data for a given time window, ready for graph / embedding construction.

    Attributes
    ----------
    window_start, window_end : pd.Timestamp (UTC)
        The requested time window boundaries.

    markets : pd.DataFrame
        One row per market whose active period intersects [window_start, window_end].
        Columns include all universe.markets fields plus a derived `market_end`
        column (the authoritative end-of-trading timestamp).

    transactions : pd.DataFrame
        RAW individual trades from universe.transactions in [window_start, window_end].
        Contains wallet-level data needed for graph edge construction.
        Columns: block_number, timestamp, tx_hash, order_index, wallet, role,
                 token_id, side, tokens, price, total_usdc, fee_rate, market_id

    orderbook : pd.DataFrame
        Pre-aggregated 2-second grid orderbook features from parquet files.
        Columns: grid_time (= timestamp_received alias), market_id,
                 mid_price, spread, best_bid, best_ask
        NOTE: raw tick-level orderbook events are NOT extracted (5.5B rows, no index).
              If raw ticks are needed, build a per-day parquet from universe.orderbook first.

    orderbook_agg : pd.DataFrame
        Alias for orderbook (kept for backward compatibility).

    snapshots : pd.DataFrame
        Raw snapshot events from universe.snapshots in [window_start, window_end].
        Columns: timestamp_received, timestamp_created_at, market_id,
                 update_type, data
    """

    window_start: pd.Timestamp
    window_end:   pd.Timestamp

    markets:      pd.DataFrame = field(default_factory=pd.DataFrame)
    transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    orderbook:    pd.DataFrame = field(default_factory=pd.DataFrame)
    orderbook_agg: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshots:    pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Human-readable summary of the extracted window."""
        dur = self.window_end - self.window_start
        lines = [
            f"WindowData [{self.window_start.isoformat()} → {self.window_end.isoformat()}]  "
            f"(duration: {dur})",
            f"  Markets       : {len(self.markets):>8,}",
            f"  Transactions  : {len(self.transactions):>8,}",
            f"  Orderbook evts: {len(self.orderbook):>8,}",
            f"  Snapshots     : {len(self.snapshots):>8,}",
        ]
        if len(self.markets) > 0:
            sports = self.markets['is_sports_market'].sum()
            open_end = self.markets['market_end'].isna().sum()
            lines += [
                f"  ↳ Sports markets          : {int(sports):>6,}",
                f"  ↳ Markets w/ unknown end  : {int(open_end):>6,}",
            ]
        if len(self.transactions) > 0:
            lines += [
                f"  ↳ Unique wallets          : {self.transactions['wallet'].nunique():>6,}",
                f"  ↳ Total USDC volume       : {self.transactions['total_usdc'].sum():>12,.2f}",
            ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────

def extract_window(
    db_path: str | Path,
    start:   str | pd.Timestamp,
    end:     str | pd.Timestamp,
    verbose: bool = True,
) -> WindowData:
    """
    Extract all raw Polymarket data whose timestamps fall within [start, end].

    Market selection logic
    ----------------------
    A market is included if its active trading period intersects [start, end].
    The active period is [accepting_orders_timestamp, market_end), where:

      - For SPORTS markets (game_start_time IS NOT NULL):
            market_end = game_start_time
        Rationale: once the game starts, the market effectively stops
        accepting meaningful new orders.

      - For NON-SPORTS markets:
            market_end = closed_time  (may be NULL for still-open markets)

    Intersection condition:
        market_start <= end   AND   (market_end IS NULL OR market_end >= start)

    Markets with NULL accepting_orders_timestamp are excluded — we cannot
    know when they became tradeable.

    Parameters
    ----------
    db_path : str or Path
        Path to the DuckDB database file.
    start, end : str or pd.Timestamp
        Window boundaries. Strings are parsed as UTC timestamps.
    verbose : bool
        If True, print extraction progress and summary.

    Returns
    -------
    WindowData
        Dataclass holding all extracted DataFrames.
    """
    db_path = str(Path(db_path).resolve())
    start   = pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end     = pd.Timestamp(end,   tz="UTC") if not isinstance(end,   pd.Timestamp) else end.tz_localize("UTC")   if end.tzinfo is None   else end.tz_convert("UTC")

    if start >= end:
        raise ValueError(f"start ({start}) must be strictly before end ({end})")

    start_sql = start.strftime("%Y-%m-%d %H:%M:%S")
    end_sql   = end.strftime("%Y-%m-%d %H:%M:%S")

    t_total = time.time()

    db = duckdb.connect(db_path, read_only=True)
    # Also open a separate in-memory DuckDB connection for parquet queries
    # (avoids locking issues with the main DB and is much faster for OB).
    pdb = duckdb.connect()   # in-memory, for parquet reads

    # Resolve project root from DB path (db lives at <root>/data/preprocessing.duckdb)
    project_root  = Path(db_path).resolve().parents[1]  # …/agy_project
    part_tx_dir   = project_root / _PART_TX_DIR
    part_ob_dir   = project_root / _PART_OB_DIR
    part_snap_dir = project_root / _PART_SNAP_DIR
    day_ob_dir    = project_root / _DAY_OB_DIR

    try:
        # ── 1. Markets ────────────────────────────────────────────
        # Derive market_end:
        #   - sports markets  → game_start_time
        #   - other markets   → closed_time  (may be NULL)
        # Include a market if:
        #   accepting_orders_timestamp IS NOT NULL
        #   AND accepting_orders_timestamp <= window_end    (started before window ends)
        #   AND (market_end IS NULL OR market_end >= window_start)  (hadn't ended before window starts)

        if verbose:
            print(f"[1/4] Extracting markets active in [{start_sql} → {end_sql}] …")

        t0 = time.time()
        markets_df = db.execute(f"""
            SELECT
                *,
                CASE
                    WHEN game_start_time IS NOT NULL THEN game_start_time
                    ELSE closed_time
                END AS market_end,
                (game_start_time IS NOT NULL) AS is_sports_market
            FROM universe.markets
            WHERE
                -- Must have a known start of trading
                accepting_orders_timestamp IS NOT NULL
                -- Market must have started before the window ends
                AND accepting_orders_timestamp <= TIMESTAMP '{end_sql}'
                -- Market must not have ended before the window starts
                -- (NULL market_end means still open → always included)
                AND (
                    CASE
                        WHEN game_start_time IS NOT NULL THEN game_start_time
                        ELSE closed_time
                    END IS NULL
                    OR
                    CASE
                        WHEN game_start_time IS NOT NULL THEN game_start_time
                        ELSE closed_time
                    END >= TIMESTAMP '{start_sql}'
                )
            ORDER BY accepting_orders_timestamp ASC
        """).df()
        markets_df['accepting_orders_timestamp'] = pd.to_datetime(markets_df['accepting_orders_timestamp'], utc=True)
        markets_df['closed_time']    = pd.to_datetime(markets_df['closed_time'],    utc=True, errors='coerce')
        markets_df['game_start_time'] = pd.to_datetime(markets_df['game_start_time'], utc=True, errors='coerce')
        markets_df['market_end']     = pd.to_datetime(markets_df['market_end'],     utc=True, errors='coerce')

        if verbose:
            print(f"   → {len(markets_df):,} markets  ({time.time()-t0:.1f}s)")

        if markets_df.empty:
            if verbose:
                print("   ⚠  No markets found in this window. Returning empty WindowData.")
            return WindowData(window_start=start, window_end=end, markets=markets_df)

        # Register active markets as a virtual table in DuckDB for high-speed semi-joins
        active_markets_df = markets_df[['market_id']].copy()
        db.register("active_markets", active_markets_df)
        pdb.register("active_markets", active_markets_df)

        # ── Helpers ───────────────────────────────────────────────
        def _hourly_files(base_dir: Path) -> list[Path]:
            """Collect hourly Hive-partition files covering [start, end]."""
            files = []
            h = start.floor("h")
            while h <= end:
                part = (base_dir /
                        f"year={h.year}" /
                        f"month={h.month}" /
                        f"day={h.day}" /
                        f"hour={h.hour}")
                files.extend(sorted(part.glob("*.parquet")))
                h += pd.Timedelta(hours=1)
            return files

        def _day_files(base_dir: Path) -> list[Path]:
            """Collect day-level parquet files covering [start, end]."""
            files = []
            if not base_dir.exists():
                return files
            for day in pd.date_range(start.date(), end.date(), freq="D"):
                p = base_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
                if p.exists():
                    files.append(p)
            return files

        def _read_parquet_expr(files: list[Path]) -> Optional[str]:
            """Return a DuckDB read_parquet([...]) expression, or None if empty."""
            if not files:
                return None
            return "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"

        # ── 2. Transactions (raw, wallet-level) ───────────────────
        # Priority: hourly partitions → raw DB (both fast, <0.05s per 2h window)
        if verbose:
            print(f"[2/4] Extracting transactions (raw wallet-level events) …")
        t0 = time.time()
        tx_hourly = _hourly_files(part_tx_dir)
        if tx_hourly:
            tx_expr = _read_parquet_expr(tx_hourly)
            tx_select = "* EXCLUDE (year, month, day, hour)"
        else:
            tx_expr = None
            tx_select = "*"
        if tx_hourly:
            transactions_df = pdb.execute(f"""
                SELECT {tx_select}
                FROM {tx_expr}
                WHERE
                    timestamp >= TIMESTAMP '{start_sql}'
                    AND timestamp <= TIMESTAMP '{end_sql}'
                    AND market_id IN (SELECT market_id FROM active_markets)
                ORDER BY timestamp ASC, block_number ASC, order_index ASC
            """).df()
        else:
            # Fallback: raw DB (0.02s per 2h window)
            transactions_df = db.execute(f"""
                SELECT *
                FROM universe.transactions
                WHERE
                    timestamp >= TIMESTAMP '{start_sql}'
                    AND timestamp <= TIMESTAMP '{end_sql}'
                    AND market_id IN (SELECT market_id FROM active_markets)
                ORDER BY timestamp ASC, block_number ASC, order_index ASC
            """).df()
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'], utc=True)
        if verbose:
            src = "hourly parquet" if tx_hourly else "raw DB"
            print(f"   → {len(transactions_df):,} transactions  ({time.time()-t0:.1f}s)  [{src}]")

        # ── 3. Orderbook (2s-grid / raw events) ───────────────────
        # Priority: hourly partitions → day-level parquets
        if verbose:
            print(f"[3/4] Extracting orderbook features …")
        t0 = time.time()
        ob_hourly = _hourly_files(part_ob_dir)
        ob_daily  = _day_files(day_ob_dir)
        # Prefer hourly; fall back to day-level
        if ob_hourly:
            ob_expr = _read_parquet_expr(ob_hourly)
            ob_select = "* EXCLUDE (year, month, day, hour)"  # drop Hive partition cols
            ts_col = "timestamp_received"
        elif ob_daily:
            ob_expr = _read_parquet_expr(ob_daily)
            ob_select = "*"                                    # no partition cols to drop
            ts_col = "grid_time"
        else:
            ob_expr = None
            ob_select = "*"
            ts_col = "timestamp_received"
        if ob_expr:
            arrow_table = pdb.execute(f"""
                SELECT {ob_select}
                FROM {ob_expr}
                WHERE
                    {ts_col} >= TIMESTAMP '{start_sql}'
                    AND {ts_col} <= TIMESTAMP '{end_sql}'
                    AND market_id IN (SELECT market_id FROM active_markets)
            """).to_arrow_table()
            orderbook_df = arrow_table.to_pandas(categories=['market_id'])
            if 'grid_time' in orderbook_df.columns:
                orderbook_df = orderbook_df.rename(columns={'grid_time': 'timestamp_received'})
        else:
            if verbose:
                print("   ⚠  No orderbook parquets found. Run build_hourly_partitions.py first.")
            orderbook_df = pd.DataFrame()
        if not orderbook_df.empty:
            orderbook_df['timestamp_received'] = pd.to_datetime(
                orderbook_df['timestamp_received'], utc=True, errors='coerce'
            )
        if verbose:
            src = "hourly" if ob_hourly else ("day-level" if ob_daily else "none")
            print(f"   → {len(orderbook_df):,} orderbook events/steps  ({time.time()-t0:.1f}s)  [{src}]")

        # ── 4. Snapshots ──────────────────────────────────────────
        if verbose:
            print(f"[4/4] Extracting snapshots …")
        t0 = time.time()
        snap_hourly = _hourly_files(part_snap_dir)
        if snap_hourly:
            snap_expr = _read_parquet_expr(snap_hourly)
            snap_select = "* EXCLUDE (year, month, day, hour)"
            snapshots_df = pdb.execute(f"""
                SELECT {snap_select}
                FROM {snap_expr}
                WHERE
                    timestamp_received >= TIMESTAMP '{start_sql}'
                    AND timestamp_received <= TIMESTAMP '{end_sql}'
                    AND market_id IN (SELECT market_id FROM active_markets)
                ORDER BY timestamp_received ASC
            """).df()
        else:
            snapshots_df = db.execute(f"""
                SELECT *
                FROM universe.snapshots
                WHERE
                    timestamp_received >= TIMESTAMP '{start_sql}'
                    AND timestamp_received <= TIMESTAMP '{end_sql}'
                    AND market_id IN (SELECT market_id FROM active_markets)
                ORDER BY timestamp_received ASC
            """).df()
        snapshots_df['timestamp_received'] = pd.to_datetime(
            snapshots_df['timestamp_received'], utc=True, errors='coerce'
        )
        if 'timestamp_created_at' in snapshots_df.columns:
            snapshots_df['timestamp_created_at'] = pd.to_datetime(
                snapshots_df['timestamp_created_at'], utc=True, errors='coerce'
            )
        if verbose:
            src = "hourly parquet" if snap_hourly else "raw DB"
            print(f"   → {len(snapshots_df):,} snapshots  ({time.time()-t0:.1f}s)  [{src}]")

    finally:
        db.close()
        pdb.close()

    result = WindowData(
        window_start   = start,
        window_end     = end,
        markets        = markets_df,
        transactions   = transactions_df,
        orderbook      = orderbook_df,
        orderbook_agg  = orderbook_df,   # alias — same object
        snapshots      = snapshots_df,
    )

    if verbose:
        print(f"\nDone in {time.time()-t_total:.1f}s\n")
        print(result.summary())

    return result


# ─────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────

def find_db(project_root: Optional[str | Path] = None) -> Path:
    """
    Locate the preprocessing.duckdb file relative to the project root.
    If project_root is None, walks up from this file's location.
    """
    if project_root is not None:
        p = Path(project_root) / "data" / "preprocessing.duckdb"
        if p.exists():
            return p
        raise FileNotFoundError(f"Database not found at {p}")

    # Walk up from this file
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "preprocessing.duckdb"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate preprocessing.duckdb — pass project_root explicitly.")


def load_window(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    verbose: bool = False,
    db_path: Optional[str | Path] = None,
) -> WindowData:
    """
    Convenience function to extract all intersecting markets and all associated events
    from the hourly partitioned parquet datasets simply by passing the start and end dates.

    Automatically resolves the database location and extracts:
      1. All intersecting active markets
      2. All wallet transactions within [start, end] (from hourly parquets)
      3. All orderbook tick updates within [start, end] (from hourly parquets)
      4. All price snapshots within [start, end] (from hourly parquets)

    Example usage:
        >>> from polymarket_gnn.data import load_window
        >>> data = load_window("2026-03-08 10:00:00", "2026-03-08 12:00:00", verbose=True)
    """
    if db_path is None:
        db_path = find_db()
    return extract_window(db_path, start=start, end=end, verbose=verbose)

