"""
Recovery script for year=2026/month=3/day=10/hour=10
The snapshot parquet for that hour is corrupted. 

Strategy:
  1. Process hour=9 normally to rebuild the in-RAM OrderbookState for all markets
     (tensors already saved, so we just warm up the state and discard output).
  2. Process hour=10 using ONLY orderbook events (skip corrupted snapshot file).
     The book state is accurately initialized from hour=9 warm-up.
  3. Save tensors for hour=10.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import duckdb
import numpy as np
import pandas as pd

try:
    from polymarket_gnn.data.dataset_utils import load_and_interleave_hour
    from polymarket_gnn.data.lob_builder import OrderbookState
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_utils import load_and_interleave_hour
    from lob_builder import OrderbookState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE      = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
OUT_BASE  = Path("/data/v.pampinella/agy_project/data/derived/tensors")
K         = 20

WARMUP_HOUR = dict(year=2026, month=3, day=10, hour=9)
TARGET_HOUR = dict(year=2026, month=3, day=10, hour=10)


def hour_path(dataset: str, h: dict) -> Path:
    return BASE / dataset / f"year={h['year']}/month={h['month']}/day={h['day']}/hour={h['hour']}"


def run_extraction(df: pd.DataFrame, global_states: Dict[str, OrderbookState],
                   save: bool, out_file: Path = None) -> int:
    """
    Iterates a sorted interleaved DataFrame, updates global_states in place.
    If save=True, extracts tensors every K events and writes to out_file.
    Returns the number of tensors extracted.
    """
    extracted = []
    current_market = None

    for row in df.itertuples(index=False):
        m_id = row.market_id

        if m_id not in global_states:
            global_states[m_id] = OrderbookState(market_id=m_id)

        state = global_states[m_id]

        # Transition to new market: reset accumulators (but keep bids/asks!)
        if m_id != current_market:
            state.reset_accumulators()
            current_market = m_id

        state.apply_event(row)

        if save and state.events_processed % K == 0:
            extracted.append(state.extract_features(row.timestamp_received))

    if save and extracted:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(extracted).to_parquet(out_file, engine="pyarrow", index=False)
        logger.info(f"Saved {len(extracted)} tensors → {out_file}")

    return len(extracted)


def main():
    global_states: Dict[str, OrderbookState] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Warm up state from hour=9 (tensors already saved, skip saving)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 1: Warming up state from hour=9 ===")
    ob9   = hour_path("orderbook", WARMUP_HOUR)
    snap9 = hour_path("snapshots", WARMUP_HOUR)

    t0 = time.time()
    df9 = load_and_interleave_hour(ob9, snap9)
    logger.info(f"Loaded hour=9: {len(df9)} rows in {time.time()-t0:.2f}s")

    t0 = time.time()
    run_extraction(df9, global_states, save=False)
    logger.info(
        f"Warm-up done: {len(global_states)} markets in RAM, {time.time()-t0:.2f}s. "
        f"bids/asks intact."
    )
    del df9  # free memory

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Process hour=10 with ONLY orderbook events (snapshot is corrupted)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 2: Processing hour=10 (orderbook events only) ===")
    ob10 = hour_path("orderbook", TARGET_HOUR)

    t0 = time.time()
    # Load ONLY orderbook events, no snapshot merge (file is corrupted)
    ob10_df = duckdb.query(
        f"SELECT *, 'event' as type FROM read_parquet('{str(ob10)}/**/*.parquet')"
    ).df()
    ob10_df.sort_values(["market_id", "timestamp_received"], inplace=True)
    ob10_df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded hour=10 orderbook: {len(ob10_df)} rows in {time.time()-t0:.2f}s")

    out_file = OUT_BASE / f"year={TARGET_HOUR['year']}/month={TARGET_HOUR['month']}/day={TARGET_HOUR['day']}/hour={TARGET_HOUR['hour']}/lob_features.parquet"

    t0 = time.time()
    n = run_extraction(ob10_df, global_states, save=True, out_file=out_file)
    logger.info(f"Extracted {n} tensors from hour=10 in {time.time()-t0:.2f}s")
    logger.info("Recovery complete.")


if __name__ == "__main__":
    main()
