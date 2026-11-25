"""
LOBWindowBuffer: efficient in-memory structure for sliding window LOB queries.

Given all LOB tensors for a time period (loaded from Parquet), this class
provides fast O(log N) queries:
    get_window(market_id, t) → np.ndarray of shape (K, 52)

It also integrates market lifecycle events (OPEN/CLOSE) to maintain an
O(1)-lookup set of currently-open markets, avoiding scanning all 13k markets
every step. Active markets are found in O(|open_markets| × log N) instead of
O(|all_markets| × log N).

Usage:
    buffer = LOBWindowBuffer.from_parquet(tensor_path, lifecycle_path, K=5)
    buffer.reset_to_time(start_ns)           # fast-forward lifecycle to start time
    active_ids, windows = buffer.load_step(t_start_ns, t_end_ns)
"""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import time
import bisect
import logging
import torch
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

FEATURE_DIM = 52
OUTPUT_DIM = FEATURE_DIM + 3  # We append 3 momentum features on-the-fly
LIFECYCLE_PATH = Path("/data/v.pampinella/agy_project/data/derived/market_lifecycle_events.parquet")


class LOBWindowBuffer:
    """
    Holds all LOB snapshots for a set of markets, organized for fast window
    queries. Uses market lifecycle events to maintain an open-market set,
    so only open markets are checked at each step.

    Internal layout per market:
        _timestamps[market_id]:  sorted int64 numpy array (nanoseconds)
        _features[market_id]:    float32 array of shape (N, FEATURE_DIM)
    """

    def __init__(self, K: int = 5):
        self.K = K
        self._timestamps: Dict[str, np.ndarray] = {}
        self._features:   Dict[str, np.ndarray] = {}
        self._zero_row = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Lifecycle state
        self._lifecycle_df: Optional[pd.DataFrame] = None
        self._lifecycle_ptr: int = 0
        self._open_markets: set = set()
        self._open_with_data: set = set()
        
        # Pre-computed close times per market (built once from lifecycle_df)
        self._market_close_time_ns: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Builders
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        tensor_paths,
        lifecycle_path: Path = LIFECYCLE_PATH,
        K: int = 5,
    ) -> "LOBWindowBuffer":
        """
        Loads LOB tensors and market lifecycle events.

        Args:
            tensor_paths:   str glob, Path, or list of Paths to lob_features.parquet files
            lifecycle_path: path to market_lifecycle_events.parquet
            K:              window size (number of past snapshots per step)
        """
        buf = cls(K=K)
        buf.reload_lob_data(tensor_paths)

        # ── Load lifecycle events ────────────────────────────────────────────
        t0 = time.time()
        lc = pd.read_parquet(lifecycle_path)
        lc["event_timestamp"] = pd.to_datetime(lc["event_timestamp"])
        lc = lc.sort_values("event_timestamp").reset_index(drop=True)
        buf._lifecycle_df  = lc
        buf._lifecycle_ptr = 0
        buf._open_markets  = set()
        
        # Pre-compute the close time for each market (last CLOSE event)
        # Used by get_future_targets to filter short-lived markets
        close_events = lc[lc["event_type"] == "CLOSE"]
        buf._market_close_time_ns = {
            row["market_id"]: int(row["event_timestamp"].value)
            for _, row in close_events.iterrows()
        }
        logger.info(
            f"Lifecycle events loaded: {len(lc)} events "
            f"({lc['event_type'].value_counts().to_dict()}) "
            f"in {time.time() - t0:.3f}s"
        )

        return buf

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Management & Data Swapping
    # ─────────────────────────────────────────────────────────────────────────

    def reload_lob_data(self, tensor_paths):
        """
        Unloads current LOB data and loads a new hour's data,
        updating the _open_with_data intersection set without
        resetting the lifecycle pointer.
        """
        self._timestamps.clear()
        self._features.clear()
        
        t0 = time.time()
        if isinstance(tensor_paths, (str, Path)):
            tensor_paths = str(tensor_paths)
            df = duckdb.query(
                f"SELECT market_id, timestamp, features FROM read_parquet('{tensor_paths}')"
            ).df()
        else:
            paths_str = ", ".join(f"'{str(p)}'" for p in tensor_paths)
            df = duckdb.query(
                f"SELECT market_id, timestamp, features FROM read_parquet([{paths_str}])"
            ).df()

        for market_id, group in df.groupby("market_id", sort=False):
            group = group.sort_values("timestamp")
            self._timestamps[market_id] = group["timestamp"].astype("int64").values
            self._features[market_id]   = np.stack(group["features"].values).astype(np.float32)

        # Update intersection
        self._open_with_data = {m for m in self._open_markets if m in self._timestamps}

        logger.info(
            f"Reloaded LOB data: {len(self._timestamps)} markets in {time.time() - t0:.3f}s. "
            f"open_with_data updated to {len(self._open_with_data)} markets."
        )

    def reset_to_time(self, t_ns: int):
        """
        Fast-forwards the lifecycle pointer to time t_ns (int64 nanoseconds),
        building the correct open_markets set for that point in time.

        Call this once before starting the training loop for a given hour.
        """
        self._lifecycle_ptr = 0
        self._open_markets  = set()
        self._open_with_data = set()
        self._advance_lifecycle(t_ns)
        logger.info(
            f"Lifecycle reset to t={pd.Timestamp(t_ns)}: "
            f"{len(self._open_markets)} open markets total, "
            f"{len(self._open_with_data)} with LOB data (will be scanned)"
        )

    def _advance_lifecycle(self, t_end_ns: int):
        """
        Processes all lifecycle events with timestamp <= t_end_ns.
        Maintains both _open_markets (all open) and _open_with_data
        (open AND present in the LOB buffer) — we only scan the latter.
        """
        lc  = self._lifecycle_df
        ptr = self._lifecycle_ptr

        while ptr < len(lc):
            event_ts_ns = lc.iloc[ptr]["event_timestamp"].value  # int64 ns
            if event_ts_ns > t_end_ns:
                break
            ev      = lc.iloc[ptr]
            m_id    = ev["market_id"]
            has_lob = m_id in self._timestamps

            if ev["event_type"] == "OPEN":
                self._open_markets.add(m_id)
                if has_lob:
                    self._open_with_data.add(m_id)
            else:
                self._open_markets.discard(m_id)
                self._open_with_data.discard(m_id)
            ptr += 1

        self._lifecycle_ptr = ptr

    # ─────────────────────────────────────────────────────────────────────────
    # Core Query
    # ─────────────────────────────────────────────────────────────────────────

    def get_window(self, market_id: str, t_ns: int) -> np.ndarray:
        """
        Returns the last K LOB snapshots for market_id with timestamp < t_ns.
        Shape: (K, OUTPUT_DIM), padded with zero rows at the beginning if
        fewer than K snapshots exist.
        """
        if market_id not in self._timestamps:
            return np.zeros((self.K, OUTPUT_DIM), dtype=np.float32)

        ts  = self._timestamps[market_id]
        idx = bisect.bisect_left(ts, t_ns)  # first index with ts[idx] >= t_ns

        if idx == 0:
            return np.zeros((self.K, OUTPUT_DIM), dtype=np.float32)

        start = max(0, idx - self.K)
        rows  = self._features[market_id][start:idx]  # shape (<=K, 52)

        n = len(rows)
        if n < self.K:
            pad = np.zeros((self.K - n, FEATURE_DIM), dtype=np.float32)
            base_window = np.concatenate([pad, rows], axis=0) # shape (K, 52)
        else:
            base_window = rows
            
        # Add momentum features (price_change_1, price_change_k, sign)
        momentum = np.zeros((self.K, 3), dtype=np.float32)
        mid_prices = base_window[:, 0]
        
        for i in range(self.K):
            if i > 0:
                momentum[i, 0] = mid_prices[i] - mid_prices[i-1] # price_change_1
            momentum[i, 1] = mid_prices[i] - mid_prices[0]       # price_change_k
            momentum[i, 2] = np.sign(momentum[i, 1])             # momentum_sign
            
        return np.concatenate([base_window, momentum], axis=1) # shape (K, 55)

    def load_step(self, t_start_ns: int, t_end_ns: int) -> Tuple[List[str], np.ndarray]:
        """
        Main entry point for the training loop.

        1. Advances the lifecycle pointer → updates open_markets (O(k) events)
        2. Scans only open markets for new LOB snapshots (O(|open| × log N))
        3. Returns windows tensor (N_active, K, FEATURE_DIM) ready for GPU

        Returns:
            active_markets: List[str]  - market_ids with new data in [t_start, t_end)
            windows:        np.ndarray - shape (N_active, K, FEATURE_DIM), float32
        """
        # Advance lifecycle to t_end
        self._advance_lifecycle(t_end_ns)

        # Scan only markets that are OPEN AND have LOB data (intersection)
        # _open_with_data is much smaller than _open_markets (no dead weight)
        active = []
        for market_id in self._open_with_data:
            ts = self._timestamps[market_id]
            lo = bisect.bisect_left(ts, t_start_ns)
            hi = bisect.bisect_left(ts, t_end_ns)
            if hi > lo:
                active.append(market_id)

        if not active:
            return [], np.empty((0, self.K, OUTPUT_DIM), dtype=np.float32)

        windows = np.stack([self.get_window(m, t_end_ns) for m in active])
        return active, windows

    # ─────────────────────────────────────────────────────────────────────────
    # Target Generation (Step C)
    # ─────────────────────────────────────────────────────────────────────────

    def get_future_targets(
        self,
        active_ids: List[str],
        current_t_ns: int,
        horizon_ns: int,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Looks ahead in the loaded LOB buffer to generate causal targets
        for the given active markets.

        Target is 1.0 if the mid-price (feature index 0) goes UP at
        current_t_ns + horizon_ns, else 0.0.

        If a market has no data at or after the future time, it returns NaN.
        """
        targets = []
        target_t_ns = current_t_ns + horizon_ns

        for market_id in active_ids:
            ts = self._timestamps.get(market_id)
            if ts is None or len(ts) == 0:
                targets.append(float('nan'))
                continue

            # Current mid-price (last snapshot strictly before current_t_ns)
            idx_curr = bisect.bisect_left(ts, current_t_ns)
            if idx_curr == 0:
                # No data before current time
                targets.append(float('nan'))
                continue
            mid_curr = self._features[market_id][idx_curr - 1, 0]

            # Future mid-price: look in buffer which now also contains future-hour data
            idx_fut = bisect.bisect_left(ts, target_t_ns)
            if idx_fut == 0:
                targets.append(float('nan'))
                continue
            
            # Filter out markets that close before the prediction horizon.
            # A market that closes at t_close < target_t_ns doesn't have a
            # meaningful future price to predict toward.
            close_t = self._market_close_time_ns.get(market_id, None)
            if close_t is not None and close_t < target_t_ns:
                targets.append(float('nan'))
                continue

            mid_fut = self._features[market_id][idx_fut - 1, 0]

            # Target: 0.0 (DOWN), 1.0 (CONSTANT), 2.0 (UP)
            if mid_fut > mid_curr:
                targets.append(2.0)
            elif mid_fut < mid_curr:
                targets.append(0.0)
            else:
                targets.append(1.0)

        return torch.tensor(targets, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    TENSOR_PATH = "/data/v.pampinella/agy_project/data/derived/tensors/year=2026/month=3/day=23/hour=13/lob_features.parquet"
    TXN_PATH    = "/data/v.pampinella/agy_project/data/derived/partitioned/transactions/year=2026/month=3/day=23/hour=13"
    K = 5

    # Build buffer with lifecycle awareness
    logger.info("Loading LOB buffer + lifecycle events...")
    buf = LOBWindowBuffer.from_parquet(TENSOR_PATH, K=K)

    # Load transaction buckets
    txn_df = duckdb.query(
        f"SELECT timestamp FROM read_parquet('{TXN_PATH}/**/*.parquet') ORDER BY timestamp"
    ).df()
    txn_df["bucket"] = txn_df["timestamp"].dt.floor("2s")
    buckets = sorted(txn_df["bucket"].unique())
    logger.info(f"Transaction buckets: {len(buckets)}")

    # Fast-forward lifecycle to the start of the hour
    start_ns = buckets[0].value
    buf.reset_to_time(start_ns)

    # Run 100 steps and benchmark
    step_times = []
    for i, t in enumerate(buckets[:100]):
        t_ns      = t.value
        t_prev_ns = (t - pd.Timedelta("2s")).value

        t0 = time.time()
        active, windows = buf.load_step(t_prev_ns, t_ns)
        elapsed = time.time() - t0
        step_times.append(elapsed)

        if i < 3 or i % 20 == 0:
            logger.info(
                f"Step {i:3d} | t={t} | open_markets={len(buf._open_markets):4d} | "
                f"active={len(active):3d} | windows={windows.shape} | "
                f"step_time={elapsed*1000:.2f}ms"
            )

    logger.info(f"\n--- Summary (100 steps) ---")
    logger.info(f"Avg step time: {np.mean(step_times)*1000:.2f}ms")
    logger.info(f"Max step time: {np.max(step_times)*1000:.2f}ms")
