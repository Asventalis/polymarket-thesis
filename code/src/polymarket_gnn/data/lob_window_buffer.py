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

from polymarket_gnn.data.lob_layout import FEATURE_DIM, MID_IDX

logger = logging.getLogger(__name__)

# 52 stored columns + 3 momentum features appended on the fly in get_window().
# NOTE: the mid price is column MID_IDX (=40); column 0 is (best_bid - mid)/mid.
OUTPUT_DIM = FEATURE_DIM + 3
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
        buf._market_close_time_ns = dict(zip(
            close_events["market_id"].values,
            close_events["event_timestamp"].values.astype("datetime64[ns]").astype("int64").tolist(),
        ))
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

        # Momentum features (price_change_1, price_change_k, sign) computed on
        # the TRUE mid price (column MID_IDX) and only over real rows: padded
        # rows are all-zero and must not be read as "price was 0".
        momentum = np.zeros((self.K, 3), dtype=np.float32)
        first = self.K - n                       # index of the first real row
        mid_prices = base_window[:, MID_IDX]
        for i in range(first + 1, self.K):
            momentum[i, 0] = mid_prices[i] - mid_prices[i - 1]   # price_change_1
        for i in range(first, self.K):
            momentum[i, 1] = mid_prices[i] - mid_prices[first]   # price_change_k
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
        DEPRECATED — use ``polymarket_gnn.data.price_index.MidPriceIndex.labels``.

        Looks ahead in the *loaded* LOB buffer to generate 3-class targets
        (0=DOWN, 1=CONSTANT, 2=UP) on the mid price (column MID_IDX = 40).

        Limitations (fixed in MidPriceIndex):
          * only the hours currently loaded are visible, so if the hour
            containing t+H (or the hours in between) is not loaded the label
            silently degrades to CONSTANT;
          * no NaN is returned when t+H is beyond the available data.
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
            mid_curr = self._features[market_id][idx_curr - 1, MID_IDX]

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

            mid_fut = self._features[market_id][idx_fut - 1, MID_IDX]

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


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised implementation
# ─────────────────────────────────────────────────────────────────────────────

class FastLOBWindowBuffer:
    """
    Drop-in replacement for ``LOBWindowBuffer`` (same constructor, same
    ``reload_lob_data`` / ``reset_to_time`` / ``load_step`` contract, same
    values) without per-market Python loops.

    ``LOBWindowBuffer.load_step`` scans every open market with two bisects per
    block (~13k markets → ~30 ms) and builds each window with Python loops.
    Here all rows of the resident hours live in three flat arrays sorted by
    (market, time) plus a time-sorted event stream, so that a block costs two
    ``searchsorted`` calls and one fancy-indexing gather.

    Equivalence with the reference class is exact (bit-identical windows, same
    active set), including two details that are easy to get wrong:
      * rows of a market with IDENTICAL timestamps: the reference orders each
        market with ``DataFrame.sort_values("timestamp")`` (unstable sort), so
        tied rows are ordered by that very call here as well;
      * lifecycle events are replayed in the reference's own sorted order.
    The only difference is the ORDER of the returned markets: the reference
    iterates a Python set (order changes between processes), this class
    returns them in a deterministic order (first resident row of the market).

    ``tests/test_fast_lob_buffer.py`` and ``scripts/verify_fast_buffer.py``
    check the equivalence on synthetic and on real data.
    """

    def __init__(self, K: int = 5):
        self.K = K
        self._markets: List[str] = []                 # code → market_id
        self._code_of: Dict[str, int] = {}
        self._ts = np.empty(0, dtype=np.int64)        # rows sorted by (market code, time)
        self._feat = np.empty((0, FEATURE_DIM), dtype=np.float32)
        self._start = np.empty(0, dtype=np.int64)     # per code: first row
        self._stop = np.empty(0, dtype=np.int64)      # per code: one past last row
        self._ev_ts = np.empty(0, dtype=np.int64)     # event stream sorted by time
        self._ev_code = np.empty(0, dtype=np.int64)
        self._ev_row = np.empty(0, dtype=np.int64)
        self._open_mask = np.empty(0, dtype=bool)     # per code: market currently open

        self._lc_ts = np.empty(0, dtype=np.int64)
        self._lc_market = np.empty(0, dtype=object)
        self._lc_is_open = np.empty(0, dtype=bool)
        self._lifecycle_ptr = 0
        self._open_markets: set = set()
        self._market_close_time_ns: Dict[str, int] = {}

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(cls, tensor_paths, lifecycle_path: Path = LIFECYCLE_PATH, K: int = 5) -> "FastLOBWindowBuffer":
        buf = cls(K=K)
        buf.reload_lob_data(tensor_paths)
        t0 = time.time()
        # identical preparation to LOBWindowBuffer.from_parquet (same sort call → same event order)
        lc = pd.read_parquet(lifecycle_path)
        lc["event_timestamp"] = pd.to_datetime(lc["event_timestamp"])
        lc = lc.sort_values("event_timestamp").reset_index(drop=True)
        buf._lc_ts = lc["event_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        buf._lc_market = lc["market_id"].values
        buf._lc_is_open = (lc["event_type"].values == "OPEN")
        close_events = lc[lc["event_type"] == "CLOSE"]
        buf._market_close_time_ns = dict(zip(
            close_events["market_id"].values,
            close_events["event_timestamp"].values.astype("datetime64[ns]").astype("int64").tolist(),
        ))
        logger.info(f"Lifecycle events loaded: {len(lc)} events in {time.time() - t0:.3f}s")
        return buf

    def reload_lob_data(self, tensor_paths) -> None:
        t0 = time.time()
        if isinstance(tensor_paths, (str, Path)):
            src = f"'{str(tensor_paths)}'"
        else:
            src = "[" + ", ".join(f"'{str(p)}'" for p in tensor_paths) + "]"
        # same query (hence same row order) as the reference implementation
        rel = duckdb.query(f"SELECT market_id, timestamp, features FROM read_parquet({src})")
        tbl = rel.to_arrow_table() if hasattr(rel, "to_arrow_table") else rel.fetch_arrow_table()
        n = tbl.num_rows
        if n == 0:
            self._markets, self._code_of = [], {}
            self._ts = np.empty(0, dtype=np.int64)
            self._feat = np.empty((0, FEATURE_DIM), dtype=np.float32)
            self._start = self._stop = np.empty(0, dtype=np.int64)
            self._ev_ts = self._ev_code = self._ev_row = np.empty(0, dtype=np.int64)
            self._open_mask = np.empty(0, dtype=bool)
            return

        ts_dt = tbl.column("timestamp").combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[ns]")
        flat = tbl.column("features").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        if flat.shape[0] != n * FEATURE_DIM:
            raise ValueError(f"LOB feature rows are not all of length {FEATURE_DIM}")
        feats = flat.reshape(n, FEATURE_DIM).astype(np.float32)
        # codes in order of first appearance == groupby(sort=False) group order of the reference
        codes, uniques = pd.factorize(tbl.column("market_id").to_pandas(), sort=False)
        codes = codes.astype(np.int64)

        # contiguous groups, original (file) order inside each group
        order = np.argsort(codes, kind="stable")
        g_codes = codes[order]
        g_ts = ts_dt[order]
        n_markets = len(uniques)
        counts = np.bincount(g_codes, minlength=n_markets)
        stop = np.cumsum(counts)
        start = stop - counts

        # The reference sorts each group with DataFrame.sort_values("timestamp").  When a group is
        # already strictly increasing every correct sort is the identity; otherwise (ties, or rows
        # out of order) reproduce the reference's permutation with the very same pandas call.
        ts_i8 = g_ts.astype(np.int64)
        non_increasing = np.zeros(n, dtype=bool)
        non_increasing[1:] = (ts_i8[1:] <= ts_i8[:-1]) & (g_codes[1:] == g_codes[:-1])
        for c in np.unique(g_codes[non_increasing]):
            s, e = int(start[c]), int(stop[c])
            perm = pd.DataFrame({"timestamp": g_ts[s:e]}).sort_values("timestamp").index.to_numpy()
            order[s:e] = order[s:e][perm]
            ts_i8[s:e] = ts_i8[s:e][perm]

        self._markets = list(uniques)
        self._code_of = {m: i for i, m in enumerate(self._markets)}
        self._ts = ts_i8
        self._feat = np.ascontiguousarray(feats[order])
        self._start, self._stop = start, stop

        # time-sorted event stream; the stable sort keeps tied rows of a market in their row order
        ev = np.argsort(ts_i8, kind="stable")
        self._ev_ts = ts_i8[ev]
        self._ev_row = ev.astype(np.int64)
        self._ev_code = g_codes[ev]

        self._open_mask = np.fromiter((m in self._open_markets for m in self._markets), dtype=bool, count=n_markets)
        logger.info(
            f"Reloaded LOB data: {n_markets} markets, {n} rows in {time.time() - t0:.3f}s. "
            f"open_with_data updated to {int(self._open_mask.sum())} markets."
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset_to_time(self, t_ns: int) -> None:
        self._lifecycle_ptr = 0
        self._open_markets = set()
        self._open_mask[:] = False
        self._advance_lifecycle(t_ns)
        logger.info(
            f"Lifecycle reset to t={pd.Timestamp(t_ns)}: {len(self._open_markets)} open markets total, "
            f"{int(self._open_mask.sum())} with LOB data (will be scanned)"
        )

    def _advance_lifecycle(self, t_end_ns: int) -> None:
        """Applies, in order, every lifecycle event with timestamp <= t_end_ns."""
        end = int(np.searchsorted(self._lc_ts, t_end_ns, side="right"))
        ptr = self._lifecycle_ptr
        if end <= ptr:
            return
        code_of, open_mask, open_markets = self._code_of, self._open_mask, self._open_markets
        markets, is_open = self._lc_market, self._lc_is_open
        for i in range(ptr, end):
            m = markets[i]
            c = code_of.get(m)
            if is_open[i]:
                open_markets.add(m)
                if c is not None:
                    open_mask[c] = True
            else:
                open_markets.discard(m)
                if c is not None:
                    open_mask[c] = False
        self._lifecycle_ptr = end

    # ── queries ──────────────────────────────────────────────────────────────

    def _windows_for(self, codes: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Windows (N, K, OUTPUT_DIM) made of the K rows before ``ends`` (exclusive) of each market."""
        K = self.K
        rows = ends[:, None] - K + np.arange(K)[None, :]                 # (N, K) global row indices
        valid = rows >= self._start[codes][:, None]
        base = self._feat[np.where(valid, rows, 0)]                      # (N, K, 52)
        base[~valid] = 0.0                                               # zero padding at the beginning

        # momentum features on the true mid price, over real rows only (see LOBWindowBuffer.get_window)
        n_real = valid.sum(axis=1)
        first = K - n_real                                               # index of the first real row
        pos = np.arange(K)[None, :]
        mid = base[:, :, MID_IDX]
        momentum = np.zeros((len(codes), K, 3), dtype=np.float32)
        d1 = np.zeros_like(mid)
        d1[:, 1:] = mid[:, 1:] - mid[:, :-1]
        momentum[:, :, 0] = np.where(pos > first[:, None], d1, np.float32(0.0))
        mid_first = mid[np.arange(len(codes)), np.minimum(first, K - 1)]
        dk = np.where(pos >= first[:, None], mid - mid_first[:, None], np.float32(0.0))
        momentum[:, :, 1] = dk
        momentum[:, :, 2] = np.sign(dk)
        return np.concatenate([base, momentum], axis=2)

    def get_window(self, market_id: str, t_ns: int) -> np.ndarray:
        """Last K snapshots of ``market_id`` strictly before ``t_ns`` (zero padded), shape (K, OUTPUT_DIM)."""
        c = self._code_of.get(market_id)
        if c is None:
            return np.zeros((self.K, OUTPUT_DIM), dtype=np.float32)
        s, e = int(self._start[c]), int(self._stop[c])
        end = s + int(np.searchsorted(self._ts[s:e], t_ns, side="left"))
        if end == s:
            return np.zeros((self.K, OUTPUT_DIM), dtype=np.float32)
        return self._windows_for(np.array([c], dtype=np.int64), np.array([end], dtype=np.int64))[0]

    def load_step(self, t_start_ns: int, t_end_ns: int) -> Tuple[List[str], np.ndarray]:
        """
        Same contract as ``LOBWindowBuffer.load_step``: advances the lifecycle to
        ``t_end_ns`` and returns the open markets with a snapshot in
        [t_start_ns, t_end_ns) together with their windows ending before ``t_end_ns``.
        """
        self._advance_lifecycle(t_end_ns)
        lo = int(np.searchsorted(self._ev_ts, t_start_ns, side="left"))
        hi = int(np.searchsorted(self._ev_ts, t_end_ns, side="left"))
        if hi <= lo:
            return [], np.empty((0, self.K, OUTPUT_DIM), dtype=np.float32)
        # last event of each market in the slice = its last row strictly before t_end
        codes_rev = self._ev_code[lo:hi][::-1]
        codes, last_rev = np.unique(codes_rev, return_index=True)
        keep = self._open_mask[codes]
        if not keep.any():
            return [], np.empty((0, self.K, OUTPUT_DIM), dtype=np.float32)
        codes, last_rev = codes[keep], last_rev[keep]
        ends = self._ev_row[lo:hi][::-1][last_rev] + 1
        windows = self._windows_for(codes, ends)
        markets = self._markets
        return [markets[c] for c in codes.tolist()], windows
