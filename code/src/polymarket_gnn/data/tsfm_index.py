"""
tsfm_index.py

Causal index over pre-computed foundation-model forecasts of the mid price
(``tsfm_cache.py``), turned into per-market features for the training loop.

Layout of a cache directory (one per model / horizon / grid):

    meta.json                      horizon, context, step_min, grid_min, steps (minutes ahead
                                   kept), quantiles (10, 50, 90), model id
    day=YYYY-MM-DD.parquet         one row per (market_id, t_grid) forecast made at grid time
                                   t_grid from the mids strictly before t_grid:
                                   market_id, t_grid (ns), mid_last, known,
                                   q{Q}_{S} for every step S and quantile Q (price levels)

Features for a market at block time ``t`` (``n_features = 3 * len(steps) + 2``):

    scale * (q50_S - mid(t)), scale * (q10_S - mid(t)), scale * (q90_S - mid(t))   per step S
    sign(q50_last - mid(t))
    available

from the LAST forecast with ``t_grid <= t`` that is at most ``max_stale_ns`` old.  A forecast
made at ``t_grid`` only reads prices before ``t_grid <= t``, so the feature is causal at ``t``;
expressing the forecast relative to the CURRENT mid (not the mid the forecast was made from)
turns it into the move still expected from now, and absorbs whatever happened between
``t_grid`` and ``t``.  All zeros (available = 0) when there is no recent forecast or ``mid(t)``
is unknown.  The same three-column convention as ``MidPriceIndex.lookback_features``
(scaled change, sign, available) lets the direct path read them side by side.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
MIN_NS = 60_000_000_000


def day_file(root: Path, date: pd.Timestamp) -> Path:
    return Path(root) / f"day={date.strftime('%Y-%m-%d')}.parquet"


def quantile_columns(steps: Sequence[int], quantiles: Sequence[int] = (50, 10, 90)) -> List[str]:
    return [f"q{q}_{s}" for s in steps for q in quantiles]


class TSFMIndex:
    """
    Sliding per-market (t_grid, quantiles) arrays over the forecast cache.

    Args:
        root:          cache directory written by ``tsfm_cache.py``
        max_stale_ns:  a forecast older than this at query time is not used (available = 0)
        scale:         multiplier of the price changes (10, as the lookback features)
    """

    def __init__(self, root: Path, max_stale_ns: int, scale: float = 10.0):
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            self.meta = json.load(f)
        self.steps: List[int] = [int(s) for s in self.meta["steps"]]
        self.cols = quantile_columns(self.steps)
        self.n_features = 3 * len(self.steps) + 2
        self.max_stale_ns = int(max_stale_ns)
        self.scale = float(scale)
        self._day_cache: Dict[pd.Timestamp, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._missing_days: set = set()
        self._ts: Dict[str, np.ndarray] = {}
        self._q: Dict[str, np.ndarray] = {}
        self._current_key: Optional[int] = None
        self.n_available = 0
        self.n_queried = 0

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_day(self, date: pd.Timestamp) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        path = day_file(self.root, date)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["market_id", "t_grid"] + self.cols)
        df = df.sort_values(["market_id", "t_grid"], kind="stable")
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if df.empty:
            return out
        markets = df["market_id"].values
        ts = df["t_grid"].values.astype(np.int64)
        q = df[self.cols].values.astype(np.float32)
        boundaries = np.flatnonzero(markets[1:] != markets[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(markets)]])
        for s, e in zip(starts, ends):
            out[markets[s]] = (ts[s:e], q[s:e])
        return out

    def set_hour(self, year: int, month: int, day: int, hour: int) -> None:
        """Make the day of the hour (and the previous one if the staleness window crosses midnight) resident."""
        t_hour = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        key = t_hour.value // HOUR_NS
        if key == self._current_key:
            return
        t0 = time.time()
        wanted = sorted({(t_hour - pd.Timedelta(self.max_stale_ns, unit="ns")).normalize(), t_hour.normalize()})
        for d in list(self._day_cache):
            if d not in wanted:
                del self._day_cache[d]
        self._missing_days = {d for d in self._missing_days if d in wanted}
        n_loaded = 0
        for d in wanted:
            if d in self._day_cache or d in self._missing_days:
                continue
            data = self._load_day(d)
            if data is None:
                self._missing_days.add(d)
            else:
                self._day_cache[d] = data
                n_loaded += 1
        parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for d in wanted:
            for m, arrs in self._day_cache.get(d, {}).items():
                parts.setdefault(m, []).append(arrs)
        self._ts, self._q = {}, {}
        for m, lst in parts.items():
            if len(lst) == 1:
                self._ts[m], self._q[m] = lst[0]
            else:
                self._ts[m] = np.concatenate([a for a, _ in lst])
                self._q[m] = np.concatenate([b for _, b in lst])
        self._current_key = key
        logger.info(f"TSFMIndex hour={t_hour} loaded {n_loaded} new day(s), resident={[d.strftime('%m-%d') for d in self._day_cache]}, "
                    f"missing={[d.strftime('%m-%d') for d in sorted(self._missing_days)]}, markets={len(self._ts)} in {time.time() - t0:.2f}s")

    # ── queries ──────────────────────────────────────────────────────────────

    def latest(self, market_id: str, t_ns: int) -> Optional[np.ndarray]:
        """Quantile levels of the last forecast made at t_grid <= t within the staleness window, else None."""
        ts = self._ts.get(market_id)
        if ts is None:
            return None
        i = int(ts.searchsorted(t_ns, side="right")) - 1
        if i < 0 or t_ns - int(ts[i]) > self.max_stale_ns:
            return None
        return self._q[market_id][i]

    def features(self, market_ids: Sequence[str], t_ns: int, mid_now: np.ndarray) -> np.ndarray:
        """
        (len(market_ids), n_features) float32: [scaled change of q50, q10, q90 per step..., sign, available].
        ``mid_now[j]`` is the current mid of market j (NaN = unknown → all zeros).
        """
        n = len(market_ids)
        out = np.zeros((n, self.n_features), dtype=np.float32)
        self.n_queried += n
        for j, m in enumerate(market_ids):
            mid = float(mid_now[j])
            if np.isnan(mid):
                continue
            q = self.latest(m, t_ns)
            if q is None:
                continue
            d = q.astype(np.float64) - mid
            out[j, :-2] = self.scale * d
            d50 = d[3 * (len(self.steps) - 1)]                             # q50 of the last step
            out[j, -2] = np.sign(d50) if abs(d50) > 1e-6 else 0.0           # float32 noise is not a direction
            out[j, -1] = 1.0
            self.n_available += 1
        return out
