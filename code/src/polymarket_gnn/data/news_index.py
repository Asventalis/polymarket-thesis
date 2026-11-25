"""
news_index.py

Causal index over the GDELT news-volume features of the LOB tensor rows (``news_features.py rows
--tensor-days``) for the training loop: a market at block time ``t`` gets the row with the largest
``ts <= t`` (the row of the window the model reads in the same block; the row's features only use
GKG files published before its own ``ts`` minus a lag, so the lookup is causal by construction).

Layout of a directory:  meta.json (``features``: column names; ``stats``: robust location / scale
of every column on the rows of the training days) and day=YYYY-MM-DD.parquet (market_id, ts ns,
one float column per feature).

Served as ``n_features = len(features) + 1`` input columns: the standardised features (clipped to
±10, NaN → 0) and ``news_available`` (0, with all zeros, when there is no row).
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
CLIP = 10.0


def day_file(root: Path, date: pd.Timestamp) -> Path:
    return Path(root) / f"day={date.strftime('%Y-%m-%d')}.parquet"


class NewsIndex:
    """Sliding per-market (ts, features) arrays over the directory, two days resident."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.columns: List[str] = list(self.meta["features"])
        stats = self.meta.get("stats", {})
        self._loc = np.array([stats.get(c, {}).get("loc", 0.0) for c in self.columns], dtype=np.float32)
        self._scale = np.array([stats.get(c, {}).get("scale", 1.0) for c in self.columns], dtype=np.float32)
        self._scale = np.where(self._scale > 0, self._scale, 1.0).astype(np.float32)
        self.names: List[str] = self.columns + ["news_available"]
        self.n_features = len(self.names)
        self._day_cache: Dict[pd.Timestamp, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._missing_days: set = set()
        self._ts: Dict[str, np.ndarray] = {}
        self._x: Dict[str, np.ndarray] = {}
        self._current_key: Optional[int] = None
        self.n_available = 0
        self.n_queried = 0

    def _load_day(self, date: pd.Timestamp):
        path = day_file(self.root, date)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["market_id", "ts"] + self.columns)
        df = df.sort_values(["market_id", "ts"], kind="stable")
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if df.empty:
            return out
        markets = df["market_id"].values
        ts = df["ts"].values.astype(np.int64)
        x = (df[self.columns].values.astype(np.float32) - self._loc) / self._scale
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP).astype(np.float32)
        boundaries = np.flatnonzero(markets[1:] != markets[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(markets)]])
        for s, e in zip(starts, ends):
            out[markets[s]] = (ts[s:e], x[s:e])
        return out

    def set_hour(self, year: int, month: int, day: int, hour: int) -> None:
        """Make the day of the hour and the previous one resident (a block just after midnight reads yesterday's rows)."""
        t_hour = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        key = t_hour.value // HOUR_NS
        if key == self._current_key:
            return
        t0 = time.time()
        wanted = sorted({(t_hour - pd.Timedelta(1, unit="D")).normalize(), t_hour.normalize()})
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
        parts: Dict[str, list] = {}
        for d in wanted:
            for m, arrs in self._day_cache.get(d, {}).items():
                parts.setdefault(m, []).append(arrs)
        self._ts, self._x = {}, {}
        for m, lst in parts.items():
            if len(lst) == 1:
                self._ts[m], self._x[m] = lst[0]
            else:
                self._ts[m] = np.concatenate([a[0] for a in lst])
                self._x[m] = np.concatenate([a[1] for a in lst])
        self._current_key = key
        logger.info(f"NewsIndex hour={t_hour} loaded {n_loaded} new day(s), resident={[d.strftime('%m-%d') for d in self._day_cache]}, "
                    f"missing={[d.strftime('%m-%d') for d in sorted(self._missing_days)]}, markets={len(self._ts)} in {time.time() - t0:.2f}s")

    def last_row(self, market_id: str, t_ns: int) -> int:
        """Index of the last row with ts <= t, or -1."""
        ts = self._ts.get(market_id)
        if ts is None:
            return -1
        return int(ts.searchsorted(t_ns, side="right")) - 1

    def features(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """(N, n_features): standardised features of the last row at or before ``t`` and ``available``; zeros without a row."""
        out = np.zeros((len(market_ids), self.n_features), dtype=np.float32)
        self.n_queried += len(market_ids)
        for j, m in enumerate(market_ids):
            i = self.last_row(m, t_ns)
            if i >= 0:
                out[j, :-1] = self._x[m][i]
                out[j, -1] = 1.0
                self.n_available += 1
        return out
