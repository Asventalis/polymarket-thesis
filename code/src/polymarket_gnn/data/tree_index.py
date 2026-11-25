"""
tree_index.py

Causal index over the cross-fitted tree log-probabilities of the LOB tensor rows
(``tree_features.py``) for the training loop: a market at block time ``t`` gets the row with the
largest ``ts <= t`` (the row of the window the model reads in the same block, so the lookup is
causal by construction; the trees that produced the row never saw its day).

Layout of a directory:  meta.json (settings, check against the control) and
day=YYYY-MM-DD.parquet (market_id, ts ns, lp_down, lp_const, lp_up: natural log-probabilities).

Served as (i) four input columns [lp_down, lp_const, lp_up, available] (zeros when there is no row)
and (ii) the (N, 3) logit offset, zeros when there is no row: with ``--tree-offset`` the model's
logits are added to the trees' log-probabilities, so it starts from their solution and learns the
residual.
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
LP_COLS = ["lp_down", "lp_const", "lp_up"]
NAMES = LP_COLS + ["tree_available"]


def day_file(root: Path, date: pd.Timestamp) -> Path:
    return Path(root) / f"day={date.strftime('%Y-%m-%d')}.parquet"


class TreeIndex:
    """Sliding per-market (ts, log-probabilities) arrays over the directory, two days resident."""

    def __init__(self, root: Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if self.meta and list(self.meta.get("columns", LP_COLS)) != LP_COLS:
            raise ValueError(f"meta.json columns {self.meta.get('columns')} != {LP_COLS}")
        self.names: List[str] = list(NAMES)
        self.n_features = len(self.names)
        self._day_cache: Dict[pd.Timestamp, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._missing_days: set = set()
        self._ts: Dict[str, np.ndarray] = {}
        self._lp: Dict[str, np.ndarray] = {}
        self._current_key: Optional[int] = None
        self.n_available = 0
        self.n_queried = 0

    def _load_day(self, date: pd.Timestamp):
        path = day_file(self.root, date)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["market_id", "ts"] + LP_COLS)
        df = df.sort_values(["market_id", "ts"], kind="stable")
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if df.empty:
            return out
        markets = df["market_id"].values
        ts = df["ts"].values.astype(np.int64)
        lp = np.nan_to_num(df[LP_COLS].values.astype(np.float32), nan=0.0)
        boundaries = np.flatnonzero(markets[1:] != markets[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(markets)]])
        for s, e in zip(starts, ends):
            out[markets[s]] = (ts[s:e], lp[s:e])
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
        self._ts, self._lp = {}, {}
        for m, lst in parts.items():
            if len(lst) == 1:
                self._ts[m], self._lp[m] = lst[0]
            else:
                self._ts[m] = np.concatenate([a[0] for a in lst])
                self._lp[m] = np.concatenate([a[1] for a in lst])
        self._current_key = key
        logger.info(f"TreeIndex hour={t_hour} loaded {n_loaded} new day(s), resident={[d.strftime('%m-%d') for d in self._day_cache]}, "
                    f"missing={[d.strftime('%m-%d') for d in sorted(self._missing_days)]}, markets={len(self._ts)} in {time.time() - t0:.2f}s")

    def last_row(self, market_id: str, t_ns: int) -> int:
        """Index of the last row with ts <= t, or -1."""
        ts = self._ts.get(market_id)
        if ts is None:
            return -1
        return int(ts.searchsorted(t_ns, side="right")) - 1

    def log_probs(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """(N, 3) float32 log-probabilities of the last row at or before ``t``; NaN rows where there is none."""
        out = np.full((len(market_ids), 3), np.nan, dtype=np.float32)
        self.n_queried += len(market_ids)
        for j, m in enumerate(market_ids):
            i = self.last_row(m, t_ns)
            if i >= 0:
                out[j] = self._lp[m][i]
                self.n_available += 1
        return out

    def features(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """(N, 4): [lp_down, lp_const, lp_up, available]; all zeros when there is no row."""
        lp = self.log_probs(market_ids, t_ns)
        available = ~np.isnan(lp[:, 0])
        out = np.zeros((len(market_ids), self.n_features), dtype=np.float32)
        out[available, :3] = lp[available]
        out[available, 3] = 1.0
        return out

    def offset(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """(N, 3) logit offset: the log-probabilities, zeros (a flat prior) when there is no row."""
        return np.nan_to_num(self.log_probs(market_ids, t_ns), nan=0.0)
