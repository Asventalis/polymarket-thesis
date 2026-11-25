"""
hand_index.py

Causal index over the pre-computed hand-made features of the LOB tensor rows
(``hand_features.py``), turned into per-market feature vectors for the training loop.

Layout of a feature directory:

    meta.json              stored columns, derived columns, category names, robust location /
                           scale of every model feature on the rows of the training days
    markets.parquet        market_id, category (code), is_yesno, start_ns, end_ns, duration_h
    day=YYYY-MM-DD.parquet market_id, ts (ns), t_last_move (ns, null if the mid never moved),
                           one float column per stored feature

Features for a market at block time ``t`` come from the LAST row with ``ts <= t`` (the row of
the window the model reads in the same block, so the lookup is causal by construction):

    standardised stored features (47) ............ (x - loc) / scale, clipped to ±10, NaN → 0
    sign of the 15m / 1h / 3h / 6h / 12h returns (5)
    hour of ``t`` as sin / cos (2)
    log1p of: seconds since the last mid change, seconds since the last trade, hours to the
      end of the market, hours since its start, its duration (5), standardised
    is_yesno (1), one-hot category (8)
    available (1) ................................. 0 (and all zeros) when there is no row

The order of the stored columns and the statistics come from ``meta.json`` so that the index
and the builder cannot drift apart.
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
CLIP = 10.0
RETURN_COLS = ["ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h"]
DERIVED = ["log_since_move", "log_since_trade", "log_hours_to_end", "log_hours_since_start", "log_duration_h"]


def day_file(root: Path, date: pd.Timestamp) -> Path:
    return Path(root) / f"day={date.strftime('%Y-%m-%d')}.parquet"


class HandFeatureIndex:
    """Sliding per-market (ts, features) arrays over the feature directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            self.meta = json.load(f)
        self.stored: List[str] = list(self.meta["stored"])
        self.numeric: List[str] = [c for c in self.stored if c != "since_trade_s"]
        self.categories: List[str] = list(self.meta["categories"])
        if list(self.meta["derived"]) != DERIVED:
            raise ValueError(f"meta.json derived columns {self.meta['derived']} != {DERIVED}")
        stats = self.meta["stats"]
        self._loc = np.array([stats[c]["loc"] for c in self.numeric], dtype=np.float32)
        self._scale = np.array([stats[c]["scale"] for c in self.numeric], dtype=np.float32)
        self._dloc = np.array([stats[c]["loc"] for c in DERIVED], dtype=np.float32)
        self._dscale = np.array([stats[c]["scale"] for c in DERIVED], dtype=np.float32)
        self.never_moved_s = float(self.meta.get("never_moved_seconds", 14 * 86400))
        self.never_traded_s = float(self.meta.get("never_traded_seconds", 30 * 86400))
        self.names: List[str] = (self.numeric + [f"sgn_{c[4:]}" for c in RETURN_COLS] + ["hour_sin", "hour_cos"]
                                 + DERIVED + ["is_yesno"] + [f"cat_{c}" for c in self.categories] + ["available"])
        self.n_features = len(self.names)
        mk = pd.read_parquet(self.root / "markets.parquet")
        self._static: Dict[str, Tuple[int, float, float, float, float]] = {
            m: (int(c), float(y), float(s), float(e), float(d))
            for m, c, y, s, e, d in zip(mk["market_id"].values, mk["category"].values, mk["is_yesno"].values,
                                        mk["start_ns"].values.astype(np.float64), mk["end_ns"].values.astype(np.float64),
                                        mk["duration_h"].values)
        }
        self._day_cache: Dict[pd.Timestamp, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._missing_days: set = set()
        self._ts: Dict[str, np.ndarray] = {}
        self._tlm: Dict[str, np.ndarray] = {}
        self._str: Dict[str, np.ndarray] = {}
        self._z: Dict[str, np.ndarray] = {}
        self._current_key: Optional[int] = None
        self.n_available = 0
        self.n_queried = 0

    # ── loading ──────────────────────────────────────────────────────────────

    def _standardise(self, df: pd.DataFrame) -> np.ndarray:
        """(rows, 47 + 5): standardised numerics then the return signs."""
        x = df[self.numeric].values.astype(np.float32)
        with np.errstate(invalid="ignore"):
            z = (x - self._loc) / self._scale
        z = np.clip(np.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
        r = df[RETURN_COLS].values.astype(np.float32)
        s = np.sign(np.nan_to_num(r, nan=0.0)).astype(np.float32)
        return np.concatenate([z, s], axis=1)

    def _load_day(self, date: pd.Timestamp):
        path = day_file(self.root, date)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df = df.sort_values(["market_id", "ts"], kind="stable")
        out = {}
        if df.empty:
            return out
        markets = df["market_id"].values
        ts = df["ts"].values.astype(np.int64)
        tlm = pd.to_numeric(df["t_last_move"], errors="coerce").values.astype(np.float64)   # NaN = never moved
        since_trade = pd.to_numeric(df["since_trade_s"], errors="coerce").values.astype(np.float64)
        z = self._standardise(df)
        boundaries = np.flatnonzero(markets[1:] != markets[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(markets)]])
        for s, e in zip(starts, ends):
            out[markets[s]] = (ts[s:e], tlm[s:e], since_trade[s:e], z[s:e])
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
        self._ts, self._tlm, self._str, self._z = {}, {}, {}, {}
        for m, lst in parts.items():
            if len(lst) == 1:
                self._ts[m], self._tlm[m], self._str[m], self._z[m] = lst[0]
            else:
                self._ts[m] = np.concatenate([a[0] for a in lst])
                self._tlm[m] = np.concatenate([a[1] for a in lst])
                self._str[m] = np.concatenate([a[2] for a in lst])
                self._z[m] = np.concatenate([a[3] for a in lst])
        self._current_key = key
        logger.info(f"HandFeatureIndex hour={t_hour} loaded {n_loaded} new day(s), resident={[d.strftime('%m-%d') for d in self._day_cache]}, "
                    f"missing={[d.strftime('%m-%d') for d in sorted(self._missing_days)]}, markets={len(self._ts)} in {time.time() - t0:.2f}s")

    # ── queries ──────────────────────────────────────────────────────────────

    def last_row(self, market_id: str, t_ns: int) -> int:
        """Index of the last row with ts <= t, or -1."""
        ts = self._ts.get(market_id)
        if ts is None:
            return -1
        return int(ts.searchsorted(t_ns, side="right")) - 1

    def features(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """(len(market_ids), n_features) float32 feature matrix at block time ``t_ns``."""
        n = len(market_ids)
        out = np.zeros((n, self.n_features), dtype=np.float32)
        self.n_queried += n
        if n == 0:
            return out
        n_z = len(self.numeric) + len(RETURN_COLS)
        rows = [self.last_row(m, t_ns) for m in market_ids]
        ok = np.array([i >= 0 for i in rows], dtype=bool)
        if not ok.any():
            return out
        idx = np.flatnonzero(ok)
        z = np.stack([self._z[market_ids[j]][rows[j]] for j in idx])
        ts_row = np.array([self._ts[market_ids[j]][rows[j]] for j in idx], dtype=np.float64)
        tlm = np.array([self._tlm[market_ids[j]][rows[j]] for j in idx], dtype=np.float64)
        st = np.array([self._str[market_ids[j]][rows[j]] for j in idx], dtype=np.float64)
        static = np.array([self._static.get(market_ids[j], (len(self.categories) - 1, 0.0, np.nan, np.nan, np.nan)) for j in idx],
                          dtype=np.float64)
        cat = static[:, 0].astype(int); yesno = static[:, 1]; start = static[:, 2]; end = static[:, 3]; dur = static[:, 4]
        t = float(t_ns)
        hour = ((t_ns // HOUR_NS) % 24) + ((t_ns % HOUR_NS) / HOUR_NS)
        since_move = np.where(np.isnan(tlm), self.never_moved_s, (t - np.nan_to_num(tlm)) / 1e9)
        since_trade = np.where(np.isnan(st), self.never_traded_s, st + (t - ts_row) / 1e9)
        with np.errstate(invalid="ignore"):
            derived = np.stack([
                np.log1p(np.clip(since_move, 0, None)),
                np.log1p(np.clip(since_trade, 0, None)),
                np.log1p(np.clip(np.nan_to_num((end - t) / 3.6e12), 0, None)),
                np.log1p(np.clip(np.nan_to_num((t - start) / 3.6e12), 0, None)),
                np.log1p(np.clip(np.nan_to_num(dur), 0, None)),
            ], axis=1).astype(np.float32)
            derived = np.clip((derived - self._dloc) / self._dscale, -CLIP, CLIP)
        k = n_z
        out[idx, :k] = z
        out[idx, k] = np.sin(2 * np.pi * hour / 24); out[idx, k + 1] = np.cos(2 * np.pi * hour / 24); k += 2
        out[idx, k:k + len(DERIVED)] = derived; k += len(DERIVED)
        out[idx, k] = yesno; k += 1
        out[idx, k + np.clip(cat, 0, len(self.categories) - 1)] = 1.0; k += len(self.categories)
        out[idx, k] = 1.0
        self.n_available += len(idx)
        return out
