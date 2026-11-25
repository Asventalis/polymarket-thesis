"""
price_index.py

Causal mid-price index over the precomputed LOB tensors
(data/derived/tensors/year=/month=/day=/hour=/lob_features.parquet).

Why this module exists
----------------------
1. The LOB tensor layout (see ``lob_layout.py``) stores the mid-price at
   column 40.  Column 0 is ``(best_bid - mid) / mid``.  Targets and momentum
   features were previously computed on column 0, i.e. on the relative
   half-spread instead of the price.
2. A target at ``t + H`` needs the last snapshot *strictly before* ``t + H``,
   which can live in any hour between ``h`` and ``h + H``.  Previously only
   hours ``h`` and ``h + H`` were loaded, so a market that moved in between and
   was quiet in hour ``h + H`` was mislabelled CONSTANT.
3. If ``t + H`` falls beyond the data that is actually on disk, the label is
   unknown (NaN), not "whatever the last price was".

The index keeps a sliding set of hours ``[h - hours_back, h + hours_ahead]`` of
``(timestamp, mid)`` pairs per market (two small arrays, not the 52-dim rows)
and answers, all strictly causally with respect to the query time ``t``:

    mid_before(market, t)          last mid strictly before t
    labels(markets, t, H)          3-class label of mid(t+H) vs mid(t)   (needs future data)
    past_change(markets, t, L)     mid(t) - mid(t-L)                      (causal momentum)
    last_change(markets, t)        last observed tick-to-tick change      (causal momentum)
    active_between(t0, t1)         markets with a snapshot in [t0, t1)

"Strictly before" mirrors ``LOBWindowBuffer.get_window`` so that the price the
model is asked to forecast from is the same one the window ends with.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

from polymarket_gnn.data.lob_layout import MID_IDX

logger = logging.getLogger(__name__)

HOUR_NS = 3_600_000_000_000
DOWN, CONST, UP = 0, 1, 2


def hour_key(year: int, month: int, day: int, hour: int) -> int:
    """Integer hour index (ns since epoch // 1h) of a wall-clock hour."""
    return pd.Timestamp(year=year, month=month, day=day, hour=hour).value // HOUR_NS


def hour_key_to_path(base_dir: Path, key: int) -> Path:
    ts = pd.Timestamp(key * HOUR_NS)
    return (
        Path(base_dir)
        / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"
    )


def direction_label(delta: float, eps: float = 0.0) -> int:
    """UP if delta > eps, DOWN if delta < -eps, otherwise CONSTANT."""
    if delta > eps:
        return UP
    if delta < -eps:
        return DOWN
    return CONST


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class LifecycleTracker:
    """
    Vectorised replacement for the lifecycle logic in ``LOBWindowBuffer``.

    Events come from ``market_lifecycle_events.parquet``:
        OPEN  = deploying_timestamp          (known in advance → causal)
        CLOSE = last_transaction_timestamp   (known only in hindsight!)

    ``advance(t)`` applies all events with timestamp <= t.  Note that the CLOSE
    event is *not* causal: it is the time of the last trade, which nobody knows
    at time t.  It is used only to define the evaluation sample (see
    ``MidPriceIndex.close_filter``), never as a model input.
    """

    def __init__(self, lifecycle_path: Path):
        t0 = time.time()
        lc = pd.read_parquet(lifecycle_path)
        lc["event_timestamp"] = pd.to_datetime(lc["event_timestamp"])
        lc = lc.sort_values("event_timestamp", kind="stable").reset_index(drop=True)
        self._ts = lc["event_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        self._market = lc["market_id"].values
        self._is_open = (lc["event_type"].values == "OPEN")
        self._ptr = 0
        self.open_markets: set = set()

        close = lc[lc["event_type"] == "CLOSE"]
        # keep the LAST close event per market
        self.close_time_ns: Dict[str, int] = dict(
            zip(close["market_id"].values,
                close["event_timestamp"].values.astype("datetime64[ns]").astype(np.int64).tolist())
        )
        logger.info(
            f"Lifecycle events loaded: {len(lc)} "
            f"({int(self._is_open.sum())} OPEN, {int((~self._is_open).sum())} CLOSE) "
            f"in {time.time() - t0:.2f}s"
        )

    def reset_to(self, t_ns: int) -> None:
        self._ptr = 0
        self.open_markets = set()
        self.advance(t_ns)

    def advance(self, t_ns: int) -> None:
        end = int(np.searchsorted(self._ts, t_ns, side="right"))
        for i in range(self._ptr, end):
            if self._is_open[i]:
                self.open_markets.add(self._market[i])
            else:
                self.open_markets.discard(self._market[i])
        self._ptr = max(self._ptr, end)

    def is_open(self, market_id: str) -> bool:
        return market_id in self.open_markets


# ─────────────────────────────────────────────────────────────────────────────
# Mid-price index
# ─────────────────────────────────────────────────────────────────────────────

class MidPriceIndex:
    """
    Sliding per-market (timestamp, mid) arrays over a window of hours.

    Args:
        base_dir:      ``.../data/derived``
        hours_back:    hours before the current hour to keep (for causal momentum)
        hours_ahead:   hours after the current hour to keep (for labels at t+H)
        lifecycle:     optional LifecycleTracker (open-market filter + close times)
        close_filter:  'last_tx' → label is NaN when the market's last trade is
                       before t+H (NOT causal: selection on a future event, kept
                       for comparability with earlier runs); 'none' → no filter.
        const_eps:     |mid(t+H) - mid(t)| <= const_eps → CONSTANT
        mid_idx:       column of the mid price in the tensor (default: 40)
    """

    def __init__(
        self,
        base_dir: Path,
        hours_back: int = 0,
        hours_ahead: int = 1,
        lifecycle: Optional[LifecycleTracker] = None,
        close_filter: str = "last_tx",
        const_eps: float = 0.0,
        mid_idx: int = MID_IDX,
    ):
        if close_filter not in ("last_tx", "none"):
            raise ValueError(f"close_filter must be 'last_tx' or 'none', got {close_filter!r}")
        self.base_dir = Path(base_dir)
        self.hours_back = int(hours_back)
        self.hours_ahead = int(hours_ahead)
        self.lifecycle = lifecycle
        self.close_filter = close_filter
        self.const_eps = float(const_eps)
        self.mid_idx = int(mid_idx)

        self._hour_cache: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._missing_hours: set = set()
        self._ts: Dict[str, np.ndarray] = {}
        self._mid: Dict[str, np.ndarray] = {}
        self._current_key: Optional[int] = None
        self.coverage_start_ns: int = 0
        self.coverage_end_ns: int = 0
        # event stream of the current hour for active_between()
        self._ev_ts = np.empty(0, dtype=np.int64)
        self._ev_market = np.empty(0, dtype=object)

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_hour(self, key: int) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        path = hour_key_to_path(self.base_dir, key)
        if not path.exists():
            return None
        # DuckDB lists are 1-indexed
        df = duckdb.query(
            f"SELECT market_id, timestamp, features[{self.mid_idx + 1}] AS mid "
            f"FROM read_parquet('{path}') ORDER BY market_id, timestamp"
        ).df()
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if df.empty:
            return out
        markets = df["market_id"].values
        ts = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        mid = df["mid"].values.astype(np.float32)
        boundaries = np.flatnonzero(markets[1:] != markets[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(markets)]])
        for s, e in zip(starts, ends):
            out[markets[s]] = (ts[s:e], mid[s:e])
        return out

    def set_hour(self, year: int, month: int, day: int, hour: int) -> None:
        """Make hours [h - hours_back, h + hours_ahead] resident and rebuild the merged arrays."""
        key = hour_key(year, month, day, hour)
        if key == self._current_key:
            return
        t0 = time.time()
        wanted = list(range(key - self.hours_back, key + self.hours_ahead + 1))
        for k in list(self._hour_cache):
            if k not in wanted:
                del self._hour_cache[k]
        self._missing_hours = {k for k in self._missing_hours if k in wanted}
        n_loaded = 0
        for k in wanted:
            if k in self._hour_cache or k in self._missing_hours:
                continue
            data = self._load_hour(k)
            if data is None:
                self._missing_hours.add(k)
            else:
                self._hour_cache[k] = data
                n_loaded += 1

        # contiguous coverage around the current hour
        end = key
        while (end + 1) in self._hour_cache:
            end += 1
        start = key
        while (start - 1) in self._hour_cache:
            start -= 1
        if key in self._hour_cache:
            self.coverage_start_ns = start * HOUR_NS
            self.coverage_end_ns = (end + 1) * HOUR_NS
        else:  # current hour itself missing: nothing is observable
            self.coverage_start_ns = key * HOUR_NS
            self.coverage_end_ns = key * HOUR_NS

        self._rebuild(wanted)
        self._build_event_stream(key)
        self._current_key = key
        logger.info(
            f"MidPriceIndex hour={pd.Timestamp(key * HOUR_NS)} loaded {n_loaded} new hour(s), "
            f"resident={sorted(k - key for k in self._hour_cache)} (rel. hours), "
            f"missing={sorted(k - key for k in self._missing_hours)}, "
            f"markets={len(self._ts)} in {time.time() - t0:.2f}s"
        )

    def _rebuild(self, keys_in_order: Sequence[int]) -> None:
        parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for k in keys_in_order:
            if k not in self._hour_cache:
                continue
            for m, arrs in self._hour_cache[k].items():
                parts.setdefault(m, []).append(arrs)
        self._ts = {}
        self._mid = {}
        for m, lst in parts.items():
            if len(lst) == 1:
                self._ts[m], self._mid[m] = lst[0]
            else:
                self._ts[m] = np.concatenate([a for a, _ in lst])
                self._mid[m] = np.concatenate([b for _, b in lst])

    def _build_event_stream(self, key: int) -> None:
        data = self._hour_cache.get(key)
        if not data:
            self._ev_ts = np.empty(0, dtype=np.int64)
            self._ev_market = np.empty(0, dtype=object)
            return
        ts_parts, m_parts = [], []
        for m, (ts, _) in data.items():
            ts_parts.append(ts)
            m_parts.append(np.full(len(ts), m, dtype=object))
        ts_all = np.concatenate(ts_parts)
        m_all = np.concatenate(m_parts)
        order = np.argsort(ts_all, kind="stable")
        self._ev_ts = ts_all[order]
        self._ev_market = m_all[order]

    # ── queries ──────────────────────────────────────────────────────────────

    @property
    def markets(self) -> Iterable[str]:
        return self._ts.keys()

    def has_market(self, market_id: str) -> bool:
        return market_id in self._ts

    def mid_before(self, market_id: str, t_ns: int) -> float:
        ts = self._ts.get(market_id)
        if ts is None:
            return float("nan")
        i = int(ts.searchsorted(t_ns, side="left"))
        if i == 0:
            return float("nan")
        return float(self._mid[market_id][i - 1])

    def active_between(self, t_start_ns: int, t_end_ns: int, require_open: bool = True) -> List[str]:
        """Markets with at least one snapshot in [t_start, t_end) in the current hour."""
        lo = int(self._ev_ts.searchsorted(t_start_ns, side="left"))
        hi = int(self._ev_ts.searchsorted(t_end_ns, side="left"))
        if hi <= lo:
            return []
        seen = set()
        out: List[str] = []
        lifecycle = self.lifecycle if require_open else None
        for m in self._ev_market[lo:hi]:
            if m in seen:
                continue
            seen.add(m)
            if lifecycle is not None and not lifecycle.is_open(m):
                continue
            out.append(m)
        return out

    def future_change(self, market_ids: Sequence[str], t_ns: int, horizon_ns: int) -> np.ndarray:
        """
        Signed realised change mid(t+H) - mid(t) per market (float64, NaN = not defined).
        This is the regression target; ``labels`` is its 3-class discretisation.

        mid(t)   = last snapshot strictly before t
        mid(t+H) = last snapshot strictly before t+H (price is a step function,
                   so "no update in (t, t+H)" legitimately means a change of 0)
        NaN when: no snapshot before t, t+H beyond loaded coverage, or (close_filter)
                  the market's last trade precedes t+H.
        """
        n = len(market_ids)
        out = np.full(n, np.nan, dtype=np.float64)
        t_fut = t_ns + horizon_ns
        if t_fut > self.coverage_end_ns:
            return out
        close_times = self.lifecycle.close_time_ns if (self.lifecycle is not None and self.close_filter == "last_tx") else None
        for j, m in enumerate(market_ids):
            ts = self._ts.get(m)
            if ts is None:
                continue
            i_cur = int(ts.searchsorted(t_ns, side="left"))
            if i_cur == 0:
                continue
            if close_times is not None:
                ct = close_times.get(m)
                if ct is not None and ct < t_fut:
                    continue
            i_fut = int(ts.searchsorted(t_fut, side="left"))
            mid = self._mid[m]
            out[j] = float(mid[i_fut - 1]) - float(mid[i_cur - 1])
        return out

    def labels(self, market_ids: Sequence[str], t_ns: int, horizon_ns: int) -> np.ndarray:
        """
        3-class direction label of mid(t+H) vs mid(t) per market (float32, NaN = no label):
        the discretisation of ``future_change`` with the ``const_eps`` dead zone.
        """
        delta = self.future_change(market_ids, t_ns, horizon_ns)
        out = np.full(len(market_ids), np.nan, dtype=np.float32)
        ok = ~np.isnan(delta)
        d = delta[ok]
        eps = self.const_eps
        out[ok] = np.where(d > eps, UP, np.where(d < -eps, DOWN, CONST)).astype(np.float32)
        return out

    def past_change(self, market_ids: Sequence[str], t_ns: int, lookback_ns: int) -> np.ndarray:
        """mid(t) - mid(t - L), strictly causal.  NaN if either price is unobservable."""
        n = len(market_ids)
        out = np.full(n, np.nan, dtype=np.float32)
        t_past = t_ns - lookback_ns
        if t_past < self.coverage_start_ns:
            return out
        for j, m in enumerate(market_ids):
            ts = self._ts.get(m)
            if ts is None:
                continue
            i_cur = int(ts.searchsorted(t_ns, side="left"))
            i_past = int(ts.searchsorted(t_past, side="left"))
            if i_cur == 0 or i_past == 0:
                continue
            mid = self._mid[m]
            out[j] = float(mid[i_cur - 1]) - float(mid[i_past - 1])
        return out

    def lookback_features(
        self,
        market_ids: Sequence[str],
        t_ns: int,
        lookbacks_ns: Sequence[int],
        scale: float = 10.0,
    ) -> np.ndarray:
        """
        Causal long-lookback price features, 3 per lookback L:

            scale * (mid(t) - mid(t - L)),  sign(mid(t) - mid(t - L)),  available

        ``available`` is 0 (and the other two are 0) when either price is not
        observable (market younger than L, or t - L before the loaded coverage).
        The K-snapshot LOB window only spans minutes; these features give the
        model the same information the Momentum-H / Mean-Reversion-H baselines
        use.  Requires ``hours_back >= ceil(max(L) / 1h)``.

        Returns float32 array of shape (len(market_ids), 3 * len(lookbacks_ns)).
        """
        n = len(market_ids)
        out = np.zeros((n, 3 * len(lookbacks_ns)), dtype=np.float32)
        for k, lb in enumerate(lookbacks_ns):
            delta = self.past_change(market_ids, t_ns, int(lb))
            ok = ~np.isnan(delta)
            d = np.where(ok, delta, 0.0).astype(np.float32)
            out[:, 3 * k] = scale * d
            out[:, 3 * k + 1] = np.sign(d)
            out[:, 3 * k + 2] = ok.astype(np.float32)
        return out

    def logodds_features(
        self,
        market_ids: Sequence[str],
        t_ns: int,
        lookbacks_ns: Sequence[int],
        clip: float = 0.001,
    ) -> np.ndarray:
        """
        Causal price features in log-odds, 1 + len(lookbacks_ns) per market:

            logit(mid(t)) / 5,   logit(mid(t)) - logit(mid(t - L))  for every lookback L

        with the mids clipped to [clip, 1 - clip] (logit in ±6.9 for clip = 0.001).  A change is
        0 when either price is not observable (same rule as ``lookback_features``), the level is
        0 when the market has no snapshot before t.  In price units a move from 0.01 to 0.02 is
        as small as one from 0.50 to 0.51; in log-odds it is 17 times larger, which is how much
        more it matters for a market near the boundary.

        Returns float32 array of shape (len(market_ids), 1 + len(lookbacks_ns)).
        """
        def logit(p):
            p = np.clip(p, clip, 1.0 - clip)
            return np.log(p / (1.0 - p))

        n = len(market_ids)
        out = np.zeros((n, 1 + len(lookbacks_ns)), dtype=np.float32)
        cur = np.full(n, np.nan)
        past = np.full((n, len(lookbacks_ns)), np.nan)
        for j, m in enumerate(market_ids):
            ts = self._ts.get(m)
            if ts is None:
                continue
            i_cur = int(ts.searchsorted(t_ns, side="left"))
            if i_cur == 0:
                continue
            mid = self._mid[m]
            cur[j] = float(mid[i_cur - 1])
            for k, lb in enumerate(lookbacks_ns):
                t_past = t_ns - int(lb)
                if t_past < self.coverage_start_ns:
                    continue
                i_past = int(ts.searchsorted(t_past, side="left"))
                if i_past > 0:
                    past[j, k] = float(mid[i_past - 1])
        ok = ~np.isnan(cur)
        out[ok, 0] = logit(cur[ok]) / 5.0
        okp = ok[:, None] & ~np.isnan(past)
        delta = logit(np.where(ok, cur, 0.5))[:, None] - logit(np.where(okp, past, 0.5))
        out[:, 1:] = np.where(okp, delta, 0.0)
        return out

    def last_change(self, market_ids: Sequence[str], t_ns: int) -> np.ndarray:
        """Change between the last two snapshots strictly before t.  NaN if < 2 snapshots."""
        n = len(market_ids)
        out = np.full(n, np.nan, dtype=np.float32)
        for j, m in enumerate(market_ids):
            ts = self._ts.get(m)
            if ts is None:
                continue
            i = int(ts.searchsorted(t_ns, side="left"))
            if i < 2:
                continue
            mid = self._mid[m]
            out[j] = float(mid[i - 1]) - float(mid[i - 2])
        return out
