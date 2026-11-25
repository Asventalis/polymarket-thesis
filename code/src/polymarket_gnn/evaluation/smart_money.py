"""
smart_money.py

Pure helpers of the "smart money" test (no I/O), so they can be unit-tested.

The test asks whether the IDENTITY of the wallets that trade a market carries
information about the future mid price, i.e. the only thing the transaction
graph adds to the order book.  A trade is a bet on a direction

    d = +1  buy YES / sell NO      (bet that the market mid goes up)
    d = -1  sell YES / buy NO      (bet that it goes down)

and its outcome at horizon H is r = d * (mid(t + H) - mid(t)).  A wallet's
skill is a shrunk signed hit rate of its past, already realised, outcomes.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

DOWN, CONST, UP = 0, 1, 2


def shrunk_skill(wins, losses, shrink: float = 10.0) -> np.ndarray:
    """
    (wins - losses) / (wins + losses + shrink), in (-1, 1).

    0 for a wallet with no decided trades; a wallet needs many more wins than
    losses to get close to 1 (2 wins out of 2 → 0.17 with shrink 10, 80 out of
    100 → 0.55).  Trades with outcome exactly 0 count for neither side.
    """
    wins = np.asarray(wins, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    return (wins - losses) / (wins + losses + float(shrink))


def skill_buckets(skill: np.ndarray, eligible: np.ndarray, n_buckets: int = 10) -> np.ndarray:
    """
    Quantile bucket (0 = lowest skill … n_buckets-1 = highest) among the
    eligible wallets; -1 for the others.  Ties are broken by position, so the
    buckets have equal size.
    """
    skill = np.asarray(skill, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    out = np.full(len(skill), -1, dtype=np.int64)
    idx = np.flatnonzero(eligible)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(skill[idx], kind="stable")]
    out[order] = (np.arange(len(order)) * n_buckets) // len(order)
    return out


def cumulative(ts: np.ndarray, contrib: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts one market's trades by time and returns (ts_sorted, cum) with
    cum[i] = sum of the first i rows of ``contrib`` ((n, V) → (n + 1, V)).
    """
    order = np.argsort(ts, kind="stable")
    ts = np.asarray(ts, dtype=np.int64)[order]
    c = np.asarray(contrib, dtype=np.float64)[order]
    cum = np.zeros((len(ts) + 1, c.shape[1]), dtype=np.float64)
    np.cumsum(c, axis=0, out=cum[1:])
    return ts, cum


def window_sums(ts: np.ndarray, cum: np.ndarray, t_ns: int, windows_ns: Sequence[int]) -> np.ndarray:
    """
    Sum of the contributions of the trades with timestamp in (t - W, t], for
    every window W: array (len(windows_ns), V).  Trades after t are never read.
    """
    hi = int(ts.searchsorted(t_ns, side="right"))
    out = np.empty((len(windows_ns), cum.shape[1]), dtype=np.float64)
    for k, w in enumerate(windows_ns):
        lo = int(ts.searchsorted(t_ns - int(w), side="right"))
        out[k] = cum[hi] - cum[lo]
    return out


def direction_from_signal(signal: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """UP if signal > eps, DOWN if signal < -eps, otherwise CONSTANT (no signal)."""
    signal = np.asarray(signal, dtype=np.float64)
    out = np.full(signal.shape, CONST, dtype=np.int64)
    out[signal > eps] = UP
    out[signal < -eps] = DOWN
    return out


def cluster_bootstrap_gap(wins_a, losses_a, wins_b, losses_b, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Difference of hit rates  wins_a/(wins_a+losses_a) - wins_b/(wins_b+losses_b)
    with a percentile 95% interval obtained by resampling the clusters (one row
    per cluster, e.g. per market).  Returns (gap, lo, hi).
    """
    wa, la, wb, lb = (np.asarray(x, dtype=np.float64) for x in (wins_a, losses_a, wins_b, losses_b))

    def gap(i):
        a, b = wa[i].sum() + la[i].sum(), wb[i].sum() + lb[i].sum()
        if a == 0 or b == 0:
            return np.nan
        return wa[i].sum() / a - wb[i].sum() / b

    n = len(wa)
    point = gap(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([gap(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)
