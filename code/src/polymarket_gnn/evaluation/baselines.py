"""
baselines.py

Pure helpers for the causal baselines (no I/O), so they can be unit-tested.

All baselines map an observed *past* quantity to a class in {0=DOWN, 1=CONSTANT, 2=UP}.
NaN inputs (price not observable) fall back to CONSTANT.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

DOWN, CONST, UP = 0, 1, 2


def direction_from_change(delta: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Vectorised direction_label: UP if delta > eps, DOWN if delta < -eps, else CONSTANT (NaN → CONSTANT)."""
    delta = np.asarray(delta, dtype=np.float64)
    out = np.full(delta.shape, CONST, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        out[delta > eps] = UP
        out[delta < -eps] = DOWN
    return out


def momentum_from_change(delta: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Momentum: predict that the observed past move continues."""
    return direction_from_change(delta, eps)


def mean_reversion_from_change(delta: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Mean reversion: predict the opposite of the observed past move (CONSTANT stays CONSTANT)."""
    return 2 - direction_from_change(delta, eps)


class DedupTracker:
    """
    Keeps at most one evaluation sample per market every ``min_gap_ns``.

    Consecutive blocks (≈2 s apart) of the same market share almost the same
    target when the horizon is hours, so the ~10M "samples" of a validation
    day are far from independent.  Metrics on the deduplicated subset are a
    sanity check that headline numbers are not driven by that redundancy.
    """

    def __init__(self, min_gap_ns: int):
        self.min_gap_ns = int(min_gap_ns)
        self._last: Dict[str, int] = {}

    def keep_mask(self, market_ids, t_ns: int) -> np.ndarray:
        mask = np.zeros(len(market_ids), dtype=bool)
        for i, m in enumerate(market_ids):
            last = self._last.get(m)
            if last is None or t_ns - last >= self.min_gap_ns:
                mask[i] = True
                self._last[m] = t_ns
        return mask


def parse_days(spec: str):
    """'13' → [13]; '21-23' → [21, 22, 23]; '6,7,8' → [6, 7, 8]."""
    days = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            days.extend(range(int(a), int(b) + 1))
        else:
            days.append(int(part))
    if not days:
        raise ValueError(f"No days in spec {spec!r}")
    return days
