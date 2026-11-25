"""
FastLOBWindowBuffer must return exactly what LOBWindowBuffer returns:
same active markets and bit-identical windows, including markets with
repeated timestamps, zero padding, lifecycle events in the middle of an hour
and the reload of the next hour.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket_gnn.data.lob_layout import FEATURE_DIM, MID_IDX
from polymarket_gnn.data.lob_window_buffer import FastLOBWindowBuffer, LOBWindowBuffer, OUTPUT_DIM

HOUR = 3_600_000_000_000
SEC = 1_000_000_000
T0 = pd.Timestamp("2026-03-13 05:00:00").value


def write_hour(path: Path, rng, markets, hour_start, heavy_ties: bool):
    rows = []
    for m in markets:                                   # file is grouped by market, chronological inside
        n = int(rng.integers(0, 60))
        if n == 0:
            continue
        ts = np.sort(rng.integers(0, 3600_000, size=n)) * 1_000_000 + hour_start     # ms resolution
        if heavy_ties and n > 20:
            ts[5:25] = ts[5]                            # a long run of identical timestamps (> 16 rows)
            ts = np.sort(ts)
        elif n > 3:
            ts[2] = ts[1]                               # a single tie
        for t in ts:
            f = rng.normal(size=FEATURE_DIM) * 50
            f[MID_IDX] = rng.uniform(0.01, 0.99)
            rows.append((m, t, list(f)))
    df = pd.DataFrame({"market_id": [r[0] for r in rows],
                       "timestamp": pd.to_datetime([r[1] for r in rows]),
                       "features": [r[2] for r in rows]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    markets = [f"m{i:02d}" for i in range(40)]
    h0 = write_hour(tmp_path / "h0.parquet", rng, markets, T0 - HOUR, heavy_ties=True)
    h1 = write_hour(tmp_path / "h1.parquet", rng, markets, T0, heavy_ties=True)
    h2 = write_hour(tmp_path / "h2.parquet", rng, markets, T0 + HOUR, heavy_ties=False)
    events = []
    for i, m in enumerate(markets):
        if i == 0:
            continue                                                   # never opened
        if i % 9 == 1:
            events.append((T0 + int(rng.integers(60, 3000)) * SEC, m, "OPEN"))      # opens during the hour
        else:
            events.append((T0 - 10 * HOUR + i, m, "OPEN"))
        if i % 7 == 2:
            events.append((T0 + int(rng.integers(600, 5000)) * SEC, m, "CLOSE"))    # closes during the run
    events.append((T0 - 5 * HOUR, "no_data_market", "OPEN"))
    events.append((T0 + 100 * SEC, "m05", "CLOSE"))
    events.append((T0 + 100 * SEC, "m05", "OPEN"))                                  # same-timestamp pair
    lc = tmp_path / "lifecycle.parquet"
    pd.DataFrame({"event_timestamp": pd.to_datetime([e[0] for e in events]),
                  "market_id": [e[1] for e in events],
                  "event_type": [e[2] for e in events]}).to_parquet(lc, index=False)
    return h0, h1, h2, lc


def assert_step_equal(ref, fast, t0, t1):
    a_ref, w_ref = ref.load_step(t0, t1)
    a_fast, w_fast = fast.load_step(t0, t1)
    assert sorted(a_ref) == sorted(a_fast)
    assert w_ref.shape == w_fast.shape and w_fast.dtype == np.float32
    if a_ref:
        o_ref, o_fast = np.argsort(a_ref), np.argsort(a_fast)
        assert np.array_equal(w_ref[o_ref], w_fast[o_fast])            # bit-identical
        assert a_fast == sorted(a_fast, key=fast._code_of.get)         # deterministic order
    return len(a_ref)


@pytest.mark.parametrize("K", [3, 10])
def test_fast_buffer_matches_reference(data, K):
    h0, h1, h2, lc = data
    ref = LOBWindowBuffer.from_parquet([h0, h1], lc, K=K)
    fast = FastLOBWindowBuffer.from_parquet([h0, h1], lc, K=K)
    ref.reset_to_time(T0); fast.reset_to_time(T0)
    assert ref._open_markets == fast._open_markets

    rng = np.random.default_rng(1)
    total, t = 0, T0
    while t < T0 + HOUR:                                   # blocks of ~2-40 s over the first hour
        t_next = t + int(rng.integers(1, 20)) * 2 * SEC
        total += assert_step_equal(ref, fast, t, t_next)
        t = t_next
    assert ref._open_markets == fast._open_markets

    ref.reload_lob_data([h1, h2]); fast.reload_lob_data([h1, h2])      # next hour: [h, h+1]
    assert ref._open_with_data == {m for m, c in fast._code_of.items() if fast._open_mask[c]}
    while t < T0 + 2 * HOUR:
        t_next = t + int(rng.integers(1, 20)) * 2 * SEC
        total += assert_step_equal(ref, fast, t, t_next)
        t = t_next
    assert total > 200                                     # the comparison was not vacuous

    # single-window queries, including unknown markets and times before any data
    for m in ["m03", "m11", "m27", "unknown"]:
        for tq in [T0 + HOUR + 5 * SEC, T0 + HOUR + 1800 * SEC, T0 - 2 * HOUR]:
            assert np.array_equal(ref.get_window(m, tq), fast.get_window(m, tq))


def test_fast_buffer_empty_interval_and_shapes(data):
    h0, h1, _, lc = data
    fast = FastLOBWindowBuffer.from_parquet(h1, lc, K=5)               # single path, like the reference API
    fast.reset_to_time(T0)
    ids, w = fast.load_step(T0 - 10 * SEC, T0 - 9 * SEC)               # nothing there
    assert ids == [] and w.shape == (0, 5, OUTPUT_DIM)
    ids, w = fast.load_step(T0, T0 + 600 * SEC)
    assert len(ids) == w.shape[0] > 0 and w.shape[1:] == (5, OUTPUT_DIM)
