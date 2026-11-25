"""
Unit tests for the causal mid-price index, the fixed LOB window momentum
features and the model's predict_idx path.  All data is synthetic and uses the
real 52-column tensor layout (mid price at column 40, column 0 = relative
best-bid deviation) so that a regression to "column 0 is the price" fails.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from polymarket_gnn.data.lob_layout import FEATURE_DIM, MID_IDX
from polymarket_gnn.data.lob_window_buffer import LOBWindowBuffer
from polymarket_gnn.data.price_index import (
    CONST, DOWN, HOUR_NS, UP, LifecycleTracker, MidPriceIndex, hour_key, hour_key_to_path,
)

K0 = hour_key(2026, 3, 13, 5)          # base hour
MIN = 60 * 10**9                       # one minute in ns
H1 = HOUR_NS


def t_at(hours: int, minutes: int) -> int:
    return (K0 + hours) * HOUR_NS + minutes * MIN


def write_hour(base: Path, key: int, rows):
    """rows: list of (market_id, ts_ns, mid, col0)."""
    path = hour_key_to_path(base, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = []
    for _, _, mid, col0 in rows:
        f = np.zeros(FEATURE_DIM, dtype=np.float64)
        f[MID_IDX] = mid
        f[0] = col0
        feats.append(list(f))
    df = pd.DataFrame({
        "market_id": [r[0] for r in rows],
        "timestamp": pd.to_datetime([r[1] for r in rows]),
        "features": feats,
    })
    df.to_parquet(path, index=False)
    return path


def write_lifecycle(base: Path, events):
    """events: list of (ts_ns, market_id, 'OPEN'|'CLOSE')."""
    path = base / "market_lifecycle_events.parquet"
    pd.DataFrame({
        "event_timestamp": pd.to_datetime([e[0] for e in events]),
        "market_id": [e[1] for e in events],
        "event_type": [e[2] for e in events],
    }).to_parquet(path, index=False)
    return path


@pytest.fixture
def world(tmp_path):
    base = tmp_path / "derived"
    # hour K0-1: market D history
    write_hour(base, K0 - 1, [("D", t_at(-1, 10), 0.3, 9.0)])
    # hour K0: A rises (col0 falls), B at 0.5, D at 0.5 then 0.9, E has 3 snapshots
    write_hour(base, K0, [
        ("A", t_at(0, 10), 0.4, +0.5),
        ("B", t_at(0, 5), 0.5, 0.0),
        ("D", t_at(0, 5), 0.5, 9.0),
        ("D", t_at(0, 40), 0.9, 9.0),
        ("E", t_at(0, 1), 0.4, 9.0),
        ("E", t_at(0, 2), 0.5, 9.0),
        ("E", t_at(0, 3), 0.7, 9.0),
    ])
    # hour K0+1: A at 0.6 (col0 lower than before → old code says DOWN), B moves to 0.7
    write_hour(base, K0 + 1, [
        ("A", t_at(1, 10), 0.6, -0.5),
        ("B", t_at(1, 5), 0.7, 0.0),
    ])
    # hours K0+2, K0+3 exist but B is quiet (only a filler market)
    write_hour(base, K0 + 2, [("Z", t_at(2, 1), 0.5, 0.0)])
    write_hour(base, K0 + 3, [("Z", t_at(3, 1), 0.5, 0.0)])
    lifecycle = write_lifecycle(base, [
        (t_at(-48, 0), "A", "OPEN"),
        (t_at(-48, 0), "B", "OPEN"),
        (t_at(-48, 0), "D", "OPEN"),
        (t_at(-48, 0), "E", "OPEN"),
        (t_at(0, 30), "A", "CLOSE"),      # A's last trade at K0+30min
    ])
    return base, lifecycle


def test_labels_use_true_mid_column(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_ahead=1, lifecycle=LifecycleTracker(lc), close_filter="none")
    idx.set_hour(2026, 3, 13, 5)
    y = idx.labels(["A"], t_at(0, 20), H1)
    assert y[0] == UP                      # mid 0.4 → 0.6 ; column 0 went +0.5 → -0.5 (would be DOWN)
    # the (deprecated) buffer path now reads the same column (B: 0.5 → 0.7, no CLOSE event)
    buf = LOBWindowBuffer.from_parquet([hour_key_to_path(base, K0), hour_key_to_path(base, K0 + 1)], lc, K=3)
    assert buf.get_future_targets(["B"], t_at(0, 20), H1).item() == UP
    # ...and still applies its own last-trade filter (A closes at 30min < t+H)
    assert np.isnan(buf.get_future_targets(["A"], t_at(0, 20), H1).item())


def test_intermediate_hour_moves_are_seen(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_ahead=3, lifecycle=LifecycleTracker(lc), close_filter="none")
    idx.set_hour(2026, 3, 13, 5)
    # B: 0.5 at K0+5min, 0.7 at K0+1h05, nothing afterwards → price at t+3h is 0.7
    assert idx.labels(["B"], t_at(0, 10), 3 * H1)[0] == UP
    # The old approach loaded only hours h and h+H and therefore saw CONSTANT:
    buf = LOBWindowBuffer.from_parquet([hour_key_to_path(base, K0), hour_key_to_path(base, K0 + 3)], lc, K=3)
    assert buf.get_future_targets(["B"], t_at(0, 10), 3 * H1).item() == CONST


def test_nan_beyond_coverage(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_ahead=6, lifecycle=LifecycleTracker(lc), close_filter="none")
    idx.set_hour(2026, 3, 13, 5)           # hours K0..K0+3 exist, K0+4.. do not
    assert idx.coverage_end_ns == (K0 + 4) * HOUR_NS
    assert np.isnan(idx.labels(["B"], t_at(0, 10), 6 * H1)[0])
    assert idx.labels(["B"], t_at(0, 10), 3 * H1)[0] == UP
    # market unknown / no snapshot before t → NaN
    assert np.isnan(idx.labels(["NOPE"], t_at(0, 10), H1)[0])
    assert np.isnan(idx.labels(["A"], t_at(0, 5), H1)[0])      # A's first snapshot is at 10min


def test_close_filter(world):
    base, lc = world
    strict = MidPriceIndex(base, hours_ahead=1, lifecycle=LifecycleTracker(lc), close_filter="last_tx")
    strict.set_hour(2026, 3, 13, 5)
    assert np.isnan(strict.labels(["A"], t_at(0, 20), H1)[0])   # last trade (30min) < t+H
    loose = MidPriceIndex(base, hours_ahead=1, lifecycle=LifecycleTracker(lc), close_filter="none")
    loose.set_hour(2026, 3, 13, 5)
    assert loose.labels(["A"], t_at(0, 20), H1)[0] == UP


def test_past_and_last_change_are_causal(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_back=1, hours_ahead=1, lifecycle=LifecycleTracker(lc))
    idx.set_hour(2026, 3, 13, 5)
    # D: 0.3 @ -50min, 0.5 @ +5min, 0.9 @ +40min
    assert idx.past_change(["D"], t_at(0, 20), H1)[0] == pytest.approx(0.5 - 0.3, abs=1e-6)
    assert idx.last_change(["D"], t_at(0, 20))[0] == pytest.approx(0.5 - 0.3, abs=1e-6)
    assert idx.last_change(["D"], t_at(0, 50))[0] == pytest.approx(0.9 - 0.5, abs=1e-6)
    # lookback reaching before the loaded coverage → unknown
    assert np.isnan(idx.past_change(["D"], t_at(0, 20), 5 * H1)[0])
    # fewer than two snapshots → unknown
    assert np.isnan(idx.last_change(["A"], t_at(0, 20))[0])


def test_const_eps(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_ahead=1, lifecycle=LifecycleTracker(lc), close_filter="none", const_eps=0.25)
    idx.set_hour(2026, 3, 13, 5)
    assert idx.labels(["A"], t_at(0, 20), H1)[0] == CONST       # Δ = 0.2 <= 0.25
    idx2 = MidPriceIndex(base, hours_ahead=1, lifecycle=LifecycleTracker(lc), close_filter="none", const_eps=0.1)
    idx2.set_hour(2026, 3, 13, 5)
    assert idx2.labels(["A"], t_at(0, 20), H1)[0] == UP


def test_active_between_respects_lifecycle(world):
    base, lc = world
    tracker = LifecycleTracker(lc)
    idx = MidPriceIndex(base, hours_ahead=1, lifecycle=tracker)
    idx.set_hour(2026, 3, 13, 5)
    tracker.reset_to(t_at(0, 0))
    assert set(idx.active_between(t_at(0, 0), t_at(0, 6))) == {"B", "D", "E"}       # A's snapshot is at 10min
    tracker.advance(t_at(0, 35))                                                    # A closed at 30min
    assert "A" not in idx.active_between(t_at(0, 0), t_at(0, 60))
    assert "A" in idx.active_between(t_at(0, 0), t_at(0, 60), require_open=False)
    assert idx.active_between(t_at(0, 50), t_at(0, 55)) == []


def test_window_momentum_uses_mid_and_ignores_padding(world):
    base, lc = world
    buf = LOBWindowBuffer.from_parquet([hour_key_to_path(base, K0)], lc, K=5)
    w = buf.get_window("E", t_at(0, 10))               # E: 0.4, 0.5, 0.7 (3 real rows, 2 padded)
    assert w.shape == (5, FEATURE_DIM + 3)
    np.testing.assert_allclose(w[:, FEATURE_DIM], [0, 0, 0, 0.1, 0.2], atol=1e-6)       # price_change_1
    np.testing.assert_allclose(w[:, FEATURE_DIM + 1], [0, 0, 0, 0.1, 0.3], atol=1e-6)   # price_change_k
    np.testing.assert_allclose(w[:, FEATURE_DIM + 2], [0, 0, 0, 1, 1])                  # sign
    assert (w[:2, :FEATURE_DIM] == 0).all()
    assert np.isclose(w[2, MID_IDX], 0.4) and np.isclose(w[2, 0], 9.0)


def test_lifecycle_tracker_close_times(world):
    _, lc = world
    tr = LifecycleTracker(lc)
    assert tr.close_time_ns["A"] == t_at(0, 30)
    tr.reset_to(t_at(0, 0))
    assert tr.open_markets == {"A", "B", "D", "E"}
    tr.advance(t_at(0, 30))
    assert "A" not in tr.open_markets


def test_model_predict_idx_matches_full_logits():
    from polymarket_gnn.data.graph_builder import EDGE_DIM
    from polymarket_gnn.models.causal_gnn import CausalDynamicGNN
    torch.manual_seed(0)
    n_m, n_w, hid, K = 20, 15, 16, 4
    model = CausalDynamicGNN(lob_dim=FEATURE_DIM + 3, hidden_dim=hid, edge_dim=EDGE_DIM).eval()
    h_m, h_w = torch.randn(n_m, hid), torch.randn(n_w, hid)
    active = torch.tensor([3, 7, 11])
    windows = torch.randn(3, K, FEATURE_DIM + 3)
    ei = torch.tensor([[0, 2, 5], [3, 7, 7]])
    ef = torch.randn(3, EDGE_DIM)
    dt = torch.tensor(2.0)
    with torch.no_grad():
        full, hm1, hw1 = model(windows, active, h_m, h_w, ei, ef, dt, dt)
        sub, hm2, hw2 = model(windows, active, h_m, h_w, ei, ef, dt, dt, predict_idx=active[[0, 2]])
    assert full.shape == (n_m, 3) and sub.shape == (2, 3)
    torch.testing.assert_close(sub, full[active[[0, 2]]])
    torch.testing.assert_close(hm1, hm2)
    torch.testing.assert_close(hw1, hw2)


def test_lookback_features(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_back=1, hours_ahead=1, lifecycle=LifecycleTracker(lc))
    idx.set_hour(2026, 3, 13, 5)
    # D: 0.3 @ -50min, 0.5 @ +5min → Δ1h at +20min = 0.2 ; 5h lookback is before coverage
    # A: first snapshot at +10min → no price one hour earlier
    f = idx.lookback_features(["D", "A"], t_at(0, 20), [H1, 5 * H1], scale=10.0)
    assert f.shape == (2, 6) and f.dtype == np.float32
    np.testing.assert_allclose(f[0], [2.0, 1.0, 1.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(f[1], np.zeros(6))
    # never looks at the future: the 0.9 print at +40min must not affect a query at +20min
    assert f[0, 0] == pytest.approx(10 * (0.5 - 0.3), abs=1e-5)


def test_signed_log1p_columns():
    from polymarket_gnn.data.lob_layout import (
        BID_SIZE_START, LOG_SCALE_COLUMNS, OFI_IDX, SPREAD_IDX, signed_log1p_columns,
    )
    w = np.zeros((2, 3, FEATURE_DIM + 3), dtype=np.float32)
    w[0, 2, BID_SIZE_START] = 1e6
    w[0, 2, OFI_IDX] = -5000.0
    w[0, 2, MID_IDX] = 0.37
    w[0, 2, SPREAD_IDX] = 0.02
    w[0, 2, FEATURE_DIM] = 0.1                       # appended momentum feature
    out = signed_log1p_columns(w)
    assert out is not w and w[0, 2, BID_SIZE_START] == 1e6          # input untouched
    assert out[0, 2, BID_SIZE_START] == pytest.approx(np.log1p(1e6), rel=1e-5)
    assert out[0, 2, OFI_IDX] == pytest.approx(-np.log1p(5000.0), rel=1e-5)
    assert out[0, 2, MID_IDX] == pytest.approx(0.37) and out[0, 2, SPREAD_IDX] == pytest.approx(0.02)
    assert out[0, 2, FEATURE_DIM] == pytest.approx(0.1)
    assert (out[1] == 0).all() and (out[0, :2] == 0).all()           # padding stays zero
    assert MID_IDX not in LOG_SCALE_COLUMNS and len(set(LOG_SCALE_COLUMNS)) == len(LOG_SCALE_COLUMNS) == 34
    assert np.abs(out).max() < 100                                   # nothing left for the clamp to saturate


def test_future_change_is_consistent_with_labels(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_ahead=3, lifecycle=LifecycleTracker(lc), close_filter="none")
    idx.set_hour(2026, 3, 13, 5)
    d = idx.future_change(["A", "B", "NOPE"], t_at(0, 20), H1)
    assert d.dtype == np.float64
    assert d[0] == pytest.approx(0.6 - 0.4, abs=1e-6)           # A: 0.4 → 0.6
    assert d[1] == pytest.approx(0.7 - 0.5, abs=1e-6)           # B: 0.5 → 0.7 (at +1h05 < t+H)
    assert np.isnan(d[2])
    y = idx.labels(["A", "B", "NOPE"], t_at(0, 20), H1)
    assert y[0] == UP and y[1] == UP and np.isnan(y[2])
    # same NaN pattern beyond coverage
    assert np.isnan(idx.future_change(["B"], t_at(0, 10), 6 * H1)[0])


def test_logodds_features(world):
    base, lc = world
    idx = MidPriceIndex(base, hours_back=1, hours_ahead=1, lifecycle=LifecycleTracker(lc))
    idx.set_hour(2026, 3, 13, 5)
    lg = lambda p: np.log(p / (1 - p))
    f = idx.logodds_features(["D", "A", "nope"], t_at(0, 20), [H1, 5 * H1])
    assert f.shape == (3, 3) and f.dtype == np.float32
    # D: 0.3 one hour earlier, 0.5 now; the 5h lookback is before the coverage; the 0.9 print at +40min is in the future
    np.testing.assert_allclose(f[0], [lg(0.5) / 5, lg(0.5) - lg(0.3), 0.0], atol=1e-5)
    # A: has a price now (0.4) but none one hour earlier → level only
    np.testing.assert_allclose(f[1], [lg(0.4) / 5, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(f[2], np.zeros(3))
    # same price change, much larger in log-odds near the boundary; clipping keeps it finite
    assert lg(0.02) - lg(0.01) > 15 * (lg(0.51) - lg(0.50))
    assert np.isfinite(idx.logodds_features(["D"], t_at(0, 50), [H1])).all()
