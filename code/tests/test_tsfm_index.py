"""
Unit tests for the foundation-model forecast cache (tsfm_cache.py) and its causal index
(TSFMIndex): the activity selection and the context of the cache, and the staleness /
relative-to-current-mid rules of the features.  Chronos is replaced by a fake pipeline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket_gnn.data.lob_layout import FEATURE_DIM, MID_IDX
from polymarket_gnn.data.tsfm_index import DAY_NS, HOUR_NS, MIN_NS, TSFMIndex, day_file, quantile_columns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tsfm_cache  # noqa: E402

DAY = pd.Timestamp("2026-03-13")
D0 = int(DAY.value)


def write_hour(base: Path, hour_start_ns: int, rows):
    ts = pd.Timestamp(hour_start_ns)
    path = base / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = []
    for _, _, mid in rows:
        f = np.zeros(FEATURE_DIM, dtype=np.float64)
        f[MID_IDX] = mid
        feats.append(list(f))
    pd.DataFrame({"market_id": [r[0] for r in rows], "timestamp": pd.to_datetime([r[1] for r in rows]),
                  "features": feats}).to_parquet(path, index=False)


def write_cache(root: Path, steps, rows):
    """rows: list of (market_id, t_grid_ns, {col: value})."""
    root.mkdir(parents=True, exist_ok=True)
    json.dump({"steps": steps, "quantiles": [10, 50, 90], "horizon": 3600, "step_min": 1, "grid_min": 1,
               "context": 4, "model": "fake"}, open(root / "meta.json", "w"))
    by_day = {}
    for m, t, q in rows:
        by_day.setdefault(pd.Timestamp(t).normalize(), []).append((m, t, q))
    for d, lst in by_day.items():
        df = pd.DataFrame({"market_id": [r[0] for r in lst], "t_grid": [r[1] for r in lst],
                           "mid_last": 0.0, "known": 4})
        for c in quantile_columns(steps):
            df[c] = [r[2][c] for r in lst]
        df.to_parquet(day_file(root, d), index=False)


def test_index_features_relative_to_current_mid_and_staleness(tmp_path):
    steps = [15, 60]
    q = {"q50_15": 0.52, "q10_15": 0.50, "q90_15": 0.55, "q50_60": 0.48, "q10_60": 0.40, "q90_60": 0.60}
    write_cache(tmp_path, steps, [("A", D0 + 10 * MIN_NS, q), ("A", D0 + 11 * MIN_NS, {k: 0.5 for k in q})])
    idx = TSFMIndex(tmp_path, max_stale_ns=3 * MIN_NS, scale=10.0)
    idx.set_hour(2026, 3, 13, 0)
    assert idx.n_features == 8
    # at 10:30 the 10:00 forecast is the latest; relative to the CURRENT mid 0.5
    f = idx.features(["A", "B"], D0 + 10 * MIN_NS + 30 * 10**9, np.array([0.5, 0.5]))
    np.testing.assert_allclose(f[0], [0.2, 0.0, 0.5, -0.2, -1.0, 1.0, -1.0, 1.0], atol=1e-5)
    assert (f[1] == 0).all()                                   # unknown market: all zeros
    # same forecast, different current mid: the expected move changes with it
    f = idx.features(["A"], D0 + 10 * MIN_NS + 30 * 10**9, np.array([0.48]))
    assert f[0, 3] == pytest.approx(0.0, abs=1e-5) and f[0, -2] == 0.0 and f[0, -1] == 1.0
    # the forecast made at 11:00 is not visible at 10:59 (causal) ...
    f = idx.features(["A"], D0 + 11 * MIN_NS - 1, np.array([0.5]))
    assert f[0, 0] == pytest.approx(0.2, abs=1e-5)
    # ... and at 11:00 it is (t_grid <= t)
    f = idx.features(["A"], D0 + 11 * MIN_NS, np.array([0.5]))
    assert (f[0, :-1] == 0).all() and f[0, -1] == 1.0
    # older than max_stale: unavailable; unknown current mid: unavailable
    assert (idx.features(["A"], D0 + 15 * MIN_NS, np.array([0.5])) == 0).all()
    assert (idx.features(["A"], D0 + 10 * MIN_NS + 1, np.array([np.nan])) == 0).all()


def test_index_crosses_midnight(tmp_path):
    steps = [60]
    q = {"q50_60": 0.6, "q10_60": 0.5, "q90_60": 0.7}
    write_cache(tmp_path, steps, [("A", D0 - MIN_NS, q), ("B", D0 + 5 * MIN_NS, q)])
    idx = TSFMIndex(tmp_path, max_stale_ns=3 * MIN_NS)
    idx.set_hour(2026, 3, 13, 0)                                   # previous day resident too
    f = idx.features(["A", "B"], D0 + MIN_NS, np.array([0.5, 0.5]))
    assert f[0, -1] == 1.0 and f[0, 0] == pytest.approx(1.0, abs=1e-5) and f[1, -1] == 0.0
    idx.set_hour(2026, 3, 13, 1)                                   # previous day dropped
    assert "A" not in idx._ts and "B" in idx._ts


class FakePipeline:
    """Forecast = last known value, with the known count encoded in the 90% quantile."""
    def __init__(self):
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size=None):
        import torch
        out = []
        for x in inputs:
            x = np.asarray(x, dtype=np.float32)
            self.calls.append(x.copy())
            last = x[~np.isnan(x)][-1]
            q = np.zeros((1, prediction_length, len(quantile_levels)), dtype=np.float32)
            q[..., 0] = last - 0.01
            q[..., 1] = last
            q[..., 2] = last + 0.001 * np.sum(~np.isnan(x))
            out.append(torch.from_numpy(q))
        return out, [q[..., 1] for q in out]


def test_cache_selection_and_context(tmp_path, monkeypatch):
    base = tmp_path / "derived"
    # A: snapshots at 00:00:30 (0.4), 00:01:40 (0.6), 00:02:10 (0.7); B: one snapshot at 00:01:20 (0.3); C: previous day only
    write_hour(base, D0 - HOUR_NS, [("C", D0 - 30 * MIN_NS, 0.9)])
    write_hour(base, D0, [("A", D0 + 30 * 10**9, 0.4), ("B", D0 + 80 * 10**9, 0.3), ("A", D0 + 100 * 10**9, 0.6),
                          ("A", D0 + 130 * 10**9, 0.7)])
    fake = FakePipeline()
    import types
    chronos = types.SimpleNamespace(Chronos2Pipeline=types.SimpleNamespace(from_pretrained=lambda *a, **k: fake))
    monkeypatch.setitem(sys.modules, "chronos", chronos)
    out = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["tsfm_cache.py", "--horizon", "180", "--step-min", "1", "--steps", "1,3",
                                      "--context", "4", "--days", "2026-03-13", "--base-dir", str(base),
                                      "--out-dir", str(out), "--device", "cpu", "--batch-size", "2"])
    tsfm_cache.main()
    df = pd.read_parquet(day_file(out, DAY)).sort_values(["market_id", "t_grid"]).reset_index(drop=True)
    # forecasts: A at 00:00 has no known mid → none; A at 00:01 and 00:02; B needs a known mid before the grid time:
    # none at 00:01 (first snapshot 00:01:20) → B gets none in this day; C never active
    assert list(zip(df["market_id"], (df["t_grid"] - D0) // MIN_NS)) == [("A", 1), ("A", 2)]
    assert df["mid_last"].tolist() == pytest.approx([0.4, 0.6])            # last mid strictly before 00:01 / 00:02
    assert df["known"].tolist() == [1, 2]
    assert df["q50_3"].tolist() == pytest.approx([0.4, 0.6]) and df["q10_1"].tolist() == pytest.approx([0.39, 0.59])
    assert df["q90_1"].tolist() == pytest.approx([0.401, 0.602])            # known count reaches the pipeline
    # contexts: 4 points ending at the grid time, NaN before the first snapshot
    assert np.isnan(fake.calls[0][:3]).all() and fake.calls[0][3] == pytest.approx(0.4)
    assert np.isnan(fake.calls[1][:2]).all() and fake.calls[1][2:].tolist() == pytest.approx([0.4, 0.6])
    meta = json.load(open(out / "meta.json"))
    assert meta["steps"] == [1, 3] and meta["columns"] == quantile_columns([1, 3])
    # the index reads the cache back
    idx = TSFMIndex(out, max_stale_ns=2 * MIN_NS)
    idx.set_hour(2026, 3, 13, 0)
    f = idx.features(["A"], D0 + 90 * 10**9, np.array([0.4]))
    assert f[0, -1] == 1.0 and f[0, 0] == pytest.approx(0.0, abs=1e-5) and f[0, 2] == pytest.approx(0.01, abs=1e-5)
