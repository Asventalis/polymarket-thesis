"""
Hand-made features of the tensor rows as model inputs: the causal index (HandFeatureIndex) and
the predictor extras (CausalDynamicGNN.predictor_extra, read by the sparse engine).
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from polymarket_gnn.data.hand_index import CLIP, DERIVED, HOUR_NS, RETURN_COLS, HandFeatureIndex, day_file
from tests.test_navis_ctan import D, L, SKIP, build, lookback_windows, step

STORED = ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "spread_ticks", "imb1", "since_trade_s"]
CATS = ["esports", "sports", "crypto", "weather", "politics", "economy_tech", "culture", "other"]
DAY12 = pd.Timestamp("2026-03-12"); DAY13 = pd.Timestamp("2026-03-13")
T12 = int(DAY12.value); T13 = int(DAY13.value)
START = T12 - 5 * 24 * HOUR_NS; END = T13 + 10 * 24 * HOUR_NS


def write_dir(root):
    numeric = [c for c in STORED if c != "since_trade_s"]
    stats = {c: {"loc": 0.0, "scale": 1.0, "finite": 1.0} for c in numeric + DERIVED}
    stats["spread_ticks"] = {"loc": 1.0, "scale": 2.0, "finite": 1.0}
    (root / "meta.json").write_text(json.dumps({"stored": STORED, "derived": DERIVED, "categories": CATS, "stats": stats,
                                                "never_moved_seconds": 14 * 86400.0, "never_traded_seconds": 30 * 86400.0}))
    pd.DataFrame({"market_id": ["A", "B"], "category": np.array([1, 7], dtype=np.int8), "is_yesno": np.array([1.0, 0.0], dtype=np.float32),
                  "start_ns": np.array([START, START], dtype=np.int64), "end_ns": np.array([END, END], dtype=np.int64),
                  "duration_h": np.array([(END - START) / 3.6e12] * 2, dtype=np.float32)}).to_parquet(root / "markets.parquet", index=False)

    def rows(day_ns, specs):
        recs = []
        for m, offset_s, mid, ret_1h, tlm_offset_s, since_trade in specs:
            ts = day_ns + int(offset_s * 1e9)
            r = {"market_id": m, "ts": ts, "t_last_move": (ts - int(tlm_offset_s * 1e9)) if tlm_offset_s is not None else None,
                 "mid": mid, "ret_15m": 0.0, "ret_1h": ret_1h, "ret_3h": np.nan, "ret_6h": -0.02, "ret_12h": 0.0,
                 "spread_ticks": 3.0, "imb1": 0.5, "since_trade_s": since_trade}
            recs.append(r)
        df = pd.DataFrame(recs)
        df["t_last_move"] = df["t_last_move"].astype("Int64")
        return df

    # 12 March: market A has a row at 23:50 (its last of the day); market B one at 12:00
    rows(T12, [("A", 23 * 3600 + 50 * 60, 0.40, 0.01, 600.0, 100.0), ("B", 12 * 3600, 0.90, -0.05, None, np.nan)]).to_parquet(day_file(root, DAY12), index=False)
    # 13 March: market A rows at 00:10 and 00:20; market B none
    rows(T13, [("A", 600, 0.41, 0.02, 0.0, 30.0), ("A", 1200, 0.43, 0.03, 0.0, 10.0)]).to_parquet(day_file(root, DAY13), index=False)


@pytest.fixture
def index(tmp_path):
    write_dir(tmp_path)
    return HandFeatureIndex(tmp_path)


def col(index, name):
    return index.names.index(name)


def test_layout_and_last_row_lookup(index):
    assert index.n_features == 8 + 5 + 2 + 5 + 1 + 8 + 1
    index.set_hour(2026, 3, 13, 0)
    t = T13 + 15 * 60 * 10**9                               # 00:15 on 13 March: A's last row is the 00:10 one
    f = index.features(["A", "B", "Z"], t)
    assert f.shape == (3, index.n_features)
    a, b, z = f
    assert a[col(index, "available")] == 1 and b[col(index, "available")] == 1 and z[col(index, "available")] == 0
    assert np.all(z == 0)
    assert a[col(index, "mid")] == pytest.approx(0.41)       # the 00:10 row, not the 00:20 one (future) nor yesterday's
    assert a[col(index, "spread_ticks")] == pytest.approx((3.0 - 1.0) / 2.0)
    assert a[col(index, "sgn_1h")] == 1 and a[col(index, "sgn_6h")] == -1 and a[col(index, "sgn_3h")] == 0   # NaN return → sign 0
    assert a[col(index, "ret_3h")] == 0                      # NaN → 0 after standardisation
    # B's last row is yesterday's (12:00 on 12 March): the previous day is resident
    assert b[col(index, "mid")] == pytest.approx(0.90) and b[col(index, "sgn_1h")] == -1
    # static fields
    assert a[col(index, "is_yesno")] == 1 and a[col(index, "cat_sports")] == 1 and a[col(index, "cat_other")] == 0
    assert b[col(index, "is_yesno")] == 0 and b[col(index, "cat_other")] == 1
    # before any row of A on 12 March: nothing
    index.set_hour(2026, 3, 12, 10)
    assert index.features(["A"], T12 + 10 * HOUR_NS)[0, col(index, "available")] == 0


def test_time_dependent_features_follow_the_query_time(index):
    index.set_hour(2026, 3, 13, 0)
    t1 = T13 + 15 * 60 * 10**9; t2 = t1 + 3600 * 10**9
    f1 = index.features(["A", "B"], t1); f2 = index.features(["A", "B"], t2)
    # A moved at its 00:10 row (tlm offset 0): since_move = t - 00:10
    assert f1[0, col(index, "log_since_move")] == pytest.approx(np.log1p(300.0))
    assert f2[0, col(index, "log_since_move")] == pytest.approx(np.log1p(3300.0))   # at 01:15 the last row is the 00:20 one, which moved
    # B never moved: the sentinel
    assert f1[1, col(index, "log_since_move")] == pytest.approx(min(np.log1p(14 * 86400.0), CLIP))   # clipped: loc 0 / scale 1 here
    # since the last trade: the row's value plus the time elapsed since the row
    assert f1[0, col(index, "log_since_trade")] == pytest.approx(np.log1p(30.0 + 300.0))
    assert f1[1, col(index, "log_since_trade")] == pytest.approx(min(np.log1p(30 * 86400.0), CLIP))   # no trade → sentinel, clipped
    # hours to the end / since the start move with t
    assert f1[0, col(index, "log_hours_to_end")] > f2[0, col(index, "log_hours_to_end")]
    assert f1[0, col(index, "log_hours_since_start")] < f2[0, col(index, "log_hours_since_start")]
    assert f1[0, col(index, "log_hours_to_end")] == pytest.approx(np.log1p((END - t1) / 3.6e12))
    # hour of the day
    h = (t1 % (24 * HOUR_NS)) / HOUR_NS
    assert f1[0, col(index, "hour_sin")] == pytest.approx(np.sin(2 * np.pi * h / 24), abs=1e-6)
    assert np.all(np.abs(f1) <= CLIP)


# ── predictor extras ─────────────────────────────────────────────────────────

def test_extra_ctx_selects_the_columns_before_the_lookback_block():
    E = 3                                                         # the test windows have few columns: SKIP + E must fit
    w, _ = lookback_windows(6, seed=3)
    active = torch.tensor([2, 5, 8, 13, 21, 34]); pred = torch.tensor([34, 5, 13]); rows = [5, 1, 3]
    model = build(skip_dim=SKIP, skip_extra=E, predictor_extra=E)
    got = model.extra_ctx(w, active, pred)
    torch.testing.assert_close(got, w[rows, -1, -(SKIP + E):-SKIP], rtol=0, atol=0)
    # without the direct path the extras are simply the last E columns
    model = build(predictor_extra=E)
    torch.testing.assert_close(model.extra_ctx(w, active, pred), w[rows, -1, -E:], rtol=0, atol=0)
    assert model.extra_ctx(w, active, torch.empty(0, dtype=torch.long)).shape == (0, E)
    with pytest.raises(ValueError):
        build(skip_dim=SKIP, skip_extra=E, predictor_extra=E + 1)


def test_predictor_extras_reach_the_output_and_nothing_else_changes():
    E = 3
    w, _ = lookback_windows(8, seed=1)
    active = torch.tensor([1, 4, 6, 7, 12, 18, 25, 33]); pred = torch.tensor([33, 1, 12, 6, 25])
    base = build(skip_dim=SKIP, skip_extra=E)
    xp = build(skip_dim=SKIP, skip_extra=E, predictor_extra=E)
    assert xp.predictor.net[0].in_features == base.predictor.net[0].in_features + E
    # same parameters everywhere, zero weight on the extra inputs → identical output
    sd = base.state_dict()
    with torch.no_grad():
        for k, v in xp.state_dict().items():
            if k == "predictor.net.0.weight":
                v.zero_(); v[:, :sd[k].size(1)] = sd[k]
            else:
                v.copy_(sd[k])
    torch.testing.assert_close(step(xp, w, active, pred), step(base, w, active, pred), rtol=0, atol=0)
    # a weight on the extras: the output moves, and it moves only through the extra columns
    with torch.no_grad():
        xp.predictor.net[0].weight[:, -E:] = 0.5
    out = step(xp, w, active, pred)
    assert not torch.allclose(out, step(base, w, active, pred))
    w2 = w.clone(); w2[:, -1, -(SKIP + E):-SKIP] += 1.0            # change the extras only in the last row
    out2 = step(xp, w2, active, pred)
    assert not torch.allclose(out2, out)
    w3 = w.clone(); w3[:, :-1, -(SKIP + E):-SKIP] += 1.0           # change them in the other rows only
    torch.testing.assert_close(xp.extra_ctx(w3, active, pred), xp.extra_ctx(w, active, pred), rtol=0, atol=0)   # the predictor reads the last row


def test_predictor_head_options_keep_the_default_identical():
    from polymarket_gnn.models.causal_gnn import Predictor
    torch.manual_seed(0); a = Predictor(16, 0.0, 3)
    torch.manual_seed(0); b = Predictor(16, 0.0, 3, head_dim=None, head_layers=1)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb; torch.testing.assert_close(va, vb, rtol=0, atol=0)
    assert [m.out_features for m in a.net if isinstance(m, torch.nn.Linear)] == [8, 3]
    c = Predictor(16, 0.0, 3, in_dim=40, head_dim=64, head_layers=2)
    assert [(m.in_features, m.out_features) for m in c.net if isinstance(m, torch.nn.Linear)] == [(40, 64), (64, 64), (64, 3)]
    assert c(torch.randn(5, 40)).shape == (5, 3)
    model = build(skip_dim=SKIP, skip_extra=2, predictor_extra=2, head_dim=32, head_layers=2)
    w, _ = lookback_windows(4, seed=2); active = torch.tensor([1, 2, 3, 4]); pred = torch.tensor([2, 4])
    assert step(model, w, active, pred).shape == (2, 3)
