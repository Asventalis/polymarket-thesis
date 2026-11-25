"""Cross-fitted tree log-probabilities as model inputs and logit offset: the causal index (TreeIndex)."""
import json

import numpy as np
import pandas as pd
import pytest

from polymarket_gnn.data.tree_index import HOUR_NS, LP_COLS, TreeIndex, day_file

DAY12 = pd.Timestamp("2026-03-12"); DAY13 = pd.Timestamp("2026-03-13")
T12 = int(DAY12.value); T13 = int(DAY13.value)
LP_A_LATE12 = np.log([0.2, 0.5, 0.3]); LP_A_0010 = np.log([0.6, 0.3, 0.1]); LP_A_0020 = np.log([0.1, 0.1, 0.8]); LP_B = np.log([0.3, 0.4, 0.3])


def write_dir(root):
    (root / "meta.json").write_text(json.dumps({"columns": LP_COLS, "horizon_s": 3600}))

    def rows(specs):
        return pd.DataFrame({"market_id": [m for m, _, _ in specs], "ts": np.array([t for _, t, _ in specs], dtype=np.int64),
                             "lp_down": np.array([lp[0] for _, _, lp in specs], dtype=np.float32),
                             "lp_const": np.array([lp[1] for _, _, lp in specs], dtype=np.float32),
                             "lp_up": np.array([lp[2] for _, _, lp in specs], dtype=np.float32)})

    # 12 March: market A at 23:50 (its last of the day), market B at 12:00; 13 March: A at 00:10 and 00:20
    rows([("A", T12 + (23 * 3600 + 50 * 60) * 10**9, LP_A_LATE12), ("B", T12 + 12 * 3600 * 10**9, LP_B)]).to_parquet(day_file(root, DAY12), index=False)
    rows([("A", T13 + 600 * 10**9, LP_A_0010), ("A", T13 + 1200 * 10**9, LP_A_0020)]).to_parquet(day_file(root, DAY13), index=False)


@pytest.fixture
def index(tmp_path):
    write_dir(tmp_path)
    return TreeIndex(tmp_path)


def test_last_row_lookup_is_causal_and_crosses_midnight(index):
    index.set_hour(2026, 3, 13, 0)
    t = T13 + 5 * 60 * 10**9                       # 00:05 of 13 March: A's last row is yesterday's 23:50, B's is 12:00 of yesterday
    lp = index.log_probs(["A", "B", "C"], t)
    np.testing.assert_allclose(lp[0], LP_A_LATE12, rtol=1e-6)
    np.testing.assert_allclose(lp[1], LP_B, rtol=1e-6)
    assert np.isnan(lp[2]).all()                    # C has no rows
    t = T13 + 15 * 60 * 10**9                      # 00:15: A's 00:10 row, not the 00:20 one
    np.testing.assert_allclose(index.log_probs(["A"], t)[0], LP_A_0010, rtol=1e-6)
    t = T13 + 20 * 60 * 10**9                      # 00:20 exactly: ts <= t, the 00:20 row is served
    np.testing.assert_allclose(index.log_probs(["A"], t)[0], LP_A_0020, rtol=1e-6)
    assert index.n_queried == 5 and index.n_available == 4


def test_features_and_offset_are_zero_without_a_row(index):
    index.set_hour(2026, 3, 13, 1)
    t = T13 + HOUR_NS + 1
    f = index.features(["C", "A"], t)
    assert f.shape == (2, 4) and index.names == ["lp_down", "lp_const", "lp_up", "tree_available"]
    assert (f[0] == 0).all()
    np.testing.assert_allclose(f[1, :3], LP_A_0020, rtol=1e-6); assert f[1, 3] == 1.0
    off = index.offset(["C", "A"], t)
    assert off.shape == (2, 3) and (off[0] == 0).all()
    np.testing.assert_allclose(off[1], LP_A_0020, rtol=1e-6)
    assert np.isclose(np.exp(off[1]).sum(), 1.0, atol=1e-6)     # the offset is a normalised log-distribution


def test_residency_follows_the_hour(index):
    index.set_hour(2026, 3, 12, 13)
    assert sorted(d.day for d in index._day_cache) == [12] and 11 in {d.day for d in index._missing_days}
    index.set_hour(2026, 3, 13, 0)
    assert sorted(d.day for d in index._day_cache) == [12, 13]
    index.set_hour(2026, 3, 14, 0)                 # 13 resident, 14 missing; the 12 March rows are gone
    assert sorted(d.day for d in index._day_cache) == [13]
    assert np.isnan(index.log_probs(["B"], T13 + 24 * HOUR_NS + 1)).all()
