"""GDELT news-volume features as model inputs: the causal index (NewsIndex)."""
import json

import numpy as np
import pandas as pd
import pytest

from polymarket_gnn.data.news_index import NewsIndex, day_file

FEATS = ["news_log_n_1h", "news_burst_1h", "news_has_terms"]
DAY12 = pd.Timestamp("2026-03-12"); DAY13 = pd.Timestamp("2026-03-13")
T12 = int(DAY12.value); T13 = int(DAY13.value)
STATS = {"news_log_n_1h": {"loc": 1.0, "scale": 2.0}, "news_burst_1h": {"loc": 0.0, "scale": 0.5}, "news_has_terms": {"loc": 0.0, "scale": 1.0}}
A_LATE12 = [3.0, 1.0, 1.0]; A_0010 = [5.0, -1.0, 1.0]; A_0020 = [1.0, 0.0, 1.0]; B = [0.0, 100.0, 0.0]


def write_dir(root):
    (root / "meta.json").write_text(json.dumps({"features": FEATS, "stats": STATS}))

    def rows(specs):
        df = pd.DataFrame({"market_id": [m for m, _, _ in specs], "ts": np.array([t for _, t, _ in specs], dtype=np.int64)})
        for i, c in enumerate(FEATS):
            df[c] = np.array([x[i] for _, _, x in specs], dtype=np.float32)
        return df

    rows([("A", T12 + (23 * 3600 + 50 * 60) * 10**9, A_LATE12), ("B", T12 + 12 * 3600 * 10**9, B)]).to_parquet(day_file(root, DAY12), index=False)
    rows([("A", T13 + 600 * 10**9, A_0010), ("A", T13 + 1200 * 10**9, A_0020)]).to_parquet(day_file(root, DAY13), index=False)


@pytest.fixture
def index(tmp_path):
    write_dir(tmp_path)
    return NewsIndex(tmp_path)


def std(x):
    return np.clip([(x[i] - STATS[c]["loc"]) / STATS[c]["scale"] for i, c in enumerate(FEATS)], -10, 10)


def test_lookup_is_causal_standardised_and_crosses_midnight(index):
    assert index.n_features == len(FEATS) + 1
    index.set_hour(2026, 3, 13, 0)
    t = T13 + 5 * 60 * 10**9                       # 00:05 of 13 March: A's last row is yesterday's 23:50, B's 12:00 of yesterday
    f = index.features(["A", "B", "C"], t)
    np.testing.assert_allclose(f[0, :-1], std(A_LATE12), rtol=1e-6); assert f[0, -1] == 1.0
    np.testing.assert_allclose(f[1, :-1], std(B), rtol=1e-6); assert f[1, 1] == 10.0      # clipped at ±10
    assert not f[2].any()                                                                 # unknown market: zeros, not available
    f = index.features(["A"], T13 + 15 * 60 * 10**9)                                      # 00:15: the 00:10 row, not 00:20
    np.testing.assert_allclose(f[0, :-1], std(A_0010), rtol=1e-6)
    f = index.features(["A"], T13 + 1200 * 10**9)                                         # exactly 00:20: ts <= t includes it
    np.testing.assert_allclose(f[0, :-1], std(A_0020), rtol=1e-6)


def test_residency_follows_the_hour(index):
    index.set_hour(2026, 3, 13, 0)
    assert {d.day for d in index._day_cache} == {12, 13}
    index.set_hour(2026, 3, 14, 1)                 # 14 March has no file: 13 stays, 12 is dropped, 14 is missing
    assert {d.day for d in index._day_cache} == {13} and {d.day for d in index._missing_days} == {14}
    assert index.features(["B"], T13 + 2 * 86400 * 10**9)[0, -1] == 0.0                   # B's row (12 March) is no longer resident
