import numpy as np
import pytest

from polymarket_gnn.evaluation.smart_money import (
    CONST, DOWN, UP, cluster_bootstrap_gap, cumulative, direction_from_signal, shrunk_skill, skill_buckets, window_sums,
)


def test_shrunk_skill():
    s = shrunk_skill([0, 2, 80, 20, 5], [0, 0, 20, 80, 5], shrink=10)
    assert s[0] == 0 and s[4] == 0
    assert s[1] == pytest.approx(2 / 12)
    assert s[2] == pytest.approx(60 / 110) and s[3] == pytest.approx(-60 / 110)


def test_skill_buckets_equal_size_and_ordered():
    rng = np.random.default_rng(0)
    skill = rng.normal(size=1000)
    eligible = rng.random(1000) < 0.6
    b = skill_buckets(skill, eligible, 10)
    assert (b[~eligible] == -1).all() and b[eligible].min() == 0 and b[eligible].max() == 9
    counts = np.bincount(b[eligible])
    assert counts.max() - counts.min() <= 1
    means = [skill[b == k].mean() for k in range(10)]
    assert means == sorted(means)
    assert (skill_buckets(np.zeros(3), np.zeros(3, dtype=bool)) == -1).all()


def test_window_sums_are_causal_and_half_open():
    ts_raw = np.array([50, 10, 30, 30, 70])
    contrib = np.array([[5.0, 1], [1.0, 1], [3.0, 1], [-3.0, 1], [7.0, 1]])
    ts, cum = cumulative(ts_raw, contrib)
    assert ts.tolist() == [10, 30, 30, 50, 70]
    # window (t - W, t]: the trade at t is included, the one at t - W is not, later trades are never read
    s = window_sums(ts, cum, 50, [20, 21, 1000])
    assert s[0].tolist() == [5.0, 1.0]            # (30, 50] → only the trade at 50
    assert s[1].tolist() == [5.0, 3.0]            # (29, 50] → the two trades at 30 cancel, plus the one at 50
    assert s[2].tolist() == [6.0, 4.0]            # everything up to 50; the trade at 70 is in the future
    assert window_sums(ts, cum, 5, [100])[0].tolist() == [0.0, 0.0]
    # brute force on random data
    rng = np.random.default_rng(1)
    ts_raw = rng.integers(0, 1000, size=200)
    c = rng.normal(size=(200, 2))
    ts, cum = cumulative(ts_raw, c)
    for t in (0, 137, 500, 999, 2000):
        for w in (1, 50, 400):
            ref = c[(ts_raw > t - w) & (ts_raw <= t)].sum(axis=0)
            np.testing.assert_allclose(window_sums(ts, cum, t, [w])[0], ref, atol=1e-9)


def test_direction_from_signal():
    assert direction_from_signal(np.array([0.5, -2.0, 0.0, 1e-12])).tolist() == [UP, DOWN, CONST, CONST]


def test_cluster_bootstrap_gap():
    # same hit rate in both groups → interval contains 0; clearly different → it does not
    rng = np.random.default_rng(0)
    n = 300
    wa, la = rng.binomial(40, 0.5, n), None
    la = 40 - wa
    wb = rng.binomial(40, 0.5, n); lb = 40 - wb
    g, lo, hi = cluster_bootstrap_gap(wa, la, wb, lb, n_boot=300)
    assert lo < 0 < hi and abs(g) < 0.03
    wc = rng.binomial(40, 0.6, n); lc = 40 - wc
    g, lo, hi = cluster_bootstrap_gap(wc, lc, wb, lb, n_boot=300)
    assert g == pytest.approx(0.1, abs=0.02) and lo > 0.05
