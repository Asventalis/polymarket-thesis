import numpy as np
import pytest

from polymarket_gnn.evaluation.baselines import (
    CONST, DOWN, UP, DedupTracker, direction_from_change, mean_reversion_from_change,
    momentum_from_change, parse_days,
)
from polymarket_gnn.evaluation.metrics import cm_to_metrics, confusion_matrix


def test_direction_from_change_handles_nan_and_eps():
    d = np.array([0.3, -0.2, 0.0, np.nan, 0.05, -0.05])
    np.testing.assert_array_equal(direction_from_change(d), [UP, DOWN, CONST, CONST, UP, DOWN])
    np.testing.assert_array_equal(direction_from_change(d, eps=0.1), [UP, DOWN, CONST, CONST, CONST, CONST])


def test_momentum_and_mean_reversion_are_opposites():
    d = np.array([0.3, -0.2, 0.0, np.nan])
    mom = momentum_from_change(d)
    rev = mean_reversion_from_change(d)
    np.testing.assert_array_equal(mom, [UP, DOWN, CONST, CONST])
    np.testing.assert_array_equal(rev, [DOWN, UP, CONST, CONST])
    np.testing.assert_array_equal(rev, 2 - mom)


def test_dedup_tracker():
    tr = DedupTracker(min_gap_ns=10)
    np.testing.assert_array_equal(tr.keep_mask(["a", "b"], 0), [True, True])
    np.testing.assert_array_equal(tr.keep_mask(["a", "b", "c"], 5), [False, False, True])
    np.testing.assert_array_equal(tr.keep_mask(["a", "b"], 10), [True, True])
    np.testing.assert_array_equal(tr.keep_mask(["c"], 14), [False])
    np.testing.assert_array_equal(tr.keep_mask(["c"], 15), [True])


def test_parse_days():
    assert parse_days("13") == [13]
    assert parse_days("21-23") == [21, 22, 23]
    assert parse_days("6,7, 8") == [6, 7, 8]
    with pytest.raises(ValueError):
        parse_days("")


def test_confusion_and_metrics():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(y, p)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    m = cm_to_metrics(cm)
    assert m["n"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["CONSTANT"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["CONSTANT"]["precision"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert m["class_distribution"]["UP"] == pytest.approx(1 / 3)
