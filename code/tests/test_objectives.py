import numpy as np
import pytest
import torch

from polymarket_gnn.training.objectives import (
    CONST, DOWN, UP, FocalLoss, ThresholdCalibrator, classes_from_delta, classify_delta,
    default_thresholds, inverse_frequency_weights, parse_alpha, undersample_mask,
)


def test_focal_loss_reductions_and_gamma_zero_is_cross_entropy():
    torch.manual_seed(0)
    logits, y = torch.randn(50, 3), torch.randint(0, 3, (50,))
    per = FocalLoss(gamma=2.0, reduction="none")(logits, y)
    assert per.shape == (50,)
    torch.testing.assert_close(FocalLoss(gamma=2.0)(logits, y), per.mean())
    torch.testing.assert_close(FocalLoss(gamma=0.0)(logits, y), torch.nn.functional.cross_entropy(logits, y))
    # confident correct predictions are down-weighted
    easy = torch.tensor([[10.0, 0.0, 0.0]]); t = torch.tensor([0])
    assert FocalLoss(gamma=2.0)(easy, t) < 1e-3 * FocalLoss(gamma=0.0)(easy, t) + 1e-12


def test_parse_alpha():
    assert parse_alpha("2,1,2", "cpu").tolist() == [2.0, 1.0, 2.0]
    assert parse_alpha("none", "cpu") is None and parse_alpha("auto", "cpu") is None
    with pytest.raises(ValueError):
        parse_alpha("1,2", "cpu")


def test_inverse_frequency_weights():
    w = inverse_frequency_weights(torch.tensor([100.0, 800.0, 100.0]), smoothing=0.0)
    assert w.mean().item() == pytest.approx(1.0)
    assert w[0].item() == pytest.approx(w[2].item()) and w[0] / w[1] == pytest.approx(8.0)
    torch.testing.assert_close(inverse_frequency_weights(torch.zeros(3)), torch.ones(3))


def test_undersample_mask_balances_in_expectation():
    g = torch.Generator().manual_seed(0)
    counts = torch.tensor([1000.0, 8000.0, 1000.0])
    labels = torch.cat([torch.zeros(10000), torch.ones(80000), torch.full((10000,), 2.0)]).long()
    m = undersample_mask(labels, counts, ratio=1.0, generator=g)
    kept = torch.bincount(labels[m], minlength=3).float()
    assert kept[0] == 10000 and kept[2] == 10000                 # rare classes untouched
    assert kept[1].item() == pytest.approx(10000, rel=0.05)      # majority cut to the rare-class size
    m2 = undersample_mask(labels, counts, ratio=2.0, generator=g)
    assert torch.bincount(labels[m2], minlength=3)[1].item() == pytest.approx(20000, rel=0.05)
    assert undersample_mask(labels, counts, ratio=0.0).all()     # off
    assert undersample_mask(labels, torch.zeros(3), ratio=1.0).all()            # no statistics yet
    assert undersample_mask(labels[:5], torch.tensor([0.0, 50.0, 0.0]), ratio=1.0).all()   # one class seen


def test_classes_and_classify_delta():
    d = np.array([0.02, -0.01, 0.0, 0.0004])
    assert classes_from_delta(d).tolist() == [UP, DOWN, CONST, UP]
    assert classes_from_delta(d, eps=0.001).tolist() == [UP, DOWN, CONST, CONST]
    p = torch.tensor([0.02, -0.01, 0.0, 0.0004, -0.2])
    assert classify_delta(p, 0.005).tolist() == [UP, DOWN, CONST, CONST, DOWN]
    assert classify_delta(p, 0.0).tolist() == [UP, DOWN, CONST, UP, DOWN]


def test_threshold_calibrator_picks_the_separating_threshold():
    # true CONSTANT samples get small noisy predictions, true UP/DOWN get large ones
    g = torch.Generator().manual_seed(0)
    n = 3000
    y = torch.randint(0, 3, (n,), generator=g)
    pred = torch.where(y == UP, 0.05, torch.where(y == DOWN, -0.05, 0.0)) + 0.004 * torch.randn(n, generator=g)
    cal = ThresholdCalibrator(default_thresholds())
    cal.update(pred[:1500], y[:1500]); cal.update(pred[1500:], y[1500:])      # streaming
    assert cal.n == n
    best = cal.best("balanced_accuracy")
    assert 0.008 < best["tau"] < 0.045 and best["balanced_accuracy"] > 0.98
    rows = cal.table()
    assert len(rows) == len(default_thresholds())
    tau0 = next(r for r in rows if r["tau"] == 0.0)
    assert tau0["balanced_accuracy"] < 0.75          # tau = 0 never predicts CONSTANT
    assert sum(sum(r) for r in rows[0]["cm"]) == n
    assert ThresholdCalibrator(default_thresholds()).best() is None


def test_model_regression_head():
    from polymarket_gnn.data.graph_builder import EDGE_DIM
    from polymarket_gnn.models.causal_gnn import CausalDynamicGNN
    torch.manual_seed(0)
    model = CausalDynamicGNN(lob_dim=8, hidden_dim=16, edge_dim=EDGE_DIM, out_dim=1).eval()
    h_m, h_w = torch.randn(6, 16), torch.randn(4, 16)
    out, _, _ = model(torch.randn(2, 3, 8), torch.tensor([1, 4]), h_m, h_w,
                      torch.empty((2, 0), dtype=torch.long), torch.empty((0, EDGE_DIM)),
                      torch.tensor(2.0), torch.tensor(2.0), predict_idx=torch.tensor([1, 4]))
    assert out.shape == (2, 1)
