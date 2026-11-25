"""Exponential moving average of the weights (WeightEMA): bias-corrected average, swap into the model."""
import pytest
import torch
import torch.nn as nn

from polymarket_gnn.models.ema import WeightEMA


def test_constant_weights_are_recovered_exactly_at_every_step():
    torch.manual_seed(0)
    net = nn.Linear(4, 3)
    ema = WeightEMA(net, 0.9)
    ref = [p.detach().clone() for p in net.parameters()]
    for n in range(1, 6):
        ema.update(net)
        assert ema.n == n and ema.correction() == pytest.approx(1 - 0.9 ** n)
        for a, r in zip(ema.average(), ref):
            torch.testing.assert_close(a, r, rtol=0, atol=1e-6)   # the bias correction undoes the zero start


def test_average_matches_the_closed_form_and_apply_swaps_once():
    net = nn.Linear(1, 1, bias=False)
    d = 0.5
    ema = WeightEMA(net, d)
    values = [1.0, 2.0, 4.0]
    for v in values:
        with torch.no_grad():
            net.weight.fill_(v)
        ema.update(net)
    # weights (1-d) d^{n-i} on value i, normalised by 1 - d^n
    n = len(values)
    expect = sum((1 - d) * d ** (n - 1 - i) * v for i, v in enumerate(values)) / (1 - d ** n)
    assert ema.average()[0].item() == pytest.approx(expect)
    assert not ema.applied
    info = ema.apply(net)
    assert ema.applied and info["steps"] == n and net.weight.item() == pytest.approx(expect)
    # the model's current (raw) weights do not enter the average once applied unless update is called again
    assert ema.average()[0].item() == pytest.approx(expect)


def test_apply_before_any_update_is_a_no_op():
    net = nn.Linear(2, 2)
    before = [p.detach().clone() for p in net.parameters()]
    ema = WeightEMA(net, 0.99)
    assert ema.apply(net)["steps"] == 0 and not ema.applied
    for p, b in zip(net.parameters(), before):
        torch.testing.assert_close(p, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        WeightEMA(net, 1.0)
