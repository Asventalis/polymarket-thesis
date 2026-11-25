"""Lookback skip path and global state (after NAVIS), anti-symmetric state update (after CTAN)."""
import pytest
import torch

from polymarket_gnn.data.graph_builder import EDGE_DIM
from polymarket_gnn.models.causal_gnn import AntiSymmetricCell, CausalDynamicGNN
from polymarket_gnn.models.sparse_state import DecayClock, make_lazy_states, sparse_step
from tests.test_sparse_state import HID, K, LOB, N_M, N_W, make_blocks

L = 2                      # lookbacks: the last 3 * L columns of the windows are [change, sign, available] each
SKIP = 3 * L
D = torch.float64


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, dropout=0.0, **kw).double().eval()


def step(model, windows, active, pred, ei=None, ef=None, seed=7):
    g = torch.Generator().manual_seed(seed)
    clock = DecayClock("cpu")
    sm, sw = make_lazy_states(model, torch.randn(N_M, HID, generator=g, dtype=D), torch.randn(N_W, HID, generator=g, dtype=D), clock)
    ei = torch.empty((2, 0), dtype=torch.long) if ei is None else ei
    ef = torch.empty((0, EDGE_DIM), dtype=D) if ef is None else ef
    return sparse_step(model, clock, sm, sw, windows, active, ei, ef, 2.0, pred)


def lookback_windows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, K, LOB, generator=g, dtype=D)
    sign = torch.randint(-1, 2, (n, L), generator=g).to(D)
    avail = torch.ones(n, L, dtype=D)
    ctx = torch.stack([sign * torch.rand(n, L, generator=g, dtype=D), sign, avail], dim=2).reshape(n, SKIP)
    w[:, :, -SKIP:] = ctx[:, None, :]
    return w, sign


def test_defaults_leave_the_model_untouched():
    a, b = build(), build(skip_dim=SKIP)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"skip.weight", "skip.bias"}
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0)
    # zero-initialised path: same output as the model without it
    w, _ = lookback_windows(6)
    active = torch.tensor([3, 5, 9, 11, 20, 31]); pred = torch.tensor([5, 9, 31])
    torch.testing.assert_close(step(a, w, active, pred), step(b, w, active, pred), rtol=0, atol=0)


def test_skip_path_can_express_mean_reversion_and_is_trained():
    model = build(skip_dim=SKIP)
    w, sign = lookback_windows(8, seed=1)
    active = torch.tensor([1, 4, 6, 7, 12, 18, 25, 33])
    pred = torch.tensor([33, 1, 12, 6, 25])                       # a subset, in another order
    rows = [7, 0, 4, 2, 6]
    with torch.no_grad():
        # logits [DOWN, CONSTANT, UP] = big * [sign, 1 - 2|sign|... ] on lookback 1: opposite of the past move
        big = 1e3
        model.skip.weight.zero_(); model.skip.bias.zero_()
        j_sign, j_moved = 3 * 1 + 1, SKIP + 1                      # columns of lookback index 1
        model.skip.weight[0, j_sign] = big; model.skip.weight[2, j_sign] = -big
        model.skip.weight[1, j_moved] = -big; model.skip.bias[1] = big / 2
    got = step(model, w, active, pred).argmax(dim=1)
    s = sign[rows, 1]
    expected = torch.where(s > 0, 0, torch.where(s < 0, 2, 1))     # up → DOWN, down → UP, flat → CONSTANT
    assert got.tolist() == expected.tolist()

    model = build(skip_dim=SKIP)
    logits = step(model, w, active, pred)
    logits.sum().backward()
    assert model.skip.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):                                # a predicted market without a window in the block
        step(model, w, active, torch.tensor([1, 2]))
    with pytest.raises(NotImplementedError):                       # dense engine: not implemented, must not run silently
        z = torch.zeros(N_M, HID, dtype=D)
        model(w, active, z, torch.zeros(N_W, HID, dtype=D), torch.empty((2, 0), dtype=torch.long),
              torch.empty((0, EDGE_DIM), dtype=D), torch.tensor(1.0, dtype=D), torch.tensor(1.0, dtype=D), pred)


def test_global_state_is_the_mean_state_of_the_active_markets():
    model = build(global_state=True)
    assert model.predictor.net[0].in_features == 2 * HID
    w, _ = lookback_windows(5, seed=2)
    active = torch.tensor([2, 8, 9, 30, 35]); pred = torch.tensor([8, 30])
    logits = step(model, w, active, pred)
    # reference: run the same block on the states by hand
    g = torch.Generator().manual_seed(7)
    clock = DecayClock("cpu")
    sm, _ = make_lazy_states(model, torch.randn(N_M, HID, generator=g, dtype=D), torch.randn(N_W, HID, generator=g, dtype=D), clock)
    clock.advance(model.decay_rate, 2.0)
    lu = model.lob_update
    new = lu.norm(lu.gru_cell(lu.proj(lu.gru(w)[1].squeeze(0)), sm.lookup(active)))
    ref = model.predictor(torch.cat([new[[1, 3]], new.mean(dim=0, keepdim=True).expand(2, -1)], dim=1))
    torch.testing.assert_close(logits, ref, rtol=0, atol=1e-10)
    # the prediction of a market depends on the windows of the OTHER active markets
    w2 = w.clone(); w2[4] += 1.0
    assert (step(model, w2, active, pred) - logits).abs().max() > 1e-6


def test_antisymmetric_cell():
    torch.manual_seed(0)
    cell = AntiSymmetricCell(HID, eps=0.3, gamma=0.2).double()
    x, h = torch.randn(5, HID, dtype=D), torch.randn(5, HID, dtype=D)
    a = cell.weight - cell.weight.t()
    torch.testing.assert_close(a, -a.t())
    ref = h + 0.3 * torch.tanh(h @ (a - 0.2 * torch.eye(HID, dtype=D)).t() + x + cell.bias)
    torch.testing.assert_close(cell(x, h), ref, rtol=0, atol=1e-12)
    # an update moves the state by at most eps per dimension: the old state is never overwritten
    assert (cell(x, h) - h).abs().max() <= 0.3


def test_ctan_model_sparse_equals_dense_and_differs_from_gru():
    blocks = make_blocks(30, seed=4)
    out = {}
    for name, kw in (("gru", {}), ("ctan", {"state_update": "ctan"})):
        model = build(seed=3, **kw)
        if name == "ctan":
            assert isinstance(model.lob_update.gru_cell, AntiSymmetricCell) and isinstance(model.mp.gru_cell, AntiSymmetricCell)
            assert isinstance(model.lob_update.norm, torch.nn.Identity)
        with torch.no_grad():
            model.decay_rate.fill_(0.05)
        g = torch.Generator().manual_seed(7)
        h_m = torch.randn(N_M, HID, generator=g, dtype=D); h_w = torch.randn(N_W, HID, generator=g, dtype=D)
        clock = DecayClock("cpu")
        sm, sw = make_lazy_states(model, h_m, h_w, clock)
        dense, sparse = [], []
        with torch.no_grad():
            for windows, active, ei, ef, dt, pred, _ in blocks:
                dtt = torch.tensor(dt, dtype=D)
                lg, h_m, h_w = model(windows, active, h_m, h_w, ei, ef, dtt, dtt, pred)
                dense.append(lg)
                sparse.append(sparse_step(model, clock, sm, sw, windows, active, ei, ef, dt, pred))
        for a, b in zip(dense, sparse):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-9)
        assert all(torch.isfinite(x).all() for x in sparse)
        out[name] = torch.cat(sparse)
    assert (out["gru"] - out["ctan"]).abs().max() > 1e-3
    with pytest.raises(ValueError):
        build(state_update="lstm")


def test_skip_extra_columns_and_mean_reversion_init():
    E = 3                                                         # extra columns just before the lookback block
    w, sign = lookback_windows(8, seed=1)
    active = torch.tensor([1, 4, 6, 7, 12, 18, 25, 33]); pred = torch.tensor([33, 1, 12, 6, 25]); rows = [7, 0, 4, 2, 6]
    base = build()
    for extra in (0, E):
        model = build(skip_dim=SKIP, skip_extra=extra, skip_init_lookback=1)
        assert model.skip.in_features == extra + SKIP + L
        with torch.no_grad():                                     # silence the predictor: the output is the direct path alone
            model.predictor.net[-1].weight.zero_(); model.predictor.net[-1].bias.zero_()
        got = step(model, w, active, pred).argmax(dim=1)
        s = sign[rows, 1]
        assert got.tolist() == torch.where(s > 0, 0, torch.where(s < 0, 2, 1)).tolist()   # Mean Reversion on lookback 1
        # everything but the direct path is initialised as in the base model
        for k, v in base.state_dict().items():
            if not k.startswith("predictor.net.3"):
                torch.testing.assert_close(v, build(skip_dim=SKIP, skip_extra=extra, skip_init_lookback=1).state_dict()[k], rtol=0, atol=0)
    # the extra columns reach the output: with a weight on one of them the logits follow that column
    model = build(skip_dim=SKIP, skip_extra=E)
    with torch.no_grad():
        model.skip.weight[2, 0] = 1.0
    ref = step(build(skip_dim=SKIP, skip_extra=E), w, active, pred)
    out = step(model, w, active, pred)
    torch.testing.assert_close(out[:, 2] - ref[:, 2], w[rows, -1, -(SKIP + E)], rtol=0, atol=1e-10)
    with pytest.raises(ValueError):
        build(skip_dim=SKIP, skip_init_lookback=5)
