"""
The row-sparse engine must reproduce the dense training loop: same logits,
same losses, same parameter gradients, same parameters after the optimizer
steps and same final node states.  Everything runs in float64 so that the
comparison can be tight.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from polymarket_gnn.data.graph_builder import EDGE_DIM
from polymarket_gnn.models.causal_gnn import CausalDynamicGNN, half_lives_to_rates
from polymarket_gnn.models.sparse_state import DecayClock, LazyNodeState, make_lazy_states, sparse_step

N_M, N_W, HID, K, LOB = 40, 30, 16, 4, 9
TBPTT, ACCUM = 5, 2
TOL = 1e-9


def make_blocks(n_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for b in range(n_blocks):
        n_act = 0 if b % 7 == 3 else int(torch.randint(1, 9, (1,), generator=g))
        active = torch.randperm(N_M, generator=g)[:n_act].sort().values
        windows = torch.randn(n_act, K, LOB, generator=g, dtype=torch.float64)
        n_e = 0 if b % 5 == 4 else int(torch.randint(1, 15, (1,), generator=g))
        ei = torch.stack([torch.randint(0, N_W, (n_e,), generator=g), torch.randint(0, N_M, (n_e,), generator=g)])
        ef = torch.randn(n_e, EDGE_DIM, generator=g, dtype=torch.float64)
        dt = float(torch.rand(1, generator=g)) * 4 + 0.5
        n_pred = int(torch.randint(0, n_act + 1, (1,), generator=g)) if n_act else 0
        pred = active[:n_pred]
        y = torch.randint(0, 3, (n_pred,), generator=g)
        blocks.append((windows, active, ei, ef, dt, pred, y))
    return blocks


def run(engine: str, blocks, train: bool, dropout: float, use_checkpoint: bool = True,
        decay_mode: str = "learned", half_lives=None, wallet_half_lives=None):
    torch.manual_seed(123)
    model = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, dropout=dropout, decay_mode=decay_mode,
                             market_half_lives=half_lives, wallet_half_lives=wallet_half_lives).double()
    with torch.no_grad():
        model.decay_rate.fill_(0.05)          # large enough for the decay to matter
    model.train(train)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(7)
    h_m = torch.randn(N_M, HID, generator=g, dtype=torch.float64)
    h_w = torch.randn(N_W, HID, generator=g, dtype=torch.float64)
    if engine == "sparse":
        clock = DecayClock("cpu")
        sm, sw = make_lazy_states(model, h_m, h_w, clock)

    torch.manual_seed(999)                    # dropout stream
    out = {"logits": [], "loss": [], "grads": [], "params": []}
    tb_loss = torch.zeros((), dtype=torch.float64)
    for b, (windows, active, ei, ef, dt, pred, y) in enumerate(blocks):
        with torch.set_grad_enabled(train):
            if engine == "dense":
                dtt = torch.tensor(dt, dtype=torch.float64)
                hm_in, hw_in = h_m.clone(), h_w.clone()
                if train and use_checkpoint:
                    hm_in.requires_grad_(True)
                    logits, h_m, h_w = checkpoint(model, windows, active, hm_in, hw_in, ei, ef, dtt, dtt, pred,
                                                  use_reentrant=False)
                else:
                    logits, h_m, h_w = model(windows, active, hm_in, hw_in, ei, ef, dtt, dtt, pred)
            else:
                logits = sparse_step(model, clock, sm, sw, windows, active, ei, ef, dt, pred)
        out["logits"].append(logits.detach().clone())
        if len(pred) > 0:
            loss = F.cross_entropy(logits, y)
            out["loss"].append(loss.item())
            if train:
                tb_loss = tb_loss + loss
        if b > 0 and b % TBPTT == 0:
            if train:
                if tb_loss.requires_grad:
                    (tb_loss / ACCUM).backward()
                if b % (TBPTT * ACCUM) == 0:
                    out["grads"].append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
                    opt.step()
                    opt.zero_grad()
                    out["params"].append([p.detach().clone() for p in model.parameters()])
                tb_loss = torch.zeros((), dtype=torch.float64)
            if engine == "dense":
                h_m = torch.nan_to_num(h_m.detach(), nan=0.0)
                h_w = torch.nan_to_num(h_w.detach(), nan=0.0)
            else:
                sm.commit(); sw.commit(); clock.commit()
    if engine == "dense":
        out["h_m"], out["h_w"] = h_m.detach(), h_w.detach()
    else:
        out["h_m"], out["h_w"] = sm.materialize(), sw.materialize()
    return out


def assert_same(a, b):
    assert len(a["logits"]) == len(b["logits"])
    for x, y in zip(a["logits"], b["logits"]):
        torch.testing.assert_close(x, y, rtol=TOL, atol=TOL)
    assert a["loss"] == pytest.approx(b["loss"], rel=TOL, abs=TOL)
    assert len(a["grads"]) == len(b["grads"])
    for ga, gb in zip(a["grads"], b["grads"]):
        for x, y in zip(ga, gb):
            assert (x is None) == (y is None)
            if x is not None:
                torch.testing.assert_close(x, y, rtol=1e-7, atol=TOL)
    for pa, pb in zip(a["params"], b["params"]):
        for x, y in zip(pa, pb):
            torch.testing.assert_close(x, y, rtol=TOL, atol=TOL)
    torch.testing.assert_close(a["h_m"], b["h_m"], rtol=TOL, atol=TOL)
    torch.testing.assert_close(a["h_w"], b["h_w"], rtol=TOL, atol=TOL)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_sparse_equals_dense_training(dropout):
    blocks = make_blocks(43)
    dense = run("dense", blocks, train=True, dropout=dropout)
    sparse = run("sparse", blocks, train=True, dropout=dropout)
    assert len(dense["grads"]) >= 4 and any(len(b[5]) for b in blocks)
    assert_same(dense, sparse)
    # the decay rate must receive a (matching, non-zero) gradient
    names = [n for n, _ in CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM).named_parameters()]
    i = names.index("decay_rate")
    assert dense["grads"][0][i].abs().item() > 0


def test_checkpointed_dense_equals_plain_dense():
    blocks = make_blocks(23, seed=3)
    a = run("dense", blocks, train=True, dropout=0.1, use_checkpoint=True)
    b = run("dense", blocks, train=True, dropout=0.1, use_checkpoint=False)
    assert_same(a, b)


def test_sparse_equals_dense_eval():
    blocks = make_blocks(31, seed=5)
    assert_same(run("dense", blocks, train=False, dropout=0.1), run("sparse", blocks, train=False, dropout=0.1))


def test_decay_rate_change_between_segments_is_tracked():
    # after optimizer steps the decay rate differs from its initial value: states must still agree
    blocks = make_blocks(61, seed=11)
    dense = run("dense", blocks, train=True, dropout=0.0)
    sparse = run("sparse", blocks, train=True, dropout=0.0)
    names = [n for n, _ in CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM).named_parameters()]
    i = names.index("decay_rate")
    assert abs(dense["params"][-1][i].item() - 0.05) > 1e-6
    assert_same(dense, sparse)


def test_lazy_state_lookup_write_commit():
    clock = DecayClock("cpu")
    st = LazyNodeState(torch.ones(5, 2, dtype=torch.float64), clock)
    rate = torch.tensor([0.5], dtype=torch.float64)
    clock.advance(rate, 2.0)                                   # D = 1
    torch.testing.assert_close(st.lookup(torch.tensor([0, 0, 3])), torch.full((3, 2), torch.exp(torch.tensor(-1.0, dtype=torch.float64)).item(), dtype=torch.float64))
    st.write(torch.tensor([3]), torch.full((1, 2), 7.0, dtype=torch.float64))
    clock.advance(rate, 2.0)                                   # D = 2
    got = st.lookup(torch.tensor([3, 1]))
    e = torch.exp(torch.tensor([-1.0, -2.0], dtype=torch.float64))
    torch.testing.assert_close(got, torch.stack([7.0 * e[0].expand(2), e[1].expand(2)]))
    st.commit(); clock.commit()
    assert st.ov_idx.numel() == 0 and clock.base == pytest.approx(2.0)
    torch.testing.assert_close(st.lookup(torch.tensor([3, 1])), got)          # unchanged by commit
    torch.testing.assert_close(st.materialize()[3], got[0])


# ── memory modes ─────────────────────────────────────────────────────────────

def test_half_lives_to_rates():
    r = half_lives_to_rates([60, 3600, float("inf")], 9)
    assert r.shape == (9,) and r.dtype == torch.float64
    ln2 = 0.6931471805599453
    torch.testing.assert_close(r[:3], torch.full((3,), ln2 / 60, dtype=torch.float64))
    torch.testing.assert_close(r[3:6], torch.full((3,), ln2 / 3600, dtype=torch.float64))
    assert (r[6:] == 0).all()
    assert (half_lives_to_rates([10, 20], 5) > 0).sum() == 5             # uneven split still covers every dim
    with pytest.raises(ValueError):
        half_lives_to_rates([0], 4)
    with pytest.raises(ValueError):
        half_lives_to_rates([1, 2, 3], 2)


@pytest.mark.parametrize("mode,kw", [
    ("none", {}),
    ("multiscale", {"half_lives": [3.0, 40.0, float("inf")]}),
    ("multiscale", {"half_lives": [2.0, float("inf")], "wallet_half_lives": [50.0, 500.0, float("inf"), 5.0]}),
])
def test_sparse_equals_dense_for_memory_modes(mode, kw):
    blocks = make_blocks(43, seed=2)
    dense = run("dense", blocks, train=True, dropout=0.1, decay_mode=mode, **kw)
    sparse = run("sparse", blocks, train=True, dropout=0.1, decay_mode=mode, **kw)
    assert_same(dense, sparse)
    # these modes do not use the learnable scalar rate: it must receive no gradient
    names = [n for n, _ in CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM).named_parameters()]
    g = dense["grads"][0][names.index("decay_rate")]
    assert g is None or g.abs().item() == 0


def test_memory_modes_differ_from_each_other_and_decay_as_specified():
    blocks = make_blocks(21, seed=4)
    a = run("sparse", blocks, train=False, dropout=0.0, decay_mode="learned")
    b = run("sparse", blocks, train=False, dropout=0.0, decay_mode="none")
    c = run("sparse", blocks, train=False, dropout=0.0, decay_mode="multiscale", half_lives=[3.0, float("inf")])
    assert (a["h_m"] - b["h_m"]).abs().max() > 1e-3 and (b["h_m"] - c["h_m"]).abs().max() > 1e-3

    # an untouched node: the fast half keeps 1/2 per half-life, the "inf" half is unchanged, 'none' keeps everything
    for mode, hl in (("multiscale", [10.0, float("inf")]), ("none", None)):
        model = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, decay_mode=mode, market_half_lives=hl).double().eval()
        h_m, h_w = torch.ones(3, HID, dtype=torch.float64), torch.ones(2, HID, dtype=torch.float64)
        empty = (torch.empty((0, K, LOB), dtype=torch.float64), torch.empty(0, dtype=torch.long),
                 torch.empty((2, 0), dtype=torch.long), torch.empty((0, EDGE_DIM), dtype=torch.float64))
        clock = DecayClock("cpu"); sm, sw = make_lazy_states(model, h_m, h_w, clock)
        with torch.no_grad():
            for _ in range(2):                                       # two blocks of 10 s
                _, h_m, h_w = model(empty[0], empty[1], h_m, h_w, empty[2], empty[3], torch.tensor(10.0), torch.tensor(10.0), empty[1])
                sparse_step(model, clock, sm, sw, empty[0], empty[1], empty[2], empty[3], 10.0, empty[1])
        expected_fast = 0.25 if mode == "multiscale" else 1.0
        torch.testing.assert_close(h_m[:, : HID // 2], torch.full((3, HID // 2), expected_fast, dtype=torch.float64))
        torch.testing.assert_close(h_m[:, HID // 2:], torch.ones(3, HID // 2, dtype=torch.float64))
        torch.testing.assert_close(sm.materialize(), h_m)
        torch.testing.assert_close(sw.materialize(), h_w)


# ── hierarchical memory ──────────────────────────────────────────────────────
# Reference written from the specification with plain dense tensors and a Python loop over
# markets, independent of the sparse implementation:
#   * message passing and LOB update act on the per-block state only;
#   * level 0 (own GRU cell) reads the per-block state of a market at most once per period,
#     level 1 (own GRU cell) reads the level-0 state just refreshed, at most once per its period;
#   * the predictor reads [per-block, level 0, level 1].

PERIODS = (6.0, 25.0)


def run_hier(engine: str, blocks, train: bool = True, dropout: float = 0.1, periods=PERIODS, mem_seed: int = 5):
    from polymarket_gnn.models.sparse_state import HierarchicalMemory
    torch.manual_seed(123)
    model = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, dropout=dropout, memory_periods=periods).double()
    with torch.no_grad():
        model.decay_rate.fill_(0.05)
    model.train(train)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(7)
    h_m = torch.randn(N_M, HID, generator=g, dtype=torch.float64)
    h_w = torch.randn(N_W, HID, generator=g, dtype=torch.float64)
    if engine == "sparse":
        clock = DecayClock("cpu")
        sm, sw = make_lazy_states(model, h_m, h_w, clock)
        mem = HierarchicalMemory(model, N_M, clock, "cpu", dtype=torch.float64, seed=mem_seed)
    else:
        pg = torch.Generator().manual_seed(mem_seed)
        phase = [torch.rand(N_M, generator=pg, dtype=torch.float64) * p for p in periods]
        slow = [torch.zeros(N_M, HID, dtype=torch.float64) for _ in periods]
        next_due = [[float("-inf")] * N_M for _ in periods]
        t = 0.0

    torch.manual_seed(999)
    out = {"logits": [], "loss": [], "grads": [], "params": [], "refreshes": [[] for _ in periods]}
    tb_loss = torch.zeros((), dtype=torch.float64)
    for b, (windows, active, ei, ef, dt, pred, y) in enumerate(blocks):
        with torch.set_grad_enabled(train):
            if engine == "dense":
                t += dt
                dtt = torch.tensor(dt, dtype=torch.float64)
                h_m, h_w = model.advance(windows, active, h_m, h_w, ei, ef, dtt, dtt)
                for m in sorted(set(active.tolist()) | set(ei[1].tolist())):
                    x = h_m[m]
                    for lvl, period in enumerate(periods):
                        if next_due[lvl][m] > t:
                            break
                        first = next_due[lvl][m] == float("-inf")
                        x = model.slow_cells[lvl](x.unsqueeze(0), slow[lvl][m].unsqueeze(0)).squeeze(0)
                        slow[lvl] = slow[lvl].clone()
                        slow[lvl][m] = x
                        next_due[lvl][m] = t + period - (phase[lvl][m].item() if first else 0.0)
                        out["refreshes"][lvl].append((b, m))
                logits = model.predictor(torch.cat([h_m[pred]] + [s[pred] for s in slow], dim=1))
            else:
                before = [nd.clone() for nd in mem.next_due]
                logits = sparse_step(model, clock, sm, sw, windows, active, ei, ef, dt, pred, mem)
                for lvl in range(len(periods)):
                    out["refreshes"][lvl] += [(b, m) for m in (mem.next_due[lvl] != before[lvl]).nonzero().flatten().tolist()]
        out["logits"].append(logits.detach().clone())
        if len(pred) > 0:
            loss = F.cross_entropy(logits, y)
            out["loss"].append(loss.item())
            if train:
                tb_loss = tb_loss + loss
        if b > 0 and b % TBPTT == 0:
            if train:
                if tb_loss.requires_grad:
                    (tb_loss / ACCUM).backward()
                if b % (TBPTT * ACCUM) == 0:
                    out["grads"].append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
                    opt.step()
                    opt.zero_grad()
                    out["params"].append([p.detach().clone() for p in model.parameters()])
                tb_loss = torch.zeros((), dtype=torch.float64)
            if engine == "dense":
                h_m, h_w = h_m.detach(), h_w.detach()
                slow = [s.detach() for s in slow]
            else:
                sm.commit(); sw.commit(); clock.commit(); mem.commit()
    if engine == "dense":
        out["h_m"], out["h_w"], out["slow"] = h_m.detach(), h_w.detach(), [s.detach() for s in slow]
    else:
        out["h_m"], out["h_w"], out["slow"] = sm.materialize(), sw.materialize(), [s.materialize() for s in mem.states]
    out["names"] = [n for n, _ in model.named_parameters()]
    return out


@pytest.mark.parametrize("dropout,train", [(0.0, True), (0.1, True), (0.1, False)])
def test_hierarchical_memory_matches_reference(dropout, train):
    blocks = make_blocks(83, seed=6)
    ref = run_hier("dense", blocks, train=train, dropout=dropout)
    got = run_hier("sparse", blocks, train=train, dropout=dropout)
    assert_same(ref, got)
    for a, b in zip(ref["slow"], got["slow"]):
        torch.testing.assert_close(a, b, rtol=TOL, atol=TOL)
    assert ref["refreshes"] == got["refreshes"]
    assert len(ref["refreshes"][0]) > len(ref["refreshes"][1]) > N_M        # both levels tick repeatedly
    if train:
        # each level's own GRU cell is trained
        for lvl in range(len(PERIODS)):
            i = ref["names"].index(f"slow_cells.{lvl}.weight_ih")
            assert max(g[i].abs().max().item() for g in got["grads"] if g[i] is not None) > 0


def test_hierarchical_memory_schedule():
    blocks = make_blocks(83, seed=6)
    out = run_hier("sparse", blocks, train=False, dropout=0.0)
    lvl0, lvl1 = out["refreshes"]
    times = torch.tensor([b[4] for b in blocks], dtype=torch.float64).cumsum(0)
    assert set(lvl1) <= set(lvl0)                       # a level is refreshed only from a level below refreshed in that block
    for refreshes, period in zip((lvl0, lvl1), PERIODS):
        per_market = {}
        for b, m in refreshes:
            per_market.setdefault(m, []).append(times[b].item())
        gaps23 = [ts[i + 1] - ts[i] for ts in per_market.values() for i in range(1, len(ts) - 1)]
        assert gaps23 and min(gaps23) >= period - 1e-9   # never more often than once per period (after the phased first gap)
        first_gaps = [ts[1] - ts[0] for ts in per_market.values() if len(ts) > 1]
        assert len(set(round(x, 6) for x in first_gaps)) > 1          # staggered
    # a market is refreshed only in a block where its per-block state is written
    for b, m in lvl0:
        assert m in set(blocks[b][1].tolist()) | set(blocks[b][2][1].tolist())


def test_memory_option_leaves_the_base_model_untouched():
    torch.manual_seed(0); base = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM)
    torch.manual_seed(0); hier = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, memory_periods=[60, 3600])
    sb, sh = base.state_dict(), hier.state_dict()
    assert not any(k.startswith("slow_cells") for k in sb)
    assert set(sh) - set(sb) == {f"slow_cells.{l}.{n}" for l in (0, 1) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
    for k in sb:                                        # same initial weights for everything but the predictor's wider input layer
        if not k.startswith("predictor"):
            torch.testing.assert_close(sb[k], sh[k], rtol=0, atol=0)
    assert sh["predictor.net.0.weight"].shape == (HID // 2, 3 * HID)
    with pytest.raises(ValueError):
        CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, memory_periods=[3600, 60])
    with pytest.raises(ValueError):                     # the sparse step refuses a model/state mismatch
        m = hier.double(); clock = DecayClock("cpu")
        sm, sw = make_lazy_states(m, torch.zeros(3, HID, dtype=torch.float64), torch.zeros(2, HID, dtype=torch.float64), clock)
        sparse_step(m, clock, sm, sw, torch.empty((0, K, LOB), dtype=torch.float64), torch.empty(0, dtype=torch.long),
                    torch.empty((2, 0), dtype=torch.long), torch.empty((0, EDGE_DIM), dtype=torch.float64), 1.0)
