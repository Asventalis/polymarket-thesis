"""
sparse_state.py

Row-sparse execution of ``CausalDynamicGNN`` — same parameters, same
mathematics, without touching the full node-state matrices at every block.

Why
---
The dense training loop keeps ``h_market`` (175k x 128) and ``h_wallet``
(646k x 128) as plain tensors and, at EVERY block:
    * clones both (to keep autograd happy),
    * multiplies both by the decay factor,
    * clones them again inside the LOB update and inside each message-passing
      direction, and allocates a zero matrix of the same size for aggregation,
although a block only reads and writes a few hundred rows.  That is several GB
of memory traffic per block (forward, and again in backward), which is what
the GPU spends its time on.

How
---
1. Lazy decay.  The dense code multiplies every row by exp(-|r|·dt_k) at each
   block k.  Since exp(a)·exp(b) = exp(a+b), a row written at block j and read
   at block k equals  value_j · exp(-(D_k − D_j))  with the "decay clock"
   D_k = Σ_{i≤k} |r_i|·dt_i  (r is the learnable decay rate, so the clock
   accumulates the rate that was in force at each block).  We store, per row,
   the value and the clock at its last write and apply the decay on read.
   The clock is kept in float64 (it grows to hundreds while per-block
   increments are ~1e-3).
2. Overlay.  Within a truncated-BPTT segment the rows that have been written
   are kept in a small overlay (with their autograd graph); every other row is
   a constant read from the big storage matrix.  At the end of the segment the
   overlay is written back into the storage in place (detached, NaN → 0, as in
   the dense loop).

Gradients are identical to the dense loop: a row not written in the current
segment is a constant in both formulations, and the decay rate receives
gradient only from the blocks of the current segment (the part of the clock
accumulated in earlier segments is a detached constant, exactly like the
detached state it multiplies in the dense loop).

``tests/test_sparse_state.py`` runs the dense loop (with checkpointing, as in
training) and this one side by side in float64 and checks logits, losses,
parameter gradients, updated parameters and final states.
"""
from __future__ import annotations

from typing import Optional

import torch

from polymarket_gnn.models.causal_gnn import CausalDynamicGNN, TransactionAttentionMP


def make_lazy_states(model: CausalDynamicGNN, h_market: torch.Tensor, h_wallet: torch.Tensor, clock: "DecayClock"):
    """Market and wallet lazy states configured for the model's decay mode."""
    if model.decay_mode == "multiscale":
        return (LazyNodeState(h_market, clock, model.market_decay_rates),
                LazyNodeState(h_wallet, clock, model.wallet_decay_rates))
    return LazyNodeState(h_market, clock), LazyNodeState(h_wallet, clock)


class DecayClock:
    """
    Shared clock of the market and wallet states.

    * ``now()``  — cumulative decay exponent D = Σ |rate|·dt of the LEARNED scalar rate
                   (advanced only when a rate is passed; stays 0 in the other modes);
    * ``t``      — elapsed seconds, used by states with fixed per-dimension rates.
    """

    def __init__(self, device):
        self.base = 0.0                                                   # detached part (python float64)
        self.win = torch.zeros((), dtype=torch.float64, device=device)    # current segment (carries grad)
        self.t = 0.0                                                      # elapsed time in seconds

    def advance(self, rate: Optional[torch.Tensor], dt: float) -> None:
        self.t += float(dt)
        if rate is not None:
            self.win = self.win + torch.abs(rate).reshape(()).to(torch.float64) * float(dt)

    def now(self) -> torch.Tensor:
        return self.win + self.base

    def commit(self) -> None:
        self.base += float(self.win.detach())
        self.win = torch.zeros_like(self.win)


class LazyNodeState:
    """
    Node-state matrix with lazy decay and a per-segment overlay of written rows.

    ``rates=None``: scalar decay driven by the clock's learned exponent (also covers "no decay":
    the exponent then never advances).  ``rates`` = (D,) tensor: fixed per-dimension rates,
    decay of a row = exp(-rates · seconds since its last write).
    """

    def __init__(self, h0: torch.Tensor, clock: DecayClock, rates: Optional[torch.Tensor] = None):
        self.clock = clock
        self.rates = None if rates is None else rates.detach().to(device=h0.device, dtype=torch.float64)
        self.H = h0.detach().clone()                                      # (N, D) values at their last write
        self.D_last = torch.zeros(h0.size(0), dtype=torch.float64, device=h0.device)
        self._reset_overlay()

    def _reset_overlay(self) -> None:
        dev = self.H.device
        self.ov_idx = torch.empty(0, dtype=torch.long, device=dev)        # sorted, unique
        self.ov_val = torch.empty((0, self.H.size(1)), dtype=self.H.dtype, device=dev)
        self.ov_D = torch.empty(0, dtype=torch.float64, device=dev)

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """Current (decayed) state of rows ``idx`` (repeats allowed)."""
        vals = self.H[idx]
        d_st = self.D_last[idx]
        if self.ov_idx.numel() > 0 and idx.numel() > 0:
            pos = torch.searchsorted(self.ov_idx, idx).clamp(max=self.ov_idx.numel() - 1)
            found = self.ov_idx[pos] == idx
            vals = torch.where(found.unsqueeze(1), self.ov_val[pos], vals)
            d_st = torch.where(found, self.ov_D[pos], d_st)
        elapsed = self._stamp() - d_st
        if self.rates is None:
            decay = torch.exp(-elapsed).to(vals.dtype).unsqueeze(1)
        else:
            decay = torch.exp(-elapsed.unsqueeze(1) * self.rates.unsqueeze(0)).to(vals.dtype)
        return vals * decay

    def _stamp(self) -> torch.Tensor:
        """Current value of the quantity rows are stamped with: learned exponent, or elapsed seconds."""
        if self.rates is None:
            return self.clock.now()
        return torch.tensor(self.clock.t, dtype=torch.float64, device=self.H.device)

    def write(self, idx: torch.Tensor, vals: torch.Tensor) -> None:
        """Sets rows ``idx`` (unique) to ``vals`` as of the current clock."""
        if idx.numel() == 0:
            return
        d_now = self._stamp().expand(idx.numel())
        if self.ov_idx.numel() > 0:
            keep = ~torch.isin(self.ov_idx, idx)
            idx = torch.cat([self.ov_idx[keep], idx])
            vals = torch.cat([self.ov_val[keep], vals])
            d_now = torch.cat([self.ov_D[keep], d_now])
        order = torch.argsort(idx)
        self.ov_idx, self.ov_val, self.ov_D = idx[order], vals[order], d_now[order]

    @torch.no_grad()
    def commit(self) -> None:
        """End of a TBPTT segment: write the overlay back (detached, NaN → 0) and drop the graph."""
        if self.ov_idx.numel() > 0:
            self.H[self.ov_idx] = torch.nan_to_num(self.ov_val.detach(), nan=0.0)
            self.D_last[self.ov_idx] = self.ov_D.detach()
        self._reset_overlay()

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        """Dense (N, D) state as of the current clock — what the dense loop would hold."""
        idx = torch.arange(self.H.size(0), device=self.H.device)
        return self.lookup(idx).detach()


class HierarchicalMemory:
    """
    Slow market states of ``CausalDynamicGNN(memory_periods=...)`` and their update schedule.

    Level l holds one state per market and is refreshed at most once every ``periods[l]``
    seconds by its own GRU cell ``model.slow_cells[l]``:

        level 0 :  s0 ← GRUCell_0(per-block state, s0)        e.g. once per minute
        level l :  sl ← GRUCell_l(s(l-1) just refreshed, sl)   e.g. once per hour

    so each level sees a sequence of snapshots of the level below taken one period apart,
    and learns with its own weights what to keep at that frequency.  No decay and no
    normalisation are applied to these states: what is kept or overwritten is decided by
    the gates alone.

    Schedule.  The refresh is event-driven: a market is refreshed at the first block in which
    its per-block state is written (LOB update or incoming message) once its period has
    elapsed.  A market that is not written has no new information, and its states stay as they
    are.  The very first refresh of a market is immediate; the following one is anticipated by
    a fixed per-market random phase, so that markets do not all tick in the same block.
    Everything depends only on the past (clock time and which markets were written).
    """

    def __init__(self, model: CausalDynamicGNN, n_markets: int, clock: DecayClock,
                 device, dtype=torch.float32, seed: int = 0):
        self.periods = model.memory_periods
        hid = model.hidden_dim
        no_decay = torch.zeros(hid, dtype=torch.float64)
        self.states = [LazyNodeState(torch.zeros(n_markets, hid, dtype=dtype, device=device), clock, no_decay)
                       for _ in self.periods]
        gen = torch.Generator().manual_seed(seed)                      # private: leaves the global RNG stream alone
        self.phase = [(torch.rand(n_markets, generator=gen, dtype=torch.float64) * p).to(device) for p in self.periods]
        self.next_due = [torch.full((n_markets,), float("-inf"), dtype=torch.float64, device=device) for _ in self.periods]

    def update(self, model: CausalDynamicGNN, markets: LazyNodeState, written: torch.Tensor) -> None:
        """Refreshes the levels that are due for the markets ``written`` (unique) in this block."""
        t = markets.clock.t
        idx, x = written, None
        for lvl, (period, cell, state) in enumerate(zip(self.periods, model.slow_cells, self.states)):
            due = self.next_due[lvl][idx] <= t
            idx = idx[due]
            if idx.numel() == 0:
                return
            x = markets.lookup(idx) if lvl == 0 else x[due]
            x = cell(x, state.lookup(idx))
            state.write(idx, x)
            first = torch.isinf(self.next_due[lvl][idx])
            self.next_due[lvl][idx] = t + period - torch.where(first, self.phase[lvl][idx], torch.zeros_like(self.phase[lvl][idx]))

    def lookup(self, idx: torch.Tensor) -> list:
        return [s.lookup(idx) for s in self.states]

    def commit(self) -> None:
        for s in self.states:
            s.commit()


def _mp_sparse(mp: TransactionAttentionMP, src: LazyNodeState, dst: LazyNodeState,
               src_idx: torch.Tensor, dst_idx: torch.Tensor, z_edge: torch.Tensor) -> None:
    """``TransactionAttentionMP.forward`` restricted to the rows that appear in the edges."""
    h_s = src.lookup(src_idx)                                             # (E, D)
    h_d = dst.lookup(dst_idx)                                             # (E, D)
    uniq, inv = torch.unique(dst_idx, return_inverse=True)                # sorted, like torch.unique in the dense code

    attn_logits = mp.leaky_relu(mp.attn_proj(torch.cat([h_s, h_d, z_edge], dim=-1)))
    attn_weights = mp._edge_softmax(attn_logits, inv, uniq.numel())

    sign = mp.sign_proj(z_edge)
    msg = mp.msg_proj(h_s)
    msg = sign * msg
    msg = attn_weights * msg
    msg = mp.dropout(msg)

    agg = torch.zeros((uniq.numel(), msg.size(1)), dtype=msg.dtype, device=msg.device)
    agg = agg.scatter_add(0, inv.unsqueeze(1).expand_as(msg), msg)
    dst.write(uniq, mp.norm(mp.gru_cell(agg, dst.lookup(uniq))))


def sparse_step(
    model: CausalDynamicGNN,
    clock: DecayClock,
    markets: LazyNodeState,
    wallets: LazyNodeState,
    windows: torch.Tensor,        # (N_active, K, lob_dim)
    active_idx: torch.Tensor,     # (N_active,) unique
    edge_index: torch.Tensor,     # (2, E) [wallet_idx, market_idx]
    edge_features: torch.Tensor,  # (E, edge_dim)
    dt: float,                    # seconds since the previous block
    predict_idx: Optional[torch.Tensor] = None,
    memory: Optional["HierarchicalMemory"] = None,
) -> torch.Tensor:
    """One block of ``CausalDynamicGNN.forward`` on lazy states.  Returns the logits of ``predict_idx``."""
    # Step 0: continuous-time decay (lazy).  Only the 'learned' mode feeds a rate to the clock;
    # 'multiscale' states use elapsed seconds and their own fixed rates; 'none' never decays.
    clock.advance(model.decay_rate if model.decay_mode == "learned" else None, dt)

    # Step 1: LOB update
    if windows.size(0) > 0:
        lu = model.lob_update
        delta = lu.proj(lu.encode(windows))
        markets.write(active_idx, lu.norm(lu.gru_cell(delta, markets.lookup(active_idx))))

    # Step 2: message passing, wallet → market then market → wallet
    if edge_index.size(1) > 0:
        z_edge = model.mp.edge_encoder(edge_features)
        _mp_sparse(model.mp, wallets, markets, edge_index[0], edge_index[1], z_edge)
        _mp_sparse(model.mp, markets, wallets, edge_index[1], edge_index[0], z_edge)

    # Step 2b: hierarchical memory — slow states of the markets written in this block, if due
    if (memory is None) != (not model.memory_periods):
        raise ValueError("a model with memory_periods needs a HierarchicalMemory (and vice versa)")
    if memory is not None:
        written = active_idx if edge_index.size(1) == 0 else torch.unique(torch.cat([active_idx, edge_index[1]]))
        memory.update(model, markets, written)

    # Step 3: predict
    if predict_idx is None:
        predict_idx = torch.arange(markets.H.size(0), device=markets.H.device)
    feats = [markets.lookup(predict_idx)]
    if memory is not None:
        feats += memory.lookup(predict_idx)
    if model.global_state:
        g = (markets.lookup(active_idx).mean(dim=0, keepdim=True) if active_idx.numel()
             else feats[0].new_zeros((1, feats[0].size(1))))
        feats.append(g.expand(predict_idx.numel(), -1))
    if model.predictor_extra:
        feats.append(model.extra_ctx(windows, active_idx, predict_idx))
    logits = model.predictor(feats[0] if len(feats) == 1 else torch.cat(feats, dim=1))
    if model.skip_dim:
        logits = logits + model.skip_logits(windows, active_idx, predict_idx)
    return logits
