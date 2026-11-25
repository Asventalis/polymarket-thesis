import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from polymarket_gnn.data.lob_layout import FEATURE_DIM, MID_IDX


DECAY_MODES = ("learned", "none", "multiscale")
STATE_UPDATES = ("gru", "ctan")
LOB_ENCODERS = ("gru", "timefirst")
N_SERIES_DEFAULT = FEATURE_DIM + 3      # 52 LOB features + 3 window-momentum columns (OUTPUT_DIM of the buffer)


class AntiSymmetricCell(nn.Module):
    """
    State update of CTAN (Gravina et al., "Long Range Propagation on Continuous-Time Dynamic
    Graphs", Eq. 5), one forward-Euler step:

        h_new = h + eps * tanh((W - W^T - gamma * I) h + x)

    x is what the GRU cell received (the projected LOB signal, or the aggregated messages).
    The anti-symmetric matrix keeps the Jacobian of the update close to the identity, so the
    information already in h is not dissipated by the following updates; gamma > 0 keeps the
    Euler step stable.  Same call signature as nn.GRUCell: cell(x, h).
    """
    def __init__(self, hidden_dim: int, eps: float = 0.5, gamma: float = 0.1):
        super().__init__()
        self.eps, self.gamma = float(eps), float(gamma)
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        w = self.weight - self.weight.t() - self.gamma * torch.eye(self.weight.size(0), dtype=h.dtype, device=h.device)
        return h + self.eps * torch.tanh(h @ w.t() + x + self.bias)


def half_lives_to_rates(half_lives: Sequence[float], dim: int) -> torch.Tensor:
    """
    Per-dimension decay rates (1/s) for a multi-timescale state.

    The ``dim`` state dimensions are split into ``len(half_lives)`` contiguous
    groups of (almost) equal size; group g decays as exp(-rate_g * dt) with
    rate_g = ln 2 / half_life_g.  An infinite half-life means no decay.
    Example: half-lives (60, 3600, 86400, inf) on 128 dims → 32 dims each with a
    memory of about a minute, an hour, a day, and unlimited.
    """
    hl = [float(h) for h in half_lives]
    if not hl or any(h <= 0 for h in hl):
        raise ValueError(f"half-lives must be positive (use inf for no decay), got {half_lives}")
    if len(hl) > dim:
        raise ValueError(f"{len(hl)} timescales do not fit in {dim} dimensions")
    rates = torch.zeros(dim, dtype=torch.float64)
    bounds = [round(i * dim / len(hl)) for i in range(len(hl) + 1)]
    for g, h in enumerate(hl):
        rates[bounds[g]:bounds[g + 1]] = 0.0 if math.isinf(h) else math.log(2.0) / h
    return rates


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOB Update Module
# ─────────────────────────────────────────────────────────────────────────────

class TimeFirstEncoder(nn.Module):
    """
    "Time first, then space" encoder of a LOB window (N, T, lob_dim).

    The GRU encoder reads the T snapshots as T vectors of lob_dim features and condenses them step
    by step.  This encoder does the opposite: every one of the ``n_series`` time-varying channels
    (the 52 LOB features and the 3 window-momentum columns) is first read as a univariate series
    of length T and summarised on its own (time stage, weights shared across channels, as the
    channel-independent linear models of Zeng et al. 2023 / Nie et al. 2023), then the channel
    summaries are mixed by one self-attention layer over the channels (space stage, channels as
    tokens, as iTransformer) and projected to ``hidden_dim``.

    Time stage, per channel c of each window:
        valid rows   = rows with a positive mid price (the buffer zero-pads missing history);
        x_c          = (series - last valid value) / (std over valid rows + eps), 0 on padding;
        token_c      = W_t x_c + W_s [last value, log1p(std), fraction of valid rows] + E_c
    where W_t: T -> time_dim is shared by all channels and E_c is a learned channel embedding.
    The columns after the first ``n_series`` ones (log-odds, forecast and lookback features) are
    constant along the window and are read from the last row by a separate linear map.
    """

    def __init__(self, lob_dim: int, window: int, hidden_dim: int, n_series: int = N_SERIES_DEFAULT,
                 time_dim: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if n_series > lob_dim:
            raise ValueError(f"n_series ({n_series}) cannot exceed lob_dim ({lob_dim})")
        if MID_IDX >= n_series:
            raise ValueError("the mid-price column must be one of the time-varying channels")
        self.n_series, self.window, self.time_dim = int(n_series), int(window), int(time_dim)
        self.n_ctx = int(lob_dim) - int(n_series)
        self.time_proj = nn.Linear(self.window, time_dim)
        self.scale_proj = nn.Linear(3, time_dim)
        self.channel_emb = nn.Parameter(torch.randn(self.n_series, time_dim) * 0.02)
        self.space = nn.TransformerEncoderLayer(time_dim, heads, dim_feedforward=4 * time_dim, dropout=dropout,
                                                batch_first=True, norm_first=True)
        self.out = nn.Linear(self.n_series * time_dim, hidden_dim)
        self.ctx = nn.Linear(self.n_ctx, hidden_dim) if self.n_ctx else None

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.size(1) != self.window:
            raise ValueError(f"TimeFirstEncoder was built for windows of {self.window} rows, got {windows.size(1)}")
        x = windows[:, :, :self.n_series]                                   # (N, T, C)
        valid = (windows[:, :, MID_IDX] > 0).to(x.dtype)                     # (N, T)
        n_valid = valid.sum(dim=1).clamp(min=1.0)                            # (N,)
        v = valid.unsqueeze(-1)                                              # (N, T, 1)
        # last valid row of each window (the buffer right-aligns the real rows, so it is the last
        # row whenever at least one row is real)
        last = x[:, -1, :]                                                   # (N, C)
        mean = (x * v).sum(dim=1) / n_valid.unsqueeze(-1)                    # (N, C)
        var = (((x - mean.unsqueeze(1)) ** 2) * v).sum(dim=1) / n_valid.unsqueeze(-1)
        std = torch.sqrt(var + 1e-8)
        xn = ((x - last.unsqueeze(1)) / (std.unsqueeze(1) + 1e-3)) * v       # (N, T, C), 0 on padding
        xn = xn.clamp(-20.0, 20.0)
        tok = self.time_proj(xn.transpose(1, 2))                             # (N, C, time_dim)
        scale = torch.stack([last, torch.log1p(std), (n_valid / self.window).unsqueeze(-1).expand_as(last)], dim=-1)
        tok = tok + self.scale_proj(scale) + self.channel_emb.unsqueeze(0)
        tok = self.space(tok)                                                # (N, C, time_dim)
        h = self.out(tok.reshape(tok.size(0), -1))
        if self.ctx is not None:
            h = h + self.ctx(windows[:, -1, self.n_series:])
        return h


class LOBUpdateModule(nn.Module):
    """
    Encodes a sliding window of K LOB snapshots per market into the
    global market node embedding via a residual update.

    Input:
        windows:    (N_active, K, lob_dim)  - K snapshots for each active market
        h_market:   (total_markets, hidden_dim) - full node embedding matrix on GPU
        active_idx: (N_active,) int64 - row indices in h_market to update

    Operation:
        h_last = GRU(windows)[-1]         # (N_active, hidden_dim)
        delta  = Linear(h_last)           # (N_active, hidden_dim)
        h_market[active_idx] = LayerNorm(h_market[active_idx] + delta)

    Returns: h_market (updated in place for active rows only)
    """

    def __init__(self, lob_dim: int = 55, hidden_dim: int = 128, state_update: str = "gru",
                 ctan_eps: float = 0.5, ctan_gamma: float = 0.1, encoder: str = "gru", window: int = 10,
                 n_series: int = N_SERIES_DEFAULT, time_dim: int = 32):
        super().__init__()
        if encoder not in LOB_ENCODERS:
            raise ValueError(f"encoder must be one of {LOB_ENCODERS}, got {encoder!r}")
        self.encoder = encoder
        if encoder == "gru":
            self.gru = nn.GRU(lob_dim, hidden_dim, batch_first=True)
        else:
            self.timefirst = TimeFirstEncoder(lob_dim, window, hidden_dim, n_series=n_series, time_dim=time_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if state_update == "ctan":
            # the attribute names are kept so that both state engines work unchanged: the state is
            # written as norm(gru_cell(input, state)), here with the CTAN cell and no normalisation
            self.gru_cell = AntiSymmetricCell(hidden_dim, ctan_eps, ctan_gamma)
            self.norm = nn.Identity()

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        """Window (N, K, lob_dim) -> block input (N, hidden_dim), with the configured encoder."""
        if self.encoder == "gru":
            _, h_last = self.gru(windows)   # h_last: (1, N_active, hidden_dim)
            return h_last.squeeze(0)
        return self.timefirst(windows)

    def forward(
        self,
        windows: torch.Tensor,
        h_market: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        h_last = self.encode(windows)       # (N_active, hidden_dim)

        # Learned projection of the LOB signal
        delta = self.proj(h_last)           # (N_active, hidden_dim)

        # Gated memory update (out-of-place to preserve autograd graph)
        updated_vals = self.norm(self.gru_cell(delta, h_market[active_idx]))
        h_market_new = h_market.clone()
        h_market_new[active_idx] = updated_vals

        return h_market_new


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction Attention Message Passing
# ─────────────────────────────────────────────────────────────────────────────

class TransactionAttentionMP(nn.Module):
    """
    Attention-based message passing on transaction edges.

    For each transaction edge between a source and target node:
      1. Encode raw edge features (volume, price, side, token type, etc.)
         into a dense edge embedding.
      2. Compute an attention weight from [h_source, h_target, z_edge],
         normalized via softmax over all edges pointing to the same target.
         Volume naturally influences attention through the edge embedding.
      3. Learn a sign modulation vector from the edge embedding — this captures
         how buy/sell and YES/NO directions should modulate information flow.
      4. Compute message = sign_modulation ⊙ (attention × W·h_source)
      5. Aggregate at target via scatter_add, then residual + LayerNorm.

    This module is called twice per step:
      - Wallet → Market (wallets inform markets about trading activity)
      - Market → Wallet (markets inform wallets about market state)
    The same parameters are shared for both directions (nodes are treated
    uniformly), and the same edge features are used for both.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 7, dropout: float = 0.1, state_update: str = "gru",
                 ctan_eps: float = 0.5, ctan_gamma: float = 0.1):
        super().__init__()

        # Edge feature encoder: raw (7-dim) → dense (hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention: [h_src, h_dst, z_edge] → scalar logit
        self.attn_proj = nn.Linear(3 * hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Message projection: transforms source embedding before sending
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Sign modulation: learns how edge features influence the
        # direction/sign of information flow (buy=positive, sell=negative)
        self.sign_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if state_update == "ctan":      # see LOBUpdateModule
            self.gru_cell = AntiSymmetricCell(hidden_dim, ctan_eps, ctan_gamma)
            self.norm = nn.Identity()

    @staticmethod
    def _edge_softmax(
        logits: torch.Tensor,
        target_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Numerically-stable softmax over edges grouped by target node.

        Args:
            logits:     (E, 1) attention logits
            target_idx: (E,)   int64 indices of target nodes
            num_nodes:  int    total number of target nodes

        Returns:
            weights:    (E, 1) attention weights summing to 1 per target node
        """
        # Max per target for numerical stability
        max_per_node = torch.full(
            (num_nodes, 1), -1e16, device=logits.device, dtype=logits.dtype
        )
        max_per_node.scatter_reduce_(
            0, target_idx.unsqueeze(1), logits, reduce="amax", include_self=True
        )
        logits_stable = logits - max_per_node[target_idx]

        exp_logits = logits_stable.exp()

        sum_per_node = torch.zeros(num_nodes, 1, device=logits.device, dtype=logits.dtype)
        sum_per_node.scatter_add_(0, target_idx.unsqueeze(1), exp_logits)

        return exp_logits / (sum_per_node[target_idx] + 1e-8)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
        z_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-direction message passing: src → dst.

        Args:
            h_src:      (N_src, D) source node embeddings
            h_dst:      (N_dst, D) target node embeddings
            edge_index: (2, E) int64 — row 0 = source indices, row 1 = target indices
            z_edge:     (E, D) pre-computed dense edge embeddings

        Returns:
            h_dst_new:  (N_dst, D) updated target embeddings
        """
        if edge_index.size(1) == 0:
            return h_dst

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # Gather source and target embeddings for each edge
        h_s = h_src[src_idx]  # (E, D)
        h_d = h_dst[dst_idx]  # (E, D)

        # ── Attention ─────────────────────────────────────────────────────────
        attn_input = torch.cat([h_s, h_d, z_edge], dim=-1)         # (E, 3D)
        attn_logits = self.leaky_relu(self.attn_proj(attn_input))   # (E, 1)
        attn_weights = self._edge_softmax(attn_logits, dst_idx, h_dst.size(0))  # (E, 1)

        # ── Sign modulation ───────────────────────────────────────────────────
        sign = self.sign_proj(z_edge)  # (E, D) — element-wise learned sign

        # ── Compute messages ──────────────────────────────────────────────────
        msg = self.msg_proj(h_s)       # (E, D) — projected source
        msg = sign * msg               # modulate by learned sign/direction
        msg = attn_weights * msg       # scale by attention
        msg = self.dropout(msg)

        # ── Aggregate at targets ──────────────────────────────────────────────
        agg = torch.zeros_like(h_dst)
        agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msg), msg)

        # ── GRU Gated Update ──────────────────────────────────────────────────
        # We only want to update the nodes that actually received messages.
        unique_dst = torch.unique(dst_idx)
        if len(unique_dst) > 0:
            updated_vals = self.norm(self.gru_cell(agg[unique_dst], h_dst[unique_dst]))
            h_dst_new = h_dst.clone()
            h_dst_new[unique_dst] = updated_vals
            return h_dst_new

        return h_dst


# ─────────────────────────────────────────────────────────────────────────────
# 3. Predictor Head
# ─────────────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Prediction head on the market state.
    out_dim = 3: logits for the price classes (0=DOWN, 1=CONSTANT, 2=UP).
    out_dim = 1: regression of the (scaled) signed mid-price change.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, out_dim: int = 3, in_dim: Optional[int] = None,
                 head_dim: Optional[int] = None, head_layers: int = 1):
        super().__init__()
        # default head: one hidden layer of hidden_dim // 2 units (module indices 0 and 3 are the two
        # linear maps, as before); ``head_dim`` / ``head_layers`` widen and deepen it.
        h = int(head_dim) if head_dim else hidden_dim // 2
        layers, d_in = [], (in_dim if in_dim is not None else hidden_dim)
        for _ in range(max(1, int(head_layers))):
            layers += [nn.Linear(d_in, h), nn.SiLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(h, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h_m: torch.Tensor) -> torch.Tensor:
        return self.net(h_m)  # (N, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────────────────────────────────────

class CausalDynamicGNN(nn.Module):
    """
    Causal Auto-Regressive Dynamic GNN for Polymarket price direction prediction.

    Per-step logic:
      1. LOBUpdateModule:           LOB windows → residual update of h_market
      2. TransactionAttentionMP:    edges → attention message passing
                                     (wallet→market, then market→wallet)
      3. Predictor:                 h_market → price direction logits

    The global state (h_market, h_wallet) lives OUTSIDE this module in the
    training loop as plain tensors, detached periodically for TBPTT.
    """

    def __init__(
        self,
        lob_dim:    int   = 52,
        hidden_dim: int   = 128,
        edge_dim:   int   = 7,
        dropout:    float = 0.1,
        out_dim:    int   = 3,
        decay_mode: str   = "learned",
        market_half_lives: Optional[Sequence[float]] = None,
        wallet_half_lives: Optional[Sequence[float]] = None,
        memory_periods: Optional[Sequence[float]] = None,
        skip_dim: int = 0,
        skip_extra: int = 0,
        skip_init_lookback: Optional[int] = None,
        predictor_extra: int = 0,
        global_state: bool = False,
        state_update: str = "gru",
        ctan_eps: float = 0.5,
        ctan_gamma: float = 0.1,
        lob_encoder: str = "gru",
        window: int = 10,
        n_series: int = N_SERIES_DEFAULT,
        time_dim: int = 32,
        head_dim: Optional[int] = None,
        head_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if state_update not in STATE_UPDATES:
            raise ValueError(f"state_update must be one of {STATE_UPDATES}, got {state_update!r}")
        if skip_dim % 3:
            raise ValueError("skip_dim is 3 per lookback: [scaled change, sign, available]")
        self.state_update = state_update
        # Direct path from the long-lookback price features to the output (after NAVIS, Feldman et
        # al., ICLR 2026: a model should contain the heuristic it has to beat as a special case).
        # The last ``skip_dim`` columns of the LOB windows are [change, sign, available] per
        # lookback; a linear map of those plus |sign| can output "opposite of the past move,
        # CONSTANT if the price did not move", i.e. Mean Reversion, without going through the
        # recurrent state.  Zero-initialised: at the start the model is the one without the path.
        self.skip_dim = int(skip_dim)
        # ``skip_extra`` more columns, placed just before the lookback block in the windows, also enter
        # the direct path as they are (e.g. the log-odds price features).
        self.skip_extra = int(skip_extra) if skip_dim else 0
        # ``predictor_extra``: the last that many window columns before the lookback block (the extra
        # block, e.g. the hand-made features) are ALSO read by the predictor MLP next to the states,
        # so that nonlinear interactions between them do not have to survive the recurrent update.
        self.predictor_extra = int(predictor_extra)
        if self.predictor_extra and self.skip_dim and self.predictor_extra > self.skip_extra:
            raise ValueError("predictor_extra cannot exceed the extra columns before the lookback block")
        # Global state: mean of the per-block states of the markets active in the block, read by
        # the predictor next to the state of the market (a market-wide context, as NAVIS' virtual
        # global state).
        self.global_state = bool(global_state)

        # Hierarchical market memory (optional).  On top of the per-block state, one extra
        # market state per period (seconds, increasing, e.g. 60 and 3600), each updated by its
        # OWN GRU cell at its own frequency: level 0 reads the per-block state once per period,
        # level l reads level l-1.  The predictor sees all the states side by side.  Message
        # passing and the LOB update only ever touch the per-block state.  The states and the
        # update schedule live in ``sparse_state.HierarchicalMemory``.
        self.memory_periods = tuple(float(p) for p in (memory_periods or ()))
        if any(p <= 0 for p in self.memory_periods) or list(self.memory_periods) != sorted(set(self.memory_periods)):
            raise ValueError(f"memory_periods must be positive and strictly increasing, got {memory_periods}")

        self.lob_update = LOBUpdateModule(lob_dim, hidden_dim, state_update, ctan_eps, ctan_gamma,
                                          encoder=lob_encoder, window=window, n_series=n_series, time_dim=time_dim)
        self.mp         = TransactionAttentionMP(hidden_dim, edge_dim, dropout, state_update, ctan_eps, ctan_gamma)
        self.predictor  = Predictor(hidden_dim, dropout, out_dim,
                                    in_dim=hidden_dim * (1 + len(self.memory_periods) + int(self.global_state))
                                    + self.predictor_extra, head_dim=head_dim, head_layers=head_layers)
        if self.memory_periods:
            # created after the other modules, so their initialisation does not depend on this option
            self.slow_cells = nn.ModuleList(nn.GRUCell(hidden_dim, hidden_dim) for _ in self.memory_periods)
        if self.skip_dim:
            self.skip = nn.Linear(self.skip_extra + self.skip_dim + self.skip_dim // 3, out_dim)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)
            if skip_init_lookback is not None:
                # start from Mean Reversion on lookback index k instead of from zero: DOWN after an up
                # move, UP after a down move, CONSTANT if the price did not move (3-class output only)
                if out_dim != 3 or not 0 <= skip_init_lookback < self.skip_dim // 3:
                    raise ValueError("skip_init_lookback needs the 3-class output and a valid lookback index")
                a, k = 2.0, int(skip_init_lookback)
                j_sign = self.skip_extra + 3 * k + 1
                j_moved = self.skip_extra + self.skip_dim + k
                with torch.no_grad():
                    self.skip.weight[0, j_sign] = a
                    self.skip.weight[2, j_sign] = -a
                    self.skip.weight[1, j_moved] = -a
                    self.skip.bias[1] = a / 2
        
        # Continuous-time decay of the node states between events.
        #   learned    : one learnable scalar rate shared by all dimensions, markets and wallets
        #                (the original behaviour; trained runs end at ~0.04/s, a half-life of ~17 s)
        #   none       : no decay, memory is managed by the GRU gates alone
        #   multiscale : fixed per-dimension rates, groups of dimensions with different half-lives,
        #                so the state holds short- and long-term memory side by side
        if decay_mode not in DECAY_MODES:
            raise ValueError(f"decay_mode must be one of {DECAY_MODES}, got {decay_mode!r}")
        self.decay_mode = decay_mode
        self.decay_rate = nn.Parameter(torch.tensor([0.001]))     # used only by 'learned'; kept for checkpoint compatibility
        if decay_mode == "multiscale":
            if market_half_lives is None:
                raise ValueError("decay_mode='multiscale' needs market_half_lives")
            m_rates = half_lives_to_rates(market_half_lives, hidden_dim)
            w_rates = half_lives_to_rates(wallet_half_lives if wallet_half_lives is not None else market_half_lives, hidden_dim)
        else:
            m_rates = w_rates = torch.zeros(hidden_dim, dtype=torch.float64)
        # not persistent: checkpoints stay interchangeable between modes; the rates are a function of the run config
        self.register_buffer("market_decay_rates", m_rates, persistent=False)
        self.register_buffer("wallet_decay_rates", w_rates, persistent=False)

    def forward(
        self,
        windows:       torch.Tensor,   # (N_active, K, lob_dim)
        active_idx:    torch.Tensor,   # (N_active,) int64
        h_market:      torch.Tensor,   # (N_markets, hidden_dim)
        h_wallet:      torch.Tensor,   # (N_wallets, hidden_dim)
        edge_index:    torch.Tensor,   # (2, E) int64 [wallet_idx, market_idx]
        edge_features: torch.Tensor,   # (E, edge_dim)
        delta_t_market: torch.Tensor = None, # scalar or (N_markets, 1): seconds elapsed
        delta_t_wallet: torch.Tensor = None, # scalar or (N_wallets, 1): seconds elapsed
        predict_idx:   torch.Tensor = None,  # optional (P,) int64: markets to score
    ) -> tuple:
        """
        Executes one block step of the causal graph.

        delta_t: the training loop passes a scalar (seconds since the previous
        block) that is applied to every node.  Because exp(-r*a)*exp(-r*b) =
        exp(-r*(a+b)), applying the decay at every block to all nodes is
        mathematically identical to a per-node "seconds since own last event"
        decay applied lazily at the node's next event; the only difference is
        the state read for nodes that have no event, which is never used by the
        loss.

        Returns:
            logits:   (N_markets, 3) if predict_idx is None, else (P, 3)
            h_market: (N_markets, hidden_dim) — updated
            h_wallet: (N_wallets, hidden_dim) — updated
        """
        if self.memory_periods or self.skip_dim or self.global_state or self.predictor_extra:
            raise NotImplementedError("hierarchical memory, the lookback skip path, the global state and the "
                                      "predictor extras are implemented by the sparse engine (sparse_state.sparse_step)")
        h_market, h_wallet = self.advance(windows, active_idx, h_market, h_wallet, edge_index, edge_features,
                                          delta_t_market, delta_t_wallet)

        # ── Step 3: Predict ───────────────────────────────────────────────────
        # Scoring all ~175k markets every block is wasted work when the loss
        # only uses the active ones; callers can restrict to predict_idx.
        if predict_idx is not None:
            logits = self.predictor(h_market[predict_idx])
        else:
            logits = self.predictor(h_market)

        return logits, h_market, h_wallet

    @staticmethod
    def _window_positions(active_idx: torch.Tensor, predict_idx: torch.Tensor) -> torch.Tensor:
        """Position in ``active_idx`` of every market of ``predict_idx`` (all of them must have a window)."""
        order = torch.argsort(active_idx)
        pos = order[torch.searchsorted(active_idx[order], predict_idx).clamp(max=active_idx.numel() - 1)]
        if not torch.equal(active_idx[pos], predict_idx):
            raise ValueError("the direct path / predictor extras need every predicted market to have a LOB window in the block")
        return pos

    def extra_ctx(self, windows: torch.Tensor, active_idx: torch.Tensor, predict_idx: torch.Tensor) -> torch.Tensor:
        """The ``predictor_extra`` columns before the lookback block, last window row, for ``predict_idx``."""
        if predict_idx.numel() == 0:
            return windows.new_zeros((0, self.predictor_extra))
        pos = self._window_positions(active_idx, predict_idx)
        stop = windows.size(2) - self.skip_dim
        return windows[pos, -1, stop - self.predictor_extra:stop]

    def skip_logits(self, windows: torch.Tensor, active_idx: torch.Tensor, predict_idx: torch.Tensor) -> torch.Tensor:
        """Output of the direct path for the markets in ``predict_idx`` (all of them must be in ``active_idx``)."""
        if predict_idx.numel() == 0:
            return windows.new_zeros((0, self.skip.out_features))
        pos = self._window_positions(active_idx, predict_idx)
        ctx = windows[pos, -1, -(self.skip_dim + self.skip_extra):]            # [extra, lookback triples]
        moved = ctx[:, self.skip_extra:].reshape(ctx.size(0), -1, 3)[:, :, 1].abs()
        return self.skip(torch.cat([ctx, moved], dim=1))

    def advance(self, windows, active_idx, h_market, h_wallet, edge_index, edge_features,
                delta_t_market=None, delta_t_wallet=None) -> tuple:
        """Steps 0–2 of ``forward`` (decay, LOB update, message passing): the per-block states after this block."""
        # ── Step 0: Continuous-Time Decay ─────────────────────────────────────
        if self.decay_mode == "learned":
            if delta_t_market is not None:
                decay_m = torch.exp(-torch.abs(self.decay_rate) * delta_t_market)
                h_market = h_market * decay_m

            if delta_t_wallet is not None:
                decay_w = torch.exp(-torch.abs(self.decay_rate) * delta_t_wallet)
                h_wallet = h_wallet * decay_w
        elif self.decay_mode == "multiscale":
            if delta_t_market is not None:
                h_market = h_market * torch.exp(-self.market_decay_rates * delta_t_market).to(h_market.dtype)
            if delta_t_wallet is not None:
                h_wallet = h_wallet * torch.exp(-self.wallet_decay_rates * delta_t_wallet).to(h_wallet.dtype)
        # 'none': the states are left untouched between events

        # ── Step 1: LOB Update ────────────────────────────────────────────────
        if windows.size(0) > 0:
            h_market = self.lob_update(windows, h_market, active_idx)

        # ── Step 2: Message Passing ───────────────────────────────────────────
        if edge_index.size(1) > 0:
            # Encode edge features once (shared by both directions)
            z_edge = self.mp.edge_encoder(edge_features)  # (E, hidden_dim)

            # Wallet → Market: edge_index is already [wallet_idx, market_idx]
            h_market = self.mp(h_wallet, h_market, edge_index, z_edge)

            # Market → Wallet: flip edge_index to [market_idx, wallet_idx]
            reverse_edge_index = torch.stack([edge_index[1], edge_index[0]])
            h_wallet = self.mp(h_market, h_wallet, reverse_edge_index, z_edge)

        return h_market, h_wallet
