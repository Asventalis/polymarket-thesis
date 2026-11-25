import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOB Update Module
# ─────────────────────────────────────────────────────────────────────────────

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

    def __init__(self, lob_dim: int = 52, hidden_dim: int = 128):
        super().__init__()
        self.gru  = nn.GRU(lob_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        windows: torch.Tensor,
        h_market: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        # GRU: process K snapshots sequentially, grab final hidden state
        _, h_last = self.gru(windows)       # h_last: (1, N_active, hidden_dim)
        h_last = h_last.squeeze(0)          # (N_active, hidden_dim)

        # Learned projection of the LOB signal
        delta = self.proj(h_last)           # (N_active, hidden_dim)

        # Residual update — only active markets are touched
        h_market[active_idx] = self.norm(h_market[active_idx] + delta)

        return h_market


# ─────────────────────────────────────────────────────────────────────────────
# 2. Message Passing: Wallet <-> Market
# ─────────────────────────────────────────────────────────────────────────────

class PersistentHeteroAggregator(nn.Module):
    """
    Performs Wallet <-> Market message passing on a signed transaction graph.
    Updates BOTH market and wallet states based on their interactions.

    Relation signs encode the directional intent of each transaction type:
        yes_buy  → positive signal for the market (buying YES = bullish)
        yes_sell → negative signal
        no_buy   → negative signal (buying NO = bearish on YES price)
        no_sell  → positive signal
    """
    RELATION_SIGNS = {
        'yes_buy':  +1.0,
        'yes_sell': -1.0,
        'no_buy':   -1.0,
        'no_sell':  +1.0,
    }

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w2m_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.m_norm   = nn.LayerNorm(hidden_dim)

        self.m2w_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_norm   = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        h_m: torch.Tensor,
        h_w: torch.Tensor,
        edge_index_dict: dict,
        edge_weight_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_m: (N_markets, hidden_dim)
        h_w: (N_wallets, hidden_dim)
        edge_index_dict: relation -> (2, E) int64  [wallet_idx, market_idx]
        edge_weight_dict: relation -> (E,) float32 [USDC amount, normalized]
        """
        # ── Wallet → Market ──────────────────────────────────────────────────
        msg_to_market = torch.zeros_like(h_m)
        h_w_proj = self.dropout(self.w2m_proj(h_w))

        for rel, sign in self.RELATION_SIGNS.items():
            if rel not in edge_index_dict or edge_index_dict[rel].size(1) == 0:
                continue
            ei = edge_index_dict[rel]
            ew = edge_weight_dict[rel]
            w_idx = ei[0]
            m_idx = ei[1]
            weighted = sign * ew.unsqueeze(1) * h_w_proj[w_idx]
            msg_to_market.scatter_add_(0, m_idx.unsqueeze(1).expand_as(weighted), weighted)

        h_m_next = self.m_norm(h_m + msg_to_market)

        # ── Market → Wallet ──────────────────────────────────────────────────
        msg_to_wallet = torch.zeros_like(h_w)
        h_m_proj = self.dropout(self.m2w_proj(h_m))

        for rel in self.RELATION_SIGNS:
            if rel not in edge_index_dict or edge_index_dict[rel].size(1) == 0:
                continue
            ei = edge_index_dict[rel]
            ew = edge_weight_dict[rel]
            w_idx = ei[0]
            m_idx = ei[1]
            weighted = ew.unsqueeze(1) * h_m_proj[m_idx]
            msg_to_wallet.scatter_add_(0, w_idx.unsqueeze(1).expand_as(weighted), weighted)

        h_w_next = self.w_norm(h_w + msg_to_wallet)

        return h_m_next, h_w_next


# ─────────────────────────────────────────────────────────────────────────────
# 3. Predictor Head
# ─────────────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """Binary classifier: h_market → logit for price-up probability."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_m: torch.Tensor) -> torch.Tensor:
        return self.net(h_m)  # (N_markets, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────────────────────────────────────

class CausalDynamicGNN(nn.Module):
    """
    Causal Auto-Regressive Dynamic GNN for Polymarket price direction prediction.

    Per-step logic:
      1. LOBUpdateModule:          new LOB windows → residual update of h_market
      2. PersistentHeteroAggregator: transactions → message passing → h_market, h_wallet
      3. Predictor:                h_market → price direction logits

    The global state (h_market, h_wallet) lives OUTSIDE this module in the
    training loop as plain tensors, detached periodically for TBPTT.
    """

    def __init__(
        self,
        lob_dim:    int   = 52,
        hidden_dim: int   = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim

        self.lob_update  = LOBUpdateModule(lob_dim, hidden_dim)
        self.aggregator  = PersistentHeteroAggregator(hidden_dim)
        self.predictor   = Predictor(hidden_dim, dropout)

    def forward(
        self,
        windows:          torch.Tensor,       # (N_active, K, lob_dim)
        active_idx:       torch.Tensor,       # (N_active,) int64
        h_market:         torch.Tensor,       # (N_markets, hidden_dim)
        h_wallet:         torch.Tensor,       # (N_wallets, hidden_dim)
        edge_index_dict:  dict,
        edge_weight_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:   (N_markets, 1)
            h_market: (N_markets, hidden_dim) - updated
            h_wallet: (N_wallets, hidden_dim) - updated
        """
        # 1. Update market embeddings from fresh LOB windows
        if windows.size(0) > 0:
            h_market = self.lob_update(windows, h_market, active_idx)

        # 2. Message passing with transaction edges
        h_market, h_wallet = self.aggregator(
            h_market, h_wallet, edge_index_dict, edge_weight_dict
        )

        # 3. Predict price direction for all markets
        logits = self.predictor(h_market)

        return logits, h_market, h_wallet
