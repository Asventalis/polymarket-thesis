"""
graph_builder.py

Converts a block's raw transactions into edge tensors for the
attention-based message passing module.

Edge features (7 dimensions per edge):
    0: log(total_usdc + 1)   — log-scaled volume
    1: log(tokens + 1)       — log-scaled token count
    2: price                 — price per token (0-1 for prediction markets)
    3: fee_rate              — fee rate
    4: is_buy                — +1.0 if BUY, -1.0 if SELL
    5: token_direction       — +1.0 if YES token (1), -1.0 if NO token (-1)
    6: is_maker              — +1.0 if MAKER, 0.0 if TAKER

Orientation of feature 5.  ``token_direction`` in the transaction files is +1
for token 1 of the market and -1 for token 2.  The order book of a market
(hence its mid price, the LOB features and the prediction target) refers to
token 1 only in Yes/No markets and to token 2 in all the others (Up/Down,
Over/Under, team A / team B; verified on the data, see DEVELOPMENT_LOG
11/10/2026).  With ``market_orientation`` (market_id → +1 / -1) feature 5 is
multiplied by the orientation of the market and becomes "+1 if the traded token
is the one the book refers to", the same meaning in every market; ``price`` is
then the price of the book token when feature 5 is +1 and its complement when
it is -1.  Without it the raw column is used (original behaviour).

With ``skill=True`` three more features are appended (see data/wallet_skill.py),
read from the columns ``skill_wins`` / ``skill_losses`` of the transactions:
    7: shrunk skill of the wallet in (-1, 1)
    8: skill * is_buy * token_direction  (the side its track record points to)
    9: log1p(decided past trades) / 5
"""

import torch
import numpy as np
import pandas as pd
import logging

from polymarket_gnn.data.wallet_skill import (
    BEHAVIOUR_COLUMNS, N_BEHAVIOUR_FEATURES, N_SKILL_FEATURES, behaviour_edge_features, skill_edge_features,
)

logger = logging.getLogger(__name__)

EDGE_DIM = 7


class EdgeBuilder:
    """
    Pre-computes lookup structures once at init so that per-block
    edge construction is fast (~1ms instead of ~300ms).

    The two bottlenecks in the naive version were:
      1. isin(dict) — pandas rebuilds a 646k-key hash set every call
      2. [dict[x] for x in values] — Python-level loop

    Fix: pre-build a set for filtering and a pd.Series for vectorized .map().
    """

    def __init__(self, market_to_idx: dict, wallet_to_idx: dict, market_orientation: dict = None, skill: bool = False,
                 behaviour: bool = False):
        # skill=True appends the wallet track-record features (needs the book orientation)
        if skill and market_orientation is None:
            raise ValueError("skill features need market_orientation: the direction must be oriented like the mid")
        self._skill = skill
        # behaviour=True appends the wallet behaviour features (columns BEHAVIOUR_COLUMNS of the transactions)
        self._behaviour = behaviour
        self.edge_dim = EDGE_DIM + (N_SKILL_FEATURES if skill else 0) + (N_BEHAVIOUR_FEATURES if behaviour else 0)
        # market_id → +1 (book refers to token 1) / -1 (token 2); None = raw token_direction
        self._orientation = market_orientation
        # Pre-computed sets for fast isin filtering
        self._known_wallets = set(wallet_to_idx.keys())
        self._known_markets = set(market_to_idx.keys())

        # Keep dicts directly — Python dicts ARE hash tables,
        # so .map(dict) does O(n) lookups without rebuilding anything.
        # (.map(pd.Series) would rebuild a hash table from the Series every call)
        self._wallet_dict = wallet_to_idx
        self._market_dict = market_to_idx

    def __call__(
        self,
        block_txns: pd.DataFrame,
        device: torch.device = torch.device("cpu"),
    ) -> tuple:
        """
        Converts a block's transactions into edge tensors.
        """
        # Raw numpy arrays for speed
        w_raw = block_txns["wallet"].values
        m_raw = block_txns["market_id"].values
        
        w_idx_list = []
        m_idx_list = []
        valid_indices = []
        
        # Plain python loop over ~300 items is <0.1ms
        for i in range(len(w_raw)):
            w = w_raw[i]
            m = m_raw[i]
            if w in self._wallet_dict and m in self._market_dict:
                w_idx_list.append(self._wallet_dict[w])
                m_idx_list.append(self._market_dict[m])
                valid_indices.append(i)
                
        E = len(valid_indices)
        if E == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, self.edge_dim), dtype=torch.float32, device=device),
            )
            
        edge_index = torch.tensor(
            [w_idx_list, m_idx_list], dtype=torch.long, device=device
        )
        
        # Subset the dataframe using the valid indices
        txns = block_txns.iloc[valid_indices]

        # ── Edge features (vectorized numpy) ──────────────────────────────────
        vals = txns[["total_usdc", "tokens", "price", "fee_rate"]].values.astype(np.float32)
        log_volume = np.log1p(vals[:, 0])
        log_tokens = np.log1p(vals[:, 1])

        is_buy   = np.where(txns["side"].values == "BUY", 1.0, -1.0).astype(np.float32)
        tok_dir  = txns["token_direction"].values.astype(np.float32)
        if self._orientation is not None:
            get = self._orientation.get
            tok_dir = tok_dir * np.fromiter((get(m, 1) for m in txns["market_id"].values),
                                            dtype=np.float32, count=len(txns))
        is_maker = np.where(txns["role"].values == "MAKER", 1.0, 0.0).astype(np.float32)

        feats = np.column_stack([
            log_volume, log_tokens, vals[:, 2], vals[:, 3],
            is_buy, tok_dir, is_maker,
        ])
        if self._skill:
            feats = np.column_stack([feats, skill_edge_features(
                txns["skill_wins"].values, txns["skill_losses"].values, is_buy * tok_dir)])
        if self._behaviour:
            feats = np.column_stack([feats, behaviour_edge_features(*(txns[c].values for c in BEHAVIOUR_COLUMNS))])
        edge_features = torch.tensor(feats, dtype=torch.float32, device=device)

        return edge_index, edge_features


def load_market_orientation(universe_path) -> dict:
    """
    market_id → +1 if the order book refers to token 1 (Yes/No markets), -1 if it refers to
    token 2 (every other market).  Static market metadata: outcome_1 == "Yes".
    """
    df = pd.read_parquet(universe_path, columns=["market_id", "outcome_1"]).dropna(subset=["outcome_1"])
    df = df.drop_duplicates("market_id")
    is_yes = df["outcome_1"].astype(str).str.strip().str.lower().values == "yes"
    return dict(zip(df["market_id"].values, np.where(is_yes, 1, -1).tolist()))
