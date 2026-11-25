import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from dataclasses import dataclass
from typing import Dict, Optional
import os

@dataclass
class HetBiGNNConfig:
    num_wallets: int = 646394
    num_markets: int = 175292
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    edge_dim: int = 4  # [price, size, fee_rate, effect_tensor]
    heads: int = 4
    # Path to the offline computed LLM features
    market_features_path: str = "data/derived/market_features.pt"

class HetBiGNN(nn.Module):
    def __init__(self, config: HetBiGNNConfig):
        super().__init__()
        self.config = config
        
        # 1. Wallet Embeddings (Fully trainable from scratch, initialized to zero or random)
        self.wallet_emb = nn.Embedding(config.num_wallets, config.hidden_dim)
        # Initialize wallets strictly to zero, letting them learn entirely from incoming graph messages
        nn.init.zeros_(self.wallet_emb.weight)
        
        # 2. Market Embeddings (Frozen Offline LLM + Trainable Adapter)
        if os.path.exists(config.market_features_path):
            print(f"Loading Frozen LLM Market Features from {config.market_features_path}...")
            # Load the pre-computed static tensor (e.g. shape [175292, 390])
            market_tensor = torch.load(config.market_features_path)
            llm_dim = market_tensor.shape[1]
            
            # Lock the massive LLM weights to drastically reduce GPU memory and compute
            self.market_emb = nn.Embedding.from_pretrained(market_tensor, freeze=True)
            
            # Trainable linear adapter to map the frozen 390D semantic space into the 64D GNN hidden space
            self.market_adapter = nn.Linear(llm_dim, config.hidden_dim)
        else:
            print("WARNING: market_features.pt not found. Falling back to random embeddings.")
            self.market_emb = nn.Embedding(config.num_markets, config.hidden_dim)
            self.market_adapter = nn.Identity()
            
        # 3. Edge Feature Projection
        self.edge_proj = nn.Linear(config.edge_dim, config.hidden_dim)
        
        # 4. Message Passing Layers (GATv2)
        self.convs = nn.ModuleList()
        for _ in range(config.num_layers):
            conv = HeteroConv({
                ('wallet', 'transacts', 'market'): GATv2Conv(
                    in_channels=config.hidden_dim, 
                    out_channels=config.hidden_dim // config.heads, 
                    heads=config.heads,
                    edge_dim=config.hidden_dim, 
                    add_self_loops=False
                ),
                ('market', 'rev_transacts', 'wallet'): GATv2Conv(
                    in_channels=config.hidden_dim, 
                    out_channels=config.hidden_dim // config.heads, 
                    heads=config.heads,
                    edge_dim=config.hidden_dim, 
                    add_self_loops=False
                )
            }, aggr='mean')
            self.convs.append(conv)
            
        # 5. Prediction Head (MLP strictly on Market nodes)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor], edge_attr_dict: Dict[tuple, torch.Tensor]):
        """
        x_dict: dictionary containing 'wallet' and 'market' static integer indices
        """
        # Lookup static embeddings
        wallet_h = self.wallet_emb(x_dict['wallet'])
        
        # Lookup frozen LLM embeddings and pass through the trainable adapter
        market_raw = self.market_emb(x_dict['market'])
        market_h = self.market_adapter(market_raw.float())
        
        h_dict = {
            'wallet': wallet_h,
            'market': market_h
        }
        
        # Project Edge Features
        proj_edge_attr_dict = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            proj_edge_attr_dict[edge_type] = F.relu(self.edge_proj(edge_attr.float()))
            
        # Message Passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict, edge_attr_dict=proj_edge_attr_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            
        # Predict
        out = self.predictor(h_dict['market'])
        return out
