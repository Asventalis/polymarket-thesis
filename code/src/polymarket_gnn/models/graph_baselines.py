"""Standard Graph Neural Network baselines."""

import torch
import torch.nn as nn

class GCNBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x is [nodes, input_dim]
        # adjacency is [nodes, nodes]
        support = self.linear(x)
        out = torch.sparse.mm(adjacency, support)
        return self.activation(out)

class GraphSAGEBlock(nn.Module):
    """GraphSAGE mean-aggregation layer."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_self = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_neigh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x is [nodes, input_dim]
        # adjacency is [nodes, nodes]
        neigh_agg = torch.sparse.mm(adjacency, x)
        out = self.linear_self(x) + self.linear_neigh(neigh_agg)
        return self.activation(out)

class TemporalGCN(nn.Module):
    """
    Applies a Temporal encoder (GRU) to each node's history, 
    then applies a GCN step on the final hidden states.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Node-wise temporal encoder
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        
        # Spatial encoder
        self.gcn = GCNBlock(hidden_dim, hidden_dim)
        
        # Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, u_sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_sequence: Tensor of shape (seq_len, num_nodes, input_dim)
            adjacency: Sparse tensor of shape (num_nodes, num_nodes)
        Returns:
            Tensor of shape (num_nodes,)
        """
        _, h_n = self.gru(u_sequence)
        h = h_n[-1]
        h_gcn = self.gcn(h, adjacency)
        out = self.head(h_gcn).squeeze(-1)
        return out

class TemporalGraphSAGE(nn.Module):
    """
    Applies a Temporal encoder (GRU) to each node's history, 
    then applies a GraphSAGE step on the final hidden states.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        self.sage = GraphSAGEBlock(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, u_sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(u_sequence)
        h = h_n[-1]
        h_sage = self.sage(h, adjacency)
        out = self.head(h_sage).squeeze(-1)
        return out

