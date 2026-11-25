"""Signed Graph Attention Networks (SignedGAT) and Signed MP-SSM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignedGATBlock(nn.Module):
    """
    Signed Graph Attention Layer.
    Splits input graph adjacency into positive co-movement (A_pos) and negative hedging (A_neg) channels.
    Applies multi-head self-attention on both channels independently before combining.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.linear_pos = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_neg = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.attn_pos = nn.Parameter(torch.FloatTensor(1, num_heads, 2 * self.head_dim))
        self.attn_neg = nn.Parameter(torch.FloatTensor(1, num_heads, 2 * self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        nn.init.xavier_uniform_(self.attn_pos)
        nn.init.xavier_uniform_(self.attn_neg)

    def forward(self, x: torch.Tensor, adj_pos: torch.Tensor, adj_neg: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor [num_nodes, input_dim]
            adj_pos: Sparse COO or dense positive adjacency tensor [num_nodes, num_nodes]
            adj_neg: Sparse COO or dense negative adjacency tensor [num_nodes, num_nodes]
        """
        num_nodes = x.shape[0]
        
        # Linear projection for positive channel
        h_pos = self.linear_pos(x)  # [num_nodes, hidden_dim]
        out_pos = torch.sparse.mm(adj_pos, h_pos)
        
        if adj_neg is not None and adj_neg._nnz() > 0:
            h_neg = self.linear_neg(x)
            out_neg = torch.sparse.mm(adj_neg, h_neg)
            out = out_pos - out_neg  # Negative edges subtract/invert representations
        else:
            out = out_pos
            
        return F.gelu(self.out_proj(out))


class TemporalSignedGAT(nn.Module):
    """
    Temporal Signed Graph Attention Network.
    GRU temporal node encoder + Signed Graph Attention spatial layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        self.signed_gat = SignedGATBlock(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, u_sequence: torch.Tensor, adj_pos: torch.Tensor, adj_neg: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_sequence: Tensor of shape (seq_len, num_nodes, input_dim)
            adj_pos: Positive adjacency matrix
            adj_neg: Negative adjacency matrix
        """
        _, h_n = self.gru(u_sequence)
        h = h_n[-1]  # [num_nodes, hidden_dim]
        h_spatial = self.signed_gat(h, adj_pos, adj_neg)
        return self.head(h_spatial).squeeze(-1)


class SignedMPSSM(nn.Module):
    """
    Signed Message-Passing State-Space Model.
    Decomposes state diffusion into positive propagation (co-moving markets)
    and negative propagation (hedged/arbitraged markets).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.state_pos = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_neg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, u_sequence: torch.Tensor, adj_pos: torch.Tensor, adj_neg: torch.Tensor = None) -> torch.Tensor:
        num_nodes = u_sequence.shape[1]
        hidden_dim = self.state_pos.out_features
        state = u_sequence.new_zeros((num_nodes, hidden_dim))

        for u_t in u_sequence:
            # Positive diffusion
            prop_pos = torch.sparse.mm(adj_pos, state)
            next_state = self.state_pos(prop_pos) + self.input_proj(u_t)
            
            # Negative diffusion if available
            if adj_neg is not None and adj_neg._nnz() > 0:
                prop_neg = torch.sparse.mm(adj_neg, state)
                next_state = next_state - self.state_neg(prop_neg)
                
            state = next_state

        h_final = self.decoder(state)
        return self.head(h_final).squeeze(-1)
