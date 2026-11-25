"""Message-Passing State-Space Model (MP-SSM)."""

import torch
import torch.nn as nn

class MPSSMBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        u_sequence: torch.Tensor,   # [time, nodes, input_dim]
        adjacency: torch.Tensor,    # sparse [nodes, nodes]
    ) -> torch.Tensor:
        num_nodes = u_sequence.shape[1]
        hidden_dim = self.state_projection.out_features
        state = u_sequence.new_zeros((num_nodes, hidden_dim))

        for u_t in u_sequence:
            # Linear diffusion
            propagated = torch.sparse.mm(adjacency, state)
            state = (
                self.state_projection(propagated)
                + self.input_projection(u_t)
            )

        # Nonlinear readout
        return self.decoder(state)

class PolymarketMPSSM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MPSSMBlock(
                input_dim=input_dim if i == 0 else hidden_dim, 
                hidden_dim=hidden_dim
            ) for i in range(num_blocks)
        ])
        # Predict scalar logit return
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, u_sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_sequence: Tensor of shape (seq_len, num_nodes, input_dim)
            adjacency: Sparse tensor of shape (num_nodes, num_nodes)
        Returns:
            Tensor of shape (num_nodes,)
        """
        x = u_sequence
        for block in self.blocks:
            x_out = block(x, adjacency)
            # For stacking multiple blocks, the output sequence is just the final state broadcasted?
            # Or we can just use the final state. Since we predict one step, 1 block is easiest.
            # If multi-block, we just treat the last block's output as the hidden features.
            # Here we just pass the final state if it's the last block.
            
            # Since the block processes the sequence and returns the FINAL state, 
            # if we have multiple blocks, we can't just pass `x_out` (which is [nodes, hidden_dim])
            # as a sequence to the next block unless we repeat it.
            # For simplicity, we just use 1 block for now.
            x = x_out 
            break
            
        out = self.head(x).squeeze(-1)
        return out
