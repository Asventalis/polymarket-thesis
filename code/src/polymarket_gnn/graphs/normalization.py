"""Graph normalization routines."""

import torch

def normalize_adjacency(adj: torch.sparse_coo_tensor, add_self_loops=True) -> torch.sparse_coo_tensor:
    """
    Applies symmetric normalized Laplacian scaling: D^{-1/2} A D^{-1/2}
    """
    N = adj.shape[0]
    indices = adj.indices()
    values = adj.values()
    
    if add_self_loops:
        loop_idx = torch.arange(N, device=indices.device)
        loop_indices = torch.stack([loop_idx, loop_idx])
        loop_values = torch.ones(N, device=values.device)
        
        indices = torch.cat([indices, loop_indices], dim=1)
        values = torch.cat([values, loop_values])
        
        # Coalesce to sum duplicate entries if self loops already existed
        adj = torch.sparse_coo_tensor(indices, values, (N, N), device=indices.device).coalesce()
        indices = adj.indices()
        values = adj.values()
        
    # Calculate degree
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    
    # D^{-1/2}
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # Scale values: A_ij * D_i^{-1/2} * D_j^{-1/2}
    row, col = indices[0], indices[1]
    scaled_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]
    
    return torch.sparse_coo_tensor(indices, scaled_values, (N, N), device=indices.device).coalesce()

