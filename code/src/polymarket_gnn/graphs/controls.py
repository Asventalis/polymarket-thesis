"""Graph placebo controls for robust benchmarking."""

import torch
import numpy as np

def create_identity_graph(num_nodes: int, device: str = None) -> torch.sparse_coo_tensor:
    """Creates a graph with only self-loops (testing market-local only info)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = torch.arange(num_nodes, device=device)
    indices = torch.stack([idx, idx])
    values = torch.ones(num_nodes, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()

def create_random_graph(num_nodes: int, density: float = 0.05, device: str = None) -> torch.sparse_coo_tensor:
    """Creates a random Erdos-Renyi graph with specified edge density."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_edges = int(num_nodes * num_nodes * density)
    
    # Sample random edges
    u = np.random.randint(0, num_nodes, size=num_edges)
    v = np.random.randint(0, num_nodes, size=num_edges)
    w = np.random.uniform(0, 1, size=num_edges)
    
    # Symmetrize
    u_sym = np.concatenate([u, v])
    v_sym = np.concatenate([v, u])
    w_sym = np.concatenate([w, w])
    
    indices = torch.tensor(np.vstack((u_sym, v_sym)), dtype=torch.long, device=device)
    values = torch.tensor(w_sym, dtype=torch.float32, device=device)
    
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
    # Apply self loops and normalize
    from polymarket_gnn.graphs.normalization import normalize_adjacency
    return normalize_adjacency(adj)

