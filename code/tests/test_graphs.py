import torch
import pytest
import numpy as np

from polymarket_gnn.graphs.normalization import normalize_adjacency
from polymarket_gnn.graphs.controls import create_identity_graph, create_random_graph

def test_normalize_adjacency():
    # Create a simple 3-node graph
    # 0 - 1 - 2
    N = 3
    indices = torch.LongTensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ])
    values = torch.FloatTensor([1.0, 1.0, 1.0, 1.0])
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    
    norm_adj = normalize_adjacency(adj, add_self_loops=True)
    norm_adj = norm_adj.coalesce()
    
    # Check shape
    assert norm_adj.shape == (N, N)
    
    # Check self loops were added
    dense = norm_adj.to_dense()
    assert dense[0, 0] > 0
    assert dense[1, 1] > 0
    assert dense[2, 2] > 0
    
    # Check symmetry
    assert torch.allclose(dense, dense.T, atol=1e-6)
    
    # Check degree scaling
    # Node 1 has degree 2 originally. With self loop it has degree 3.
    # Node 0 has degree 1 originally. With self loop it has degree 2.
    # Edge weight 0-1 should be 1.0 / sqrt(2 * 3) = 1.0 / sqrt(6)
    expected_edge = 1.0 / np.sqrt(6)
    assert abs(dense[0, 1].item() - expected_edge) < 1e-5

def test_identity_graph():
    N = 10
    adj = create_identity_graph(N)
    assert adj.shape == (N, N)
    dense = adj.to_dense()
    
    # Should be exactly identity
    assert torch.all(dense == torch.eye(N, device=dense.device))

    
def test_random_graph():
    N = 20
    adj = create_random_graph(N, density=0.1)
    assert adj.shape == (N, N)
    
    dense = adj.to_dense()
    # Check symmetry
    assert torch.allclose(dense, dense.T, atol=1e-6)
    # Check self loops exist (added by normalize_adjacency)
    diag = torch.diagonal(dense)
    assert torch.all(diag > 0)
