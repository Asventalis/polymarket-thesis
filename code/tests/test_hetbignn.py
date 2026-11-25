import torch
from torch_geometric.data import HeteroData
from src.polymarket_gnn.models.hetbignn import HetBiGNN, HetBiGNNConfig

def test_hetbignn_forward():
    print("Initializing HetBiGNN Configuration...")
    # Use a small hidden_dim for fast testing
    config = HetBiGNNConfig(hidden_dim=32, num_layers=2)
    model = HetBiGNN(config)
    
    print("Building mock PyG HeteroData snapshot graph...")
    data = HeteroData()
    
    # In this specific temporal snapshot, pretend we only have 10 active wallets and 5 active markets.
    # We pass their raw global ID indices which will be used to lookup their embeddings in the static matrix.
    data['wallet'].x = torch.randint(0, config.num_wallets, (10,))
    data['market'].x = torch.randint(0, config.num_markets, (5,))
    
    # Create 20 mock transactions connecting the active wallets to active markets
    # edge_index shape: [2, num_edges]. Top row is source (wallet), bottom row is target (market).
    edge_index = torch.empty((2, 20), dtype=torch.long)
    edge_index[0, :] = torch.randint(0, 10, (20,))  # source wallet (index relative to the 10 active wallets)
    edge_index[1, :] = torch.randint(0, 5, (20,))   # target market (index relative to the 5 active markets)
    
    data['wallet', 'transacts', 'market'].edge_index = edge_index
    
    # 4D Edge features: [price, size, fee, effect]
    edge_attr = torch.rand((20, 4), dtype=torch.float)
    data['wallet', 'transacts', 'market'].edge_attr = edge_attr
    
    # Add reverse edges for bidirectional information flow
    data['market', 'rev_transacts', 'wallet'].edge_index = edge_index.flip([0])
    data['market', 'rev_transacts', 'wallet'].edge_attr = edge_attr
    
    print("Executing Mock Forward Pass...")
    out = model(
        x_dict=data.x_dict, 
        edge_index_dict=data.edge_index_dict, 
        edge_attr_dict=data.edge_attr_dict
    )
    
    print(f"Output shape: {out.shape} (Expected: [5, 1])")
    assert out.shape == (5, 1), f"Shape mismatch! Expected (5,1), got {out.shape}"
    print("✅ Forward pass successful! All PyG message-passing tensor dimensions align perfectly.")

if __name__ == "__main__":
    test_hetbignn_forward()
