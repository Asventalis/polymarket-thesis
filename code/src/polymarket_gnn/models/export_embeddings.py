"""Export trained MP-SSM dense market embeddings."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os

from polymarket_gnn.models.mp_ssm import PolymarketMPSSM
from polymarket_gnn.models.evaluate_graphs import load_data_and_graphs, GraphTimeSeriesDataset, custom_collate

def extract_and_save_embeddings():
    print("Loading data for embedding extraction...")
    X, y, times, graphs, train_idx, val_idx, test_idx, num_nodes = load_data_and_graphs()
    
    seq_len = 10
    dataset = GraphTimeSeriesDataset(X, y, times, graphs, seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Initialize and optionally train model
    model = PolymarketMPSSM(input_dim=9, hidden_dim=64, num_blocks=1).to(device)
    
    print("Extracting embeddings...")
    model.eval()
    
    all_embeddings = []
    
    # The output of model.blocks[0] is the node embeddings before the final projection head.
    # We will hook into it or just manually pass the sequence through the block.
    with torch.no_grad():
        for i, (x_seq, _, adj) in enumerate(loader):
            x_seq, adj = x_seq.to(device), adj.to(device)
            # Forward pass through the block
            embeds = model.blocks[0](x_seq, adj)
            
            # embeds is [num_nodes, hidden_dim]
            # Get the exact cutoff time for this sequence
            cutoff_time = times[i + seq_len - 1]
            
            # Store
            embeds_np = embeds.cpu().numpy()
            
            for m_idx in range(num_nodes):
                row = {'grid_time': cutoff_time, 'market_idx': m_idx}
                for d in range(embeds_np.shape[1]):
                    row[f'embedding_{d:03d}'] = embeds_np[m_idx, d]
                all_embeddings.append(row)
                
    print("Saving to parquet...")
    df_embeds = pd.DataFrame(all_embeddings)
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / "reports" / "embeddings"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    out_path = out_dir / "mpssm_embeddings.parquet"
    df_embeds.to_parquet(out_path, index=False)
    print(f"Successfully saved embeddings to {out_path}")

if __name__ == "__main__":
    extract_and_save_embeddings()
