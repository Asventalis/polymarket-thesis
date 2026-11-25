"""Graph projection utilities for loading edge lists into sparse tensors."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

def load_graph_snapshots(parquet_path, market_ids_in_order, k_neighbors=None, device=None):
    """
    Loads edge lists from parquet and builds a sequence of adjacency matrices on GPU/PyTorch.
    
    Args:
        parquet_path: Path to market_edges_2s.parquet
        market_ids_in_order: List of market_ids to map to integer node indices 0...N-1
        k_neighbors: If set, retain only the top K edges per node (sparsification).
        device: torch device (defaults to cuda if available)
        
    Returns:
        dict mapping grid_time -> torch.sparse_coo_tensor of shape [N, N] on target device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {}
        
    # Ensure edge_weight column exists (fallback to co_trades if older fixture)
    if 'edge_weight' not in df.columns:
        df['edge_weight'] = df['co_trades']
        
    market_to_idx = {m: i for i, m in enumerate(market_ids_in_order)}
    N = len(market_ids_in_order)
    
    # Filter edges to only those markets present in the requested list
    valid_mask = df['market_id_1'].isin(market_to_idx) & df['market_id_2'].isin(market_to_idx)
    df = df[valid_mask].copy()
    
    df['u'] = df['market_id_1'].map(market_to_idx)
    df['v'] = df['market_id_2'].map(market_to_idx)
    
    graphs = {}
    for t, group in df.groupby('grid_time'):
        u = group['u'].values
        v = group['v'].values
        w = group['edge_weight'].values
        
        # Symmetrize
        u_sym = np.concatenate([u, v])
        v_sym = np.concatenate([v, u])
        w_sym = np.concatenate([w, w])
        
        # Sparsification: keep top K per node
        if k_neighbors is not None:
            temp_df = pd.DataFrame({'u': u_sym, 'v': v_sym, 'w': w_sym})
            temp_df = temp_df.sort_values(['u', 'w'], ascending=[True, False])
            temp_df = temp_df.groupby('u').head(k_neighbors)
            u_sym = temp_df['u'].values
            v_sym = temp_df['v'].values
            w_sym = temp_df['w'].values
            
            # Re-symmetrize after top-k trimming to ensure undirected graph
            u_sym2 = np.concatenate([u_sym, v_sym])
            v_sym2 = np.concatenate([v_sym, u_sym])
            w_sym2 = np.concatenate([w_sym, w_sym])
            
            # Drop duplicates keeping max weight
            final_df = pd.DataFrame({'u': u_sym2, 'v': v_sym2, 'w': w_sym2})
            final_df = final_df.groupby(['u', 'v'])['w'].max().reset_index()
            u_sym = final_df['u'].values
            v_sym = final_df['v'].values
            w_sym = final_df['w'].values
            
        indices = torch.tensor(np.vstack((u_sym, v_sym)), dtype=torch.long, device=device)
        values = torch.tensor(w_sym, dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
        graphs[t] = adj
        
    return graphs

def load_signed_graph_snapshots(parquet_path, market_ids_in_order, k_neighbors=None, device=None):
    """
    Loads edge lists and builds separate positive (co-moving) and negative (hedging) adjacency matrices.
    
    Returns:
        tuple (graphs_pos, graphs_neg) mapping grid_time -> torch.sparse_coo_tensor [N, N]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {}, {}
        
    if 'edge_weight' not in df.columns:
        df['edge_weight'] = df['co_trades']
        
    market_to_idx = {m: i for i, m in enumerate(market_ids_in_order)}
    N = len(market_ids_in_order)
    
    valid_mask = df['market_id_1'].isin(market_to_idx) & df['market_id_2'].isin(market_to_idx)
    df = df[valid_mask].copy()
    
    df['u'] = df['market_id_1'].map(market_to_idx)
    df['v'] = df['market_id_2'].map(market_to_idx)
    
    # Split positive vs negative trades if side columns exist, otherwise partition by weight mean
    if 'same_side_trades' in df.columns and 'opp_side_trades' in df.columns:
        df['w_pos'] = df['same_side_trades']
        df['w_neg'] = df['opp_side_trades']
    else:
        # Fallback: top 50% weights treated as positive co-movement, lower 50% as potential hedging
        median_w = df['edge_weight'].median()
        df['w_pos'] = np.where(df['edge_weight'] >= median_w, df['edge_weight'], 0.0)
        df['w_neg'] = np.where(df['edge_weight'] < median_w, df['edge_weight'], 0.0)
        
    graphs_pos, graphs_neg = {}, {}
    for t, group in df.groupby('grid_time'):
        for channel, w_col, out_dict in [('pos', 'w_pos', graphs_pos), ('neg', 'w_neg', graphs_neg)]:
            sub = group[group[w_col] > 0]
            if sub.empty:
                indices = torch.zeros((2, 0), dtype=torch.long, device=device)
                values = torch.zeros(0, dtype=torch.float32, device=device)
                out_dict[t] = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
                continue
                
            u = sub['u'].values
            v = sub['v'].values
            w = sub[w_col].values
            
            u_sym = np.concatenate([u, v])
            v_sym = np.concatenate([v, u])
            w_sym = np.concatenate([w, w])
            
            if k_neighbors is not None and len(u_sym) > 0:
                temp_df = pd.DataFrame({'u': u_sym, 'v': v_sym, 'w': w_sym})
                temp_df = temp_df.sort_values(['u', 'w'], ascending=[True, False])
                temp_df = temp_df.groupby('u').head(k_neighbors)
                u_sym2 = np.concatenate([temp_df['u'].values, temp_df['v'].values])
                v_sym2 = np.concatenate([temp_df['v'].values, temp_df['u'].values])
                w_sym2 = np.concatenate([temp_df['w'].values, temp_df['w'].values])
                
                final_df = pd.DataFrame({'u': u_sym2, 'v': v_sym2, 'w': w_sym2})
                final_df = final_df.groupby(['u', 'v'])['w'].max().reset_index()
                u_sym, v_sym, w_sym = final_df['u'].values, final_df['v'].values, final_df['w'].values
                
            indices = torch.tensor(np.vstack((u_sym, v_sym)), dtype=torch.long, device=device)
            values = torch.tensor(w_sym, dtype=torch.float32, device=device)
            out_dict[t] = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
            
    return graphs_pos, graphs_neg


