"""Statistically rigorous multi-day evaluation of GNNs and MP-SSM with chronological splits."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import time

from polymarket_gnn.models.mp_ssm import PolymarketMPSSM
from polymarket_gnn.models.graph_baselines import TemporalGCN, TemporalGraphSAGE
from polymarket_gnn.models.signed_gat import TemporalSignedGAT, SignedMPSSM
from polymarket_gnn.graphs.projection import load_graph_snapshots, load_signed_graph_snapshots
from polymarket_gnn.graphs.normalization import normalize_adjacency
from polymarket_gnn.graphs.controls import create_identity_graph, create_random_graph

class GraphTimeSeriesDataset(Dataset):
    def __init__(self, X, y, grid_times, time_to_adj, seq_len=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.grid_times = grid_times
        self.time_to_adj = time_to_adj
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len - 1
        x_seq = self.X[start_idx:end_idx + 1]
        y_val = self.y[end_idx]
        cutoff_time = self.grid_times[end_idx]
        adj = self.time_to_adj.get(cutoff_time, None)
        if adj is None:
            num_nodes = self.X.shape[1]
            adj = create_identity_graph(num_nodes, device=self.X.device)
        return x_seq, y_val, adj

class SignedGraphTimeSeriesDataset(Dataset):
    def __init__(self, X, y, grid_times, time_to_adj_pos, time_to_adj_neg, seq_len=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.grid_times = grid_times
        self.time_to_adj_pos = time_to_adj_pos
        self.time_to_adj_neg = time_to_adj_neg
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len - 1
        x_seq = self.X[start_idx:end_idx + 1]
        y_val = self.y[end_idx]
        cutoff_time = self.grid_times[end_idx]
        
        adj_pos = self.time_to_adj_pos.get(cutoff_time, None)
        adj_neg = self.time_to_adj_neg.get(cutoff_time, None)
        
        if adj_pos is None:
            num_nodes = self.X.shape[1]
            adj_pos = create_identity_graph(num_nodes, device=self.X.device)
            
        return x_seq, y_val, adj_pos, adj_neg

def custom_collate(batch):
    return batch[0]

def signed_collate(batch):
    return batch[0]

def load_multiday_dataset(date_str="2026-03-06", num_markets=200, hours=4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "derived" / "multi_day"
    unified_file = data_dir / "unified" / f"{date_str}.parquet"
    
    total_timestamps = hours * 2 * 1800  # 2s grid steps for total hours
    
    print(f"Loading Fast Multi-Day dataset ({hours}h Train / {hours}h Val = {total_timestamps} steps) via Sequential Parquet IO...")
    start_t = time.time()
    
    query = f"""
    WITH selected_markets AS (
        SELECT DISTINCT market_id
        FROM read_parquet('{unified_file}')
        LIMIT {num_markets}
    ),
    selected_times AS (
        SELECT DISTINCT grid_time
        FROM read_parquet('{unified_file}')
        ORDER BY grid_time ASC
        LIMIT {total_timestamps}
    )
    SELECT 
        u.grid_time,
        u.market_id,
        u.mid_price, u.spread, u.best_bid, u.best_ask,
        u.volume_2s, u.num_trades_2s, u.buy_volume_2s, u.sell_volume_2s, u.unique_wallets_2s,
        u.target_10s
    FROM read_parquet('{unified_file}') u
    JOIN selected_markets sm ON u.market_id = sm.market_id
    JOIN selected_times st ON u.grid_time = st.grid_time
    ORDER BY u.grid_time ASC, u.market_id ASC
    """
    
    con = duckdb.connect()
    df = con.execute(query).df()
    df['grid_time'] = pd.to_datetime(df['grid_time'], utc=True)
    
    io_elapsed = time.time() - start_t
    print(f"Loaded sequential data in {io_elapsed*1000:.2f} ms!")
    
    features = [
        'mid_price', 'spread', 'best_bid', 'best_ask',
        'volume_2s', 'num_trades_2s', 'buy_volume_2s', 
        'sell_volume_2s', 'unique_wallets_2s'
    ]
    df[features] = df[features].fillna(0)
    
    market_ids = sorted(df['market_id'].unique().tolist())
    num_nodes = len(market_ids)
    times = np.sort(df['grid_time'].unique())
    num_times = len(times)
    
    print(f"Extracted {num_times:,} sequential timestamps across {num_nodes:,} markets.")
    
    X_tensor = np.zeros((num_times, num_nodes, len(features)))
    y_tensor = np.zeros((num_times, num_nodes))
    
    market_to_idx = {m: i for i, m in enumerate(market_ids)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    df['t_idx'] = df['grid_time'].map(time_to_idx)
    df['m_idx'] = df['market_id'].map(market_to_idx)
    
    X_tensor[df['t_idx'].values, df['m_idx'].values, :] = df[features].values
    y_tensor[df['t_idx'].values, df['m_idx'].values] = df['target_10s'].values
    
    X_flat = X_tensor.reshape(-1, len(features))
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_tensor = X_flat_scaled.reshape(num_times, num_nodes, len(features))
    
    # Chronological Split: First half = Train ({hours}h), Second half = Val ({hours}h), TEST SET LOCKED
    half_idx = num_times // 2
    train_t_idx = np.arange(0, half_idx)
    val_t_idx = np.arange(half_idx, num_times)
    test_t_idx = np.array([])  # Test Set Locked
    
    print(f"Chronological Split -> Train: {len(train_t_idx):,} steps ({hours}h Scale) | Val: {len(val_t_idx):,} steps ({hours}h Scale) | TEST SET LOCKED")
    
    edges_path = data_dir / "edges_2s" / f"{date_str}.parquet"
    if not edges_path.exists():
        edges_path = data_dir / "edges" / f"{date_str}.parquet"
        
    return X_tensor, y_tensor, times, edges_path, market_ids, train_t_idx, val_t_idx, test_t_idx, num_nodes


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    non_zero = y_true != 0
    dir_acc = np.mean(np.sign(y_true[non_zero]) == np.sign(y_pred[non_zero])) if np.sum(non_zero) > 0 else 0.0
    return mae, rmse, dir_acc

def evaluate_models_multiday(hours: int = 2, eval_test: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Fast {hours}h Train / {hours}h Validation Evaluation on GPU: {device} (TEST SET STRICTLY LOCKED)")
    
    X, y, times, edges_path, market_ids, train_idx, val_idx, _, num_nodes = load_multiday_dataset("2026-03-06", num_markets=200, hours=hours, device=device)
    seq_len = 10
    
    print("Building CPU graph snapshots...")
    raw_g_pos, raw_g_neg = load_signed_graph_snapshots(edges_path, market_ids, k_neighbors=10, device='cpu')
    norm_g_pos = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g_pos.items() if pd.Timestamp(t, tz="UTC") in times}
    norm_g_neg = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g_neg.items() if adj is not None and pd.Timestamp(t, tz="UTC") in times}
    
    raw_g_top10 = load_graph_snapshots(edges_path, market_ids, k_neighbors=10, device='cpu')
    norm_g_top10 = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g_top10.items() if pd.Timestamp(t, tz="UTC") in times}
    
    del raw_g_pos, raw_g_neg, raw_g_top10
    import gc
    gc.collect()
    
    models_to_test = {
        "Signed MP-SSM (Signed Graph)": (SignedMPSSM(input_dim=9, hidden_dim=32).to(device), "signed"),
        "Temporal SignedGAT (Signed Graph)": (TemporalSignedGAT(input_dim=9, hidden_dim=32).to(device), "signed"),
        "MP-SSM (Standard Graph)": (PolymarketMPSSM(input_dim=9, hidden_dim=32).to(device), "std"),
        "Temporal GCN": (TemporalGCN(input_dim=9, hidden_dim=32).to(device), "std"),
        "Temporal GraphSAGE": (TemporalGraphSAGE(input_dim=9, hidden_dim=32).to(device), "std"),
    }


    
    results = {}
    
    for name, (model, m_type) in models_to_test.items():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"\n-------------------------------------------------------")
        print(f"Training & Validating {name} on 1-Day Split...")
        print(f"-------------------------------------------------------")
        
        if m_type == "signed":
            train_ds = SignedGraphTimeSeriesDataset(X[train_idx], y[train_idx], times[train_idx], norm_g_pos, norm_g_neg, seq_len)
            val_ds = SignedGraphTimeSeriesDataset(X[val_idx], y[val_idx], times[val_idx], norm_g_pos, norm_g_neg, seq_len)
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=signed_collate)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=signed_collate)
        else:
            train_ds = GraphTimeSeriesDataset(X[train_idx], y[train_idx], times[train_idx], norm_g_top10, seq_len)
            val_ds = GraphTimeSeriesDataset(X[val_idx], y[val_idx], times[val_idx], norm_g_top10, seq_len)
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss()
        
        for epoch in range(2):
            model.train()
            total_loss = 0
            for batch in train_loader:
                if m_type == "signed":
                    x_b, y_b, adj_p, adj_n = batch
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    if adj_p is not None: adj_p = adj_p.to(device)
                    if adj_n is not None: adj_n = adj_n.to(device)
                    optimizer.zero_grad()
                    pred = model(x_b, adj_p, adj_n)
                else:
                    x_b, y_b, adj_b = batch
                    x_b, y_b, adj_b = x_b.to(device), y_b.to(device), adj_b.to(device)
                    optimizer.zero_grad()
                    pred = model(x_b, adj_b)
                    
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if m_type == "signed":
                        x_b, y_b, adj_p, adj_n = batch
                        x_b, y_b = x_b.to(device), y_b.to(device)
                        if adj_p is not None: adj_p = adj_p.to(device)
                        if adj_n is not None: adj_n = adj_n.to(device)
                        pred = model(x_b, adj_p, adj_n)
                    else:
                        x_b, y_b, adj_b = batch
                        x_b, y_b, adj_b = x_b.to(device), y_b.to(device), adj_b.to(device)
                        pred = model(x_b, adj_b)
                    val_loss += criterion(pred, y_b).item()
                    
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/2 | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        print(f"Evaluating {name} on VALIDATION SET...")
        model.eval()
        val_loss = 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                if m_type == "signed":
                    x_b, y_b, adj_p, adj_n = batch
                    x_b = x_b.to(device)
                    if adj_p is not None: adj_p = adj_p.to(device)
                    if adj_n is not None: adj_n = adj_n.to(device)
                    pred = model(x_b, adj_p, adj_n)
                else:
                    x_b, y_b, adj_b = batch
                    x_b, adj_b = x_b.to(device), adj_b.to(device)
                    pred = model(x_b, adj_b)
                    
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_b.numpy() if isinstance(y_b, torch.Tensor) else y_b)
                
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        
        # 1. Overall Validation Metrics
        val_mae, val_rmse, dir_all = compute_metrics(all_true, all_preds)
        
        # 2. Event-Triggered Volatility Validation Metrics (|y| > 0.001)
        active_mask = np.abs(all_true) > 0.001
        if np.sum(active_mask) > 0:
            act_mae, act_rmse, dir_act = compute_metrics(all_true[active_mask], all_preds[active_mask])
        else:
            act_mae, act_rmse, dir_act = val_mae, val_rmse, dir_all
            
        print(f"  All Steps        -> Val MAE: {val_mae:.6f} | Val Dir Acc: {dir_all*100:.2f}%")
        print(f"  Active Event Steps -> Val MAE: {act_mae:.6f} | Val Dir Acc: {dir_act*100:.2f}%")
        
        results[name] = {
            "Val Loss": val_loss / len(val_loader),
            "Val MAE (All)": val_mae,
            "Val Dir Acc (All)": f"{dir_all*100:.2f}%",
            "Val MAE (Event-Triggered)": act_mae,
            "Val Dir Acc (Event-Triggered)": f"{dir_act*100:.2f}%"
        }
        
    print("\n=======================================================================")
    print("1-Day Train / 1-Day VALIDATION SET Results (Test Set Locked):")
    print("=======================================================================")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    
    reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "experiments"
    reports_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(reports_dir / "multiday_signed_val_results.csv")
    print(f"\nSaved 1-Day Validation results to {reports_dir / 'multiday_signed_val_results.csv'}")

if __name__ == "__main__":
    evaluate_models_multiday(eval_test=False)
