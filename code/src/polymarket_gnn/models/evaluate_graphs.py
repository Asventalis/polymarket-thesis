"""Evaluate MP-SSM and Graph Baselines on the Dev Fixture."""

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

def custom_collate(batch):
    return batch[0]

def load_data_and_graphs():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dev_fixture_dir = project_root / "data" / "derived" / "dev_fixture"
    
    print("Loading dev fixture data via DuckDB...")
    query = f"""
    WITH selected_markets AS (
        SELECT DISTINCT market_id
        FROM read_parquet('{dev_fixture_dir}/targets_2s.parquet')
        LIMIT 500
    )
    SELECT 
        o.market_id,
        o.grid_time,
        o.mid_price, o.spread, o.best_bid, o.best_ask,
        tx.volume_2s, tx.num_trades_2s, tx.buy_volume_2s, tx.sell_volume_2s, tx.unique_wallets_2s,
        t.target_10s
    FROM read_parquet('{dev_fixture_dir}/orderbook_features_2s.parquet') o
    JOIN read_parquet('{dev_fixture_dir}/transaction_features_2s.parquet') tx
      ON o.market_id = tx.market_id AND o.grid_time = tx.grid_time
    JOIN read_parquet('{dev_fixture_dir}/targets_2s.parquet') t
      ON o.market_id = t.market_id AND o.grid_time = t.grid_time
    JOIN selected_markets sm
      ON o.market_id = sm.market_id
    WHERE t.target_10s IS NOT NULL
    ORDER BY o.grid_time ASC, o.market_id ASC
    """
    
    con = duckdb.connect()
    df = con.execute(query).df()
    df['grid_time'] = pd.to_datetime(df['grid_time'], utc=True)
    
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
    
    start_time = pd.Timestamp("2026-03-10 14:00:00", tz="UTC")
    train_end = start_time + pd.Timedelta(minutes=40)
    val_end = start_time + pd.Timedelta(minutes=50)
    
    train_t_idx = np.where(times < train_end)[0]
    val_t_idx = np.where((times >= train_end) & (times < val_end))[0]
    test_t_idx = np.where(times >= val_end)[0]
    
    edges_path = dev_fixture_dir / "market_edges_2s.parquet"
    raw_graphs = load_graph_snapshots(edges_path, market_ids, k_neighbors=10)
    
    norm_graphs = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_graphs.items()}
        
    return X_tensor, y_tensor, times, norm_graphs, train_t_idx, val_t_idx, test_t_idx, num_nodes

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

def signed_collate(batch):
    return batch[0]

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    non_zero = y_true != 0
    if np.sum(non_zero) > 0:
        dir_acc = np.mean(np.sign(y_true[non_zero]) == np.sign(y_pred[non_zero]))
    else:
        dir_acc = 0.0
        
    return mae, rmse, dir_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using GPU device for Dev Fixture Advanced Graph Pivot Evaluation: {device}")
    
    X, y, times, _, train_idx, val_idx, test_idx, num_nodes = load_data_and_graphs()
    seq_len = 10
    
    dev_fixture_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "derived" / "dev_fixture"
    edges_path = dev_fixture_dir / "market_edges_2s.parquet"
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    con = duckdb.connect()
    df_m = con.execute(f"""
        SELECT DISTINCT market_id FROM read_parquet('{dev_fixture_dir}/targets_2s.parquet') LIMIT 500
    """).df()
    market_ids = sorted(df_m['market_id'].unique().tolist())
    
    print("\nLoading Signed and Standard Graph Snapshots...")
    raw_g_pos, raw_g_neg = load_signed_graph_snapshots(edges_path, market_ids, k_neighbors=10, device=device)
    norm_g_pos = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g_pos.items()}
    norm_g_neg = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g_neg.items() if adj is not None}
    
    raw_g = load_graph_snapshots(edges_path, market_ids, k_neighbors=10, device=device)
    norm_g = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_g.items()}
    
    models = {
        "MP-SSM (Standard Graph)": (PolymarketMPSSM(input_dim=9, hidden_dim=32).to(device), "std"),
        "Temporal GCN": (TemporalGCN(input_dim=9, hidden_dim=32).to(device), "std"),
        "Temporal GraphSAGE": (TemporalGraphSAGE(input_dim=9, hidden_dim=32).to(device), "std"),
        "Temporal SignedGAT (Signed Graph)": (TemporalSignedGAT(input_dim=9, hidden_dim=32).to(device), "signed"),
        "Signed MP-SSM (Signed Graph)": (SignedMPSSM(input_dim=9, hidden_dim=32).to(device), "signed"),
    }
    
    results = {}
    
    print("\n=======================================================================")
    print("Evaluating Signed Graph Attention & Event-Triggered Volatility Filter")
    print("=======================================================================")
    
    for name, (model, m_type) in models.items():
        print(f"\n--- Training & Evaluating {name} on 1-Hour Dev Fixture ---")
        
        if m_type == "signed":
            train_ds = SignedGraphTimeSeriesDataset(X[train_idx], y[train_idx], times[train_idx], norm_g_pos, norm_g_neg, seq_len)
            val_ds = SignedGraphTimeSeriesDataset(X[val_idx], y[val_idx], times[val_idx], norm_g_pos, norm_g_neg, seq_len)
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=signed_collate)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=signed_collate)
        else:
            train_ds = GraphTimeSeriesDataset(X[train_idx], y[train_idx], times[train_idx], norm_g, seq_len)
            val_ds = GraphTimeSeriesDataset(X[val_idx], y[val_idx], times[val_idx], norm_g, seq_len)
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss()
        
        for epoch in range(3):
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
        
        # 1. Overall Evaluation (All Timestamps)
        mae_all, rmse_all, dir_all = compute_metrics(all_true, all_preds)
        
        # 2. Event-Triggered Volatility Evaluation (Only Active Movements where |y| > 0.001)
        active_mask = np.abs(all_true) > 0.001
        if np.sum(active_mask) > 0:
            mae_act, rmse_act, dir_act = compute_metrics(all_true[active_mask], all_preds[active_mask])
        else:
            mae_act, rmse_act, dir_act = mae_all, rmse_all, dir_all
            
        print(f"[{name}]")
        print(f"  All Steps        -> MAE: {mae_all:.6f} | Dir Acc: {dir_all*100:.2f}%")
        print(f"  Active Event Steps -> MAE: {mae_act:.6f} | Dir Acc: {dir_act*100:.2f}%")
        
        results[name] = {
            "MAE (All)": mae_all,
            "Dir Acc (All)": f"{dir_all*100:.2f}%",
            "MAE (Event-Triggered)": mae_act,
            "Dir Acc (Event-Triggered)": f"{dir_act*100:.2f}%"
        }
        
    print("\n=======================================================================")
    print("Dev Fixture Advanced Architecture & Event Filtering Benchmark Summary:")
    print("=======================================================================")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    
    reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "experiments"
    reports_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(reports_dir / "signed_gat_dev_fixture_results.csv")
    print(f"\nSaved dev fixture results to {reports_dir / 'signed_gat_dev_fixture_results.csv'}")

if __name__ == "__main__":
    main()


