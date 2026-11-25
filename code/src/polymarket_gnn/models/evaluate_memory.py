"""Evaluate MP-SSM across different graph memory horizons."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import argparse

from polymarket_gnn.models.mp_ssm import PolymarketMPSSM
from polymarket_gnn.graphs.projection import load_graph_snapshots
from polymarket_gnn.graphs.normalization import normalize_adjacency

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
            from polymarket_gnn.graphs.controls import create_identity_graph
            adj = create_identity_graph(num_nodes, device=self.X.device)
            
        return x_seq, y_val, adj


def custom_collate(batch):
    return batch[0]

def load_day_data(date_str, mem_length):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "derived" / "multi_day"
    
    # Check if files exist for this day
    target_file = data_dir / "targets" / f"{date_str}.parquet"
    if not target_file.exists():
        return None, None, None, None, None
        
    query = f"""
    WITH selected_markets AS (
        SELECT DISTINCT market_id
        FROM read_parquet('{data_dir}/targets/{date_str}.parquet')
        LIMIT 500
    )
    SELECT 
        o.market_id,
        o.grid_time,
        o.mid_price, o.spread, o.best_bid, o.best_ask,
        tx.volume_2s, tx.num_trades_2s, tx.buy_volume_2s, tx.sell_volume_2s, tx.unique_wallets_2s,
        t.target_10s
    FROM read_parquet('{data_dir}/orderbook/{date_str}.parquet') o
    JOIN read_parquet('{data_dir}/transactions/{date_str}.parquet') tx
      ON o.market_id = tx.market_id AND o.grid_time = tx.grid_time
    JOIN read_parquet('{data_dir}/targets/{date_str}.parquet') t
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
    
    edges_path = data_dir / f"edges_{mem_length}s" / f"{date_str}.parquet"
    if edges_path.exists():
        raw_graphs = load_graph_snapshots(edges_path, market_ids, k_neighbors=10)
        norm_graphs = {pd.Timestamp(t, tz="UTC"): normalize_adjacency(adj) for t, adj in raw_graphs.items()}
    else:
        norm_graphs = {}
        
    return X_tensor, y_tensor, times, norm_graphs, num_nodes

def train_and_eval(model, train_loader, val_loader, device, epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    
    best_val_ic = -1
    
    for epoch in range(epochs):
        model.train()
        for x, y, adj in train_loader:
            x, y, adj = x.to(device), y.to(device), adj.to(device)
            optimizer.zero_grad()
            pred = model(x, adj)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y, adj in val_loader:
                x, y, adj = x.to(device), y.to(device), adj.to(device)
                pred = model(x, adj)
                all_preds.append(pred.cpu().numpy())
                all_true.append(y.cpu().numpy())
                
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        mae = np.mean(np.abs(all_preds - all_true))
        
        ics = []
        for i in range(len(all_preds)):
            pred_t, true_t = all_preds[i], all_true[i]
            if np.std(pred_t) > 1e-6 and np.std(true_t) > 1e-6:
                corr, _ = spearmanr(pred_t, true_t)
                ics.append(corr)
        mean_ic = np.nanmean(ics) if len(ics) > 0 else 0.0
        best_val_ic = max(best_val_ic, mean_ic)
        
    return mae, best_val_ic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-date", type=str, default="2026-03-06")
    parser.add_argument("--val-date", type=str, default="2026-03-07")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for mem in [2, 10, 20, 40]:
        print(f"\n=============================")
        print(f"Evaluating {mem}s Memory Length")
        print(f"=============================")
        
        X_train, y_train, times_train, graphs_train, num_nodes = load_day_data(args.train_date, mem)
        X_val, y_val, times_val, graphs_val, _ = load_day_data(args.val_date, mem)
        
        if X_train is None or X_val is None:
            print("Data for requested dates not fully generated yet.")
            continue
            
        train_dataset = GraphTimeSeriesDataset(X_train, y_train, times_train, graphs_train)
        val_dataset = GraphTimeSeriesDataset(X_val, y_val, times_val, graphs_val)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
        
        model = PolymarketMPSSM(input_dim=9, hidden_dim=32, num_blocks=1).to(device)
        mae, ic = train_and_eval(model, train_loader, val_loader, device, epochs=1)
        
        print(f"Result for {mem}s - MAE: {mae:.6f} | IC: {ic:.4f}")
        results[mem] = {"MAE": mae, "IC": ic}
        
    print("\nFinal Memory Ablation Results:")
    for mem, metrics in results.items():
        print(f"{mem}s Memory - MAE: {metrics['MAE']:.6f}, IC: {metrics['IC']:.4f}")

if __name__ == "__main__":
    main()
