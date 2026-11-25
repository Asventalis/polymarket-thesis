"""Train and evaluate the Local Temporal GRU baseline."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
from sklearn.preprocessing import StandardScaler
from polymarket_gnn.models.temporal import LocalGRUModel

class MarketTimeSeriesDataset(Dataset):
    def __init__(self, X, y, market_ids, seq_len=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        
        # We can only sample sequences that don't cross market boundaries
        self.valid_indices = []
        for i in range(seq_len - 1, len(X)):
            if market_ids[i] == market_ids[i - seq_len + 1]:
                self.valid_indices.append(i)
                
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        x_seq = self.X[start_idx:end_idx + 1]
        y_val = self.y[end_idx]
        return x_seq, y_val

def load_data():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dev_fixture_dir = project_root / "data" / "derived" / "dev_fixture"
    
    query = f"""
    WITH selected_markets AS (
        SELECT DISTINCT market_id
        FROM read_parquet('{dev_fixture_dir}/targets_2s.parquet')
        LIMIT 500  -- further downsample for GRU speed
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
    ORDER BY o.market_id ASC, o.grid_time ASC
    """
    
    con = duckdb.connect()
    df = con.execute(query).df()
    df['grid_time'] = pd.to_datetime(df['grid_time'], utc=True)
    return df

def main():
    print("Loading data for GRU...")
    df = load_data()
    
    features = [
        'mid_price', 'spread', 'best_bid', 'best_ask',
        'volume_2s', 'num_trades_2s', 'buy_volume_2s', 
        'sell_volume_2s', 'unique_wallets_2s'
    ]
    
    df[features] = df[features].fillna(0)
    
    # Split chronologically
    start_time = pd.Timestamp("2026-03-10 14:00:00", tz="UTC")
    train_end = start_time + pd.Timedelta(minutes=40)
    val_end = start_time + pd.Timedelta(minutes=50)
    
    train_mask = df['grid_time'] < train_end
    val_mask = (df['grid_time'] >= train_end) & (df['grid_time'] < val_end)
    test_mask = df['grid_time'] >= val_end
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_val = scaler.transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)
    
    y_train = df_train['target_10s'].values
    y_val = df_val['target_10s'].values
    y_test = df_test['target_10s'].values
    
    m_train = df_train['market_id'].values
    m_val = df_val['market_id'].values
    m_test = df_test['market_id'].values
    
    seq_len = 10  # 20 seconds lookback
    
    print("Building datasets...")
    train_dataset = MarketTimeSeriesDataset(X_train, y_train, m_train, seq_len)
    val_dataset = MarketTimeSeriesDataset(X_val, y_val, m_val, seq_len)
    test_dataset = MarketTimeSeriesDataset(X_test, y_test, m_test, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LocalGRUModel(input_dim=len(features), hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    
    print("Training GRU...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                
        print(f"Epoch {epoch+1}/3 | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
        
    print("GRU training complete.")
    
if __name__ == "__main__":
    main()
