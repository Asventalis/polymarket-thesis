"""Evaluate non-graph baseline models on the dev fixture."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from polymarket_gnn.models.persistence import PersistenceBaseline
from polymarket_gnn.models.tabular import RidgeBaseline, LightGBMBaseline, PyTorchRidgeBaseline, PyTorchTabularMLP
import torch

import duckdb

def load_and_merge_data():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dev_fixture_dir = project_root / "data" / "derived" / "dev_fixture"
    
    print("Loading and joining parquet files via DuckDB with downsampling...")
    # Select 2000 random markets to avoid OOM and CPU overheating
    query = f"""
    WITH selected_markets AS (
        SELECT DISTINCT market_id
        FROM read_parquet('{dev_fixture_dir}/targets_2s.parquet')
        LIMIT 2000
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
    ORDER BY o.grid_time ASC
    """
    
    con = duckdb.connect()
    print("Executing query and converting to pandas...")
    df = con.execute(query).df()
    
    # Ensure datetime dtype
    df['grid_time'] = pd.to_datetime(df['grid_time'], utc=True)
    return df

def evaluate_predictions(y_true, y_pred, df_split):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Compute cross-sectional spearman correlation per timestamp
    spearman_scores = []
    
    df_eval = pd.DataFrame({
        'grid_time': df_split['grid_time'].values,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    for _, group in df_eval.groupby('grid_time'):
        if len(group) > 2 and group['y_true'].nunique() > 1:
            corr, _ = spearmanr(group['y_true'], group['y_pred'])
            if not np.isnan(corr):
                spearman_scores.append(corr)
                
    avg_spearman = np.mean(spearman_scores) if spearman_scores else 0.0
    
    # Directional accuracy
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        dir_acc = np.mean(np.sign(y_true[non_zero_mask]) == np.sign(y_pred[non_zero_mask]))
    else:
        dir_acc = 0.0
        
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Spearman IC': avg_spearman,
        'Dir Acc': dir_acc
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running baseline evaluations using PyTorch device: {device}")
    
    df = load_and_merge_data()
    
    # Target: 10s return
    target_col = 'target_10s'
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    
    features = [
        'mid_price', 'spread', 'best_bid', 'best_ask',
        'volume_2s', 'num_trades_2s', 'buy_volume_2s', 
        'sell_volume_2s', 'unique_wallets_2s'
    ]
    
    # Split chronologically (Train: first 40 mins, Val: next 10, Test: last 10)
    start_time = pd.Timestamp("2026-03-10 14:00:00", tz="UTC")
    train_end = start_time + pd.Timedelta(minutes=40)
    val_end = start_time + pd.Timedelta(minutes=50)
    
    train_mask = df['grid_time'] < train_end
    val_mask = (df['grid_time'] >= train_end) & (df['grid_time'] < val_end)
    test_mask = df['grid_time'] >= val_end
    
    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]
    
    X_train, y_train = df_train[features].values, df_train[target_col].values
    X_val, y_val = df_val[features].values, df_val[target_col].values
    X_test, y_test = df_test[features].values, df_test[target_col].values
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Models
    models = {
        'Persistence': PersistenceBaseline(device=device),
        'PyTorch_Ridge_GPU': PyTorchRidgeBaseline(alpha=1.0, device=device),
        'PyTorch_Tabular_MLP_GPU': PyTorchTabularMLP(input_dim=len(features), device=device),
        'LightGBM': LightGBMBaseline()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == 'LightGBM':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        elif name == 'PyTorch_Tabular_MLP_GPU':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], epochs=15)
        else:
            model.fit(X_train, y_train)
            
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, df_test)
        results[name] = metrics

        
    print("\\n=== Results (10s logit return on Dev Fixture) ===")
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    
    # Save results
    reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "experiments"
    reports_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(reports_dir / "baseline_dev_fixture_results.csv")
    print(f"\\nSaved results to {reports_dir / 'baseline_dev_fixture_results.csv'}")

if __name__ == "__main__":
    main()
