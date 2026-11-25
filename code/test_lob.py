import os
import sys
import logging
import pandas as pd
import torch
from src.polymarket_gnn.data.lob_encoder import ContinuousLOBMessageEngine, HybridLOBEncoder

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def main():
    print("==================================================")
    print("1. LOADING 1 RANDOM HOUR OF DATA (2026-03-06 Hour 00)")
    print("==================================================")
    
    base_dir = "/home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/partitioned"
    ob_file = os.path.join(base_dir, "orderbook/year=2026/month=3/day=6/hour=0/data_0.parquet")
    snap_file = os.path.join(base_dir, "snapshots/year=2026/month=3/day=6/hour=0/data_0.parquet")
    
    ob_df = pd.read_parquet(ob_file)
    snap_df = pd.read_parquet(snap_file)
    
    # Target ALL markets in this hour
    top_markets = ob_df["market_id"].unique().tolist()
    
    print("\n==================================================")
    print("2. RUNNING ContinuousLOBMessageEngine (Parallelized)")
    print("==================================================")
    
    min_events = 20
    k_slices = 4
    depth_levels = 10
    num_workers = 4
    t_query_str = "2026-03-06 00:02:00"
    
    print(f"--- ENGINE PARAMETERS ---")
    print(f"Time Window:       {t_query_str}")
    print(f"Sequence Ticks (n): {min_events}")
    print(f"Time Slices (k):    {k_slices}")
    print(f"Depth Levels:       {depth_levels}")
    print(f"Workers:            {num_workers}")
    print(f"-------------------------\n")
    
    engine = ContinuousLOBMessageEngine(min_events=min_events, k_slices=k_slices, depth_levels=depth_levels, num_workers=num_workers)
    t_query = pd.to_datetime(t_query_str)
    
    t_start_loop = pd.to_datetime("2026-03-06 00:00:00")
    
    print("\n==================================================")
    print("3. EXECUTING BENCHMARK LOOP (5 Steps of 5 Minutes with 2 Workers)")
    print("==================================================")
    
    # Initialize the Neural Network outside the loop
    encoder = HybridLOBEncoder(empirical_dim=12, cnn_channels=5, seq_len=k_slices, depth_levels=depth_levels, out_dim=32)
    encoder.eval()
    
    num_workers = 2
    engine = ContinuousLOBMessageEngine(min_events=min_events, k_slices=k_slices, depth_levels=depth_levels, num_workers=num_workers)
    
    for step_idx in range(1, 6):
        import time
        import sys
        
        t_query = t_start_loop + pd.Timedelta(minutes=step_idx * 5)
        print(f"\n>>> [STEP {step_idx}/5] Advancing t_query to {t_query} <<<")
        
        step_start_time = time.time()
        
        # Calculate theoretical markets up to t_query
        ob_win = ob_df[ob_df["timestamp_received"] <= t_query]
        total_markets_in_window = ob_win["market_id"].nunique()
        event_counts = ob_win.groupby("market_id", observed=True).size()
        markets_with_min_ticks = (event_counts >= (min_events + 1)).sum()
        
        print(f"      [Analytics] Total Markets in window: {total_markets_in_window}")
        print(f"      [Analytics] Markets with >= {min_events} ticks: {markets_with_min_ticks}")
        
        # Run Engine
        results = engine.step(market_ids=top_markets, orderbook_df=ob_df, snapshots_df=snap_df, t_query=t_query)
        
        # Run Neural Network
        t_nn_start = time.time()
        step_messages_generated = 0
        step_markets_processed = 0
        for m_id, messages in results.items():
            step_messages_generated += len(messages)
            step_markets_processed += 1
            t_end, emp_tensor, spatial_tensor = messages[0]
            with torch.no_grad():
                z_hybrid = encoder(emp_tensor, spatial_tensor)
        t_nn_end = time.time()
        
        step_end_time = time.time()
        
        checkpoint_size = sys.getsizeof(engine.checkpoints) / 1024
        anchor_size = sys.getsizeof(engine.anchor_states) / 1024
        
        print(f"      [Engine Timing] TOTAL STEP EXECUTION TIME:   {step_end_time - step_start_time:.3f}s")
        print(f"      [Step Result] Tensors Extracted: {step_messages_generated}")
        print(f"      [Memory] Tracked Cursors: {len(engine.checkpoints)} ({checkpoint_size:.2f} KB) | Anchors: {len(engine.anchor_states)} ({anchor_size:.2f} KB)")

if __name__ == "__main__":
    main()
