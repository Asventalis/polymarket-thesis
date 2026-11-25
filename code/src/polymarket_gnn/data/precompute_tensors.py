import os
import time
import datetime
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from polymarket_gnn.data.causal_dataset import CausalCoordinator

class DeltaLogger:
    def __init__(self):
        self.last_time = time.time()
        
    def log(self, msg, reset=False):
        now = time.time()
        if reset:
            self.last_time = now
        delta = now - self.last_time
        self.last_time = now
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp_str}] [+{delta:7.3f}s] {msg}", flush=True)

logger = DeltaLogger()

def fast_get_step_collation(dataset, t_idx, lob_updates, device):
    """
    Replicates the collation logic of CausalCoordinator.get_step but takes pre-calculated lob_updates.
    """
    N_m = len(dataset.market_ids)
    max_s = max([len(seq) for seq in lob_updates.values()]) if lob_updates else 0
    
    if max_s > 0:
        emp_feats = torch.zeros(N_m, max_s, 12, dtype=torch.float32, device=device)
        spatial_feats = torch.zeros(N_m, max_s, 5, dataset.lob_engine.k_slices, dataset.lob_engine.depth_levels, dtype=torch.float32, device=device)
        valid_seq_mask = torch.zeros(N_m, max_s, dtype=torch.bool, device=device)
        
        for m_id, seq in lob_updates.items():
            m_idx = dataset.market_to_idx[m_id]
            for s, (t_end, emp_t, spat_t) in enumerate(seq):
                if isinstance(emp_t, np.ndarray):
                    emp_t = torch.from_numpy(emp_t)
                if isinstance(spat_t, np.ndarray):
                    spat_t = torch.from_numpy(spat_t)
                emp_feats[m_idx, s] = emp_t.squeeze(0).to(device)
                spatial_feats[m_idx, s] = spat_t.squeeze(0).to(device)
                valid_seq_mask[m_idx, s] = True
    else:
        emp_feats = torch.zeros(N_m, 0, 12, dtype=torch.float32, device=device)
        spatial_feats = torch.zeros(N_m, 0, 5, dataset.lob_engine.k_slices, dataset.lob_engine.depth_levels, dtype=torch.float32, device=device)
        valid_seq_mask = torch.zeros(N_m, 0, dtype=torch.bool, device=device)
        
    y_raw = dataset.Y[t_idx]
    target_mask = torch.tensor(~np.isnan(y_raw), dtype=torch.bool, device=device)
    y = torch.tensor(np.nan_to_num(y_raw, nan=0.0), dtype=torch.float32, device=device)
    
    edge_index_dict = {}
    edge_weight_dict = {}
    RELATIONS = ['yes_buy', 'yes_sell', 'no_buy', 'no_sell']
    
    if hasattr(dataset, 'tx_by_step') and t_idx in dataset.tx_by_step.groups:
        tx_group = dataset.tx_by_step.get_group(t_idx)
        agg = tx_group.groupby(['relation', 'w_idx', 'm_idx'])['total_usdc'].sum().reset_index()
        
        for rel in RELATIONS:
            sub = agg[agg['relation'] == rel]
            if not sub.empty:
                ei = torch.tensor(np.stack([sub['w_idx'].values, sub['m_idx'].values]), dtype=torch.long, device=device)
                ew = torch.tensor(sub['total_usdc'].values, dtype=torch.float32, device=device)
                edge_index_dict[rel] = ei
                edge_weight_dict[rel] = ew
            else:
                edge_index_dict[rel] = torch.zeros(2, 0, dtype=torch.long, device=device)
                edge_weight_dict[rel] = torch.zeros(0, dtype=torch.float32, device=device)
    else:
        for rel in RELATIONS:
            edge_index_dict[rel] = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_weight_dict[rel] = torch.zeros(0, dtype=torch.float32, device=device)
            
    return emp_feats, spatial_feats, valid_seq_mask, edge_index_dict, edge_weight_dict, y, target_mask

def get_available_dates(base_path):
    dates = set()
    base = Path(base_path)
    if not base.exists():
        return list(dates)
    
    for year_dir in base.glob("year=*"):
        year = year_dir.name.split("=")[1]
        for month_dir in year_dir.glob("month=*"):
            month = month_dir.name.split("=")[1]
            for day_dir in month_dir.glob("day=*"):
                day = day_dir.name.split("=")[1]
                dates.add(f"{year}-{int(month):02d}-{int(day):02d}")
                
    return sorted(list(dates))

def main():
    parser = argparse.ArgumentParser(description="Precompute Tensors for Fast Training")
    parser.add_argument("--date", type=str, default="2026-03-06", help="Specific date to process (YYYY-MM-DD)")
    parser.add_argument("--all_days", action="store_true", help="Auto-discover and process all available dates")
    parser.add_argument("--start_hour", type=int, default=0)
    parser.add_argument("--end_hour", type=int, default=23)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Starting Precomputation Pipeline", reset=True)
    logger.log(f"Using device: {device}")
    
    dates_to_process = []
    if args.all_days:
        ob_path = os.path.join(args.data_dir, "derived", "partitioned", "orderbook")
        logger.log(f"Scanning {ob_path} for available dates...")
        dates_to_process = get_available_dates(ob_path)
        if not dates_to_process:
            logger.log("No dates found! Exiting.")
            return
        logger.log(f"Found {len(dates_to_process)} dates to process: {dates_to_process}")
    else:
        dates_to_process = [args.date]
        
    for current_date in dates_to_process:
        logger.log(f"=== PROCESSING DATE: {current_date} ===")
        out_path = os.path.join(args.out_dir, current_date)
        os.makedirs(out_path, exist_ok=True)
        logger.log(f"Output directory: {out_path}")
        
        for h in range(args.start_hour, args.end_hour + 1):
            chunk_file = os.path.join(out_path, f"chunk_{h}.pt")
            
            if os.path.exists(chunk_file):
                logger.log(f"[{current_date} {h}:00] Chunk {chunk_file} already exists. Skipping...")
                continue
                
            logger.log(f"[{current_date} {h}:00] Initializing CausalCoordinator...")
            dataset = CausalCoordinator(
                date_str=current_date,
                start_hour=h,
                end_hour=h
            )
            
            # CausalCoordinator also prints things natively, which won't use DeltaLogger,
            # but we can't change all print statements inside the dataset without touching many files.
            # DeltaLogger will capture the time delta for the entire operation.
            dataset.build()
            
            if len(dataset.grid_times) == 0:
                logger.log(f"[{current_date} {h}:00] No grid times found. Saving empty chunk.")
                torch.save([], chunk_file)
                continue
                
            logger.log(f"[{current_date} {h}:00] Running Vectorized Market-by-Market LOB Engine for the entire hour...")
            
            end_of_hour_t = dataset.grid_times[-1]
            all_lob_updates = dataset.lob_engine.step(dataset.market_ids, dataset.ob_df, dataset.snap_df, end_of_hour_t)
            
            logger.log(f"[{current_date} {h}:00] LOB Engine extracted ALL events.")
            
            step_lob_buckets = {i: {} for i in range(len(dataset.grid_times))}
            grid_series = pd.Series(dataset.grid_times)
            total_chunks = 0
            
            for m_id, chunks in all_lob_updates.items():
                total_chunks += len(chunks)
                if not chunks: continue
                
                t_ends = pd.Series([c[0] for c in chunks])
                idx_assignments = np.searchsorted(grid_series, t_ends)
                
                for i, chunk in enumerate(chunks):
                    t_idx = idx_assignments[i]
                    if t_idx >= len(dataset.grid_times):
                        t_idx = len(dataset.grid_times) - 1
                    if m_id not in step_lob_buckets[t_idx]:
                        step_lob_buckets[t_idx][m_id] = []
                    step_lob_buckets[t_idx][m_id].append(chunk)
                    
            logger.log(f"[{current_date} {h}:00] Distributed {total_chunks} LOB chunks across {len(dataset.market_ids)} markets.")
                
            chunk_data = []
            total_steps = len(dataset.grid_times)
            
            logger.log(f"[{current_date} {h}:00] Collating PyTorch tensors...")
            for t_idx in range(total_steps):
                if t_idx % 400 == 0 and t_idx > 0:
                    logger.log(f"[{current_date} {h}:00] Collated {t_idx}/{total_steps} temporal steps...")
                    
                lob_updates_for_step = step_lob_buckets[t_idx]
                emp_feats, spatial_feats, valid_seq_mask, edge_idx, edge_wt, y, target_mask = fast_get_step_collation(
                    dataset, t_idx, lob_updates_for_step, device
                )
                
                step_data = {
                    'emp_feats': emp_feats.cpu(),
                    'spatial_feats': spatial_feats.cpu(),
                    'valid_seq_mask': valid_seq_mask.cpu(),
                    'edge_idx': {k: v.cpu() for k, v in edge_idx.items()},
                    'edge_wt': {k: v.cpu() for k, v in edge_wt.items()},
                    'y': y.cpu(),
                    'target_mask': target_mask.cpu(),
                    'timestamp': dataset.grid_times[t_idx].timestamp()
                }
                chunk_data.append(step_data)
                
            logger.log(f"[{current_date} {h}:00] Saving {len(chunk_data)} steps to disk...")
            torch.save({
                'metadata': {
                    'n_markets': dataset.n_markets,
                    'n_wallets': dataset.n_wallets,
                    'grid_times': dataset.grid_times,
                    'market_ids': dataset.market_ids,
                },
                'steps': chunk_data
            }, chunk_file)
            
            logger.log(f"[{current_date} {h}:00] Hour completely processed and saved to {chunk_file}!")
            
    logger.log("=== ALL REQUESTED DATES PROCESSED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
