import os
import torch
import numpy as np

class FastCausalDataset:
    """
    A lightning-fast dataset loader that reads precomputed tensor chunks
    instead of parsing Parquet/CSV files on the fly.
    """
    def __init__(self, data_dir: str, date_str: str, start_hour: int, end_hour: int):
        self.data_dir = os.path.join(data_dir, date_str)
        self.start_hour = start_hour
        self.end_hour = end_hour
        
        self.chunks = {}
        self.grid_times = []
        
        self.n_markets = 0
        self.n_wallets = 0
        self.market_ids = []
        
        self._load_chunks()
        
    def _load_chunks(self):
        """Loads metadata and initializes chunk mapping."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Processed data directory {self.data_dir} not found. Run precompute_tensors.py first!")
            
        print(f"Loading precomputed chunks from {self.start_hour}:00 to {self.end_hour}:59...")
        
        total_steps = 0
        for h in range(self.start_hour, self.end_hour + 1):
            chunk_file = os.path.join(self.data_dir, f"chunk_{h}.pt")
            if not os.path.exists(chunk_file):
                print(f"WARNING: Missing chunk {chunk_file}. Skipping...")
                continue
                
            chunk_data = torch.load(chunk_file, weights_only=False)
            
            # Load metadata from the first valid chunk
            if self.n_markets == 0 and len(chunk_data) > 0 and 'metadata' in chunk_data:
                self.n_markets = chunk_data['metadata']['n_markets']
                self.n_wallets = chunk_data['metadata']['n_wallets']
                self.market_ids = chunk_data['metadata']['market_ids']
                
            if 'steps' not in chunk_data:
                continue
                
            steps = chunk_data['steps']
            grid_times = chunk_data['metadata']['grid_times']
            
            for i, step in enumerate(steps):
                global_idx = len(self.grid_times)
                self.chunks[global_idx] = step
                self.grid_times.append(grid_times[i])
                
            print(f"Loaded hour {h} ({len(steps)} steps)")
            total_steps += len(steps)
            
        print(f"FastCausalDataset loaded {total_steps} total steps.")
        
        # Split chronological
        if total_steps > 0:
            T = len(self.grid_times)
            split = int(T * 0.7)
            # hardcoded horizon_steps=15 as in CausalCoordinator
            self.train_indices = np.arange(0, split)
            self.val_indices = np.arange(split, max(split, T - 15))
        else:
            self.train_indices = np.array([])
            self.val_indices = np.array([])
        
    def get_step(self, t_idx: int, device: torch.device):
        """
        Returns instantaneous features and edges for EXACTLY step t_idx.
        Loads directly from RAM (precomputed tensors) and moves to device.
        """
        step = self.chunks[t_idx]
        
        emp_feats = step['emp_feats'].to(device, non_blocking=True)
        spatial_feats = step['spatial_feats'].to(device, non_blocking=True)
        valid_seq_mask = step['valid_seq_mask'].to(device, non_blocking=True)
        
        edge_idx = {k: v.to(device, non_blocking=True) for k, v in step['edge_idx'].items()}
        edge_wt = {k: v.to(device, non_blocking=True) for k, v in step['edge_wt'].items()}
        
        y = step['y'].to(device, non_blocking=True)
        target_mask = step['target_mask'].to(device, non_blocking=True)
        
        return emp_feats, spatial_feats, valid_seq_mask, edge_idx, edge_wt, y, target_mask
