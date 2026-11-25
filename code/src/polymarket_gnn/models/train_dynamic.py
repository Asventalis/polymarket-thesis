import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from polymarket_gnn.data.causal_dataset import CausalCoordinator
from polymarket_gnn.models.causal_gnn import CausalDynamicGNN

def run_tbptt_epoch(
    model: CausalDynamicGNN,
    dataset: CausalCoordinator,
    optimizer: torch.optim.Optimizer,
    is_train: bool,
    t_indices: np.ndarray,
    device: torch.device,
    truncate_steps: int = 50,
):
    """
    Runs one epoch over the temporal sequence using Truncated Backpropagation Through Time (TBPTT).
    """
    model.train() if is_train else model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Global state initialization
    h_m = torch.zeros(dataset.n_markets, model.hidden_dim, device=device)
    h_w = torch.zeros(dataset.n_wallets, model.hidden_dim, device=device)
    
    p_edge_idx = {}
    p_edge_wt = {}
    last_update_time = torch.zeros(dataset.n_markets, 1, device=device)
    
    total_loss = 0.0
    accumulated_loss = 0.0
    valid_predictions = 0
    steps_since_backward = 0
    
    t0 = time.time()
    
    for step_i, t in enumerate(t_indices):
        emp_feats, spatial_feats, valid_seq_mask, edge_idx, edge_wt, y, target_mask = dataset.get_step(int(t), device)
        
        # ── Time-Aware RNN logic ──
        current_time = dataset.grid_times[int(t)].timestamp()
        
        if step_i == 0:
            last_update_time.fill_(current_time)
            
        # Delta time in minutes for better NN scaling
        delta_time = (current_time - last_update_time) / 60.0
        
        # Update last_update_time for active markets
        active_this_step = valid_seq_mask.any(dim=1)
        last_update_time[active_this_step] = current_time
        
        # ── Edge Decay logic ──
        for rel in ['yes_buy', 'yes_sell', 'no_buy', 'no_sell']:
            if rel in p_edge_wt:
                p_edge_wt[rel] = p_edge_wt[rel] * 0.65
            
            e_idx = edge_idx.get(rel, torch.zeros(2, 0, dtype=torch.long, device=device))
            e_wt = edge_wt.get(rel, torch.zeros(0, dtype=torch.float32, device=device))
            
            if e_idx.size(1) > 0:
                if rel in p_edge_idx and p_edge_idx[rel].size(1) > 0:
                    comb_idx = torch.cat([p_edge_idx[rel], e_idx], dim=1)
                    comb_wt = torch.cat([p_edge_wt[rel], e_wt], dim=0)
                else:
                    comb_idx = e_idx
                    comb_wt = e_wt
                
                sparse = torch.sparse_coo_tensor(comb_idx, comb_wt, (dataset.n_wallets, dataset.n_markets), device=device)
                sparse = sparse.coalesce()
                p_edge_idx[rel] = sparse.indices()
                p_edge_wt[rel] = sparse.values()
            
            if rel in p_edge_wt and p_edge_wt[rel].size(0) > 0:
                mask = p_edge_wt[rel] >= 0.01
                p_edge_idx[rel] = p_edge_idx[rel][:, mask]
                p_edge_wt[rel] = p_edge_wt[rel][mask]
        
        # ── Causal Forward Pass ──
        logits, h_m, h_w = model(emp_feats, spatial_feats, valid_seq_mask, h_m, h_w, delta_time, p_edge_idx, p_edge_wt)
        
        # Compute loss on active markets
        if target_mask.sum() > 0:
            loss = criterion(logits[target_mask].squeeze(-1), y[target_mask])
            accumulated_loss = accumulated_loss + loss
            valid_predictions += target_mask.sum().item()
        
        steps_since_backward += 1
        
        # ── TBPTT Backward Pass ──
        if is_train and steps_since_backward >= truncate_steps:
            if isinstance(accumulated_loss, torch.Tensor):
                optimizer.zero_grad()
                accumulated_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += accumulated_loss.item()
            
            accumulated_loss = 0.0
            steps_since_backward = 0
            
            # Detach hidden states so gradient graph doesn't grow infinitely
            h_m = h_m.detach()
            h_w = h_w.detach()
            
    # Final trailing backward
    if is_train and isinstance(accumulated_loss, torch.Tensor) and steps_since_backward > 0:
        optimizer.zero_grad()
        accumulated_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += accumulated_loss.item()
        
    elapsed = time.time() - t0
    avg_loss = total_loss / (len(t_indices) / truncate_steps) if is_train else accumulated_loss.item() if isinstance(accumulated_loss, torch.Tensor) else 0.0
    return avg_loss, valid_predictions, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-03-06")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tbptt", type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build the Causal Coordinator
    dataset = CausalCoordinator(
        date_str=args.date,
        start_hour=14,
        end_hour=15
    )
    dataset.build()

    # Build the Model
    model = CausalDynamicGNN(hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nStarting TBPTT Causal Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_preds, train_time = run_tbptt_epoch(
            model, dataset, optimizer, True, dataset.train_indices, device, args.tbptt
        )
        
        with torch.no_grad():
            val_loss, val_preds, val_time = run_tbptt_epoch(
                model, dataset, optimizer, False, dataset.val_indices, device, args.tbptt
            )
            
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} ({train_time:.1f}s) | Val Loss: {val_loss:.4f} ({val_time:.1f}s)")

if __name__ == "__main__":
    main()
