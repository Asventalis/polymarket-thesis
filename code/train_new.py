"""
train_new.py

Main training loop orchestrating LOB updates, Transaction updates, and GNN prediction
across multiple hours/files, stepping through transactions block by block.
"""
import time
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pandas as pd
import duckdb

from polymarket_gnn.data.lob_window_buffer import LOBWindowBuffer
from polymarket_gnn.data.graph_builder import EdgeBuilder, EDGE_DIM
from polymarket_gnn.models.causal_gnn import CausalDynamicGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path("/data/v.pampinella/agy_project/data/derived")
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def get_hour_paths(base_dir: Path, year, month, day, hour):
    lob_path = base_dir / f"tensors/year={year}/month={month}/day={day}/hour={hour}/lob_features.parquet"
    txn_path = base_dir / f"partitioned/transactions/year={year}/month={month}/day={day}/hour={hour}"
    return lob_path, txn_path



def main():
    logger.info(f"Using device: {DEVICE}")

    # ── 1. Load mappings ──────────────────────────────────────────────────────
    logger.info("Loading ID mappings...")
    market_mapping = pd.read_parquet(BASE_DIR / "id_mappings/market_to_idx.parquet")
    wallet_mapping = pd.read_parquet(BASE_DIR / "id_mappings/wallet_to_idx.parquet")
    
    market_to_idx = dict(zip(market_mapping["market_id"], market_mapping["market_idx"]))
    wallet_to_idx = dict(zip(wallet_mapping["wallet"], wallet_mapping["wallet_idx"]))
    
    N_MARKETS = len(market_to_idx)
    N_WALLETS = len(wallet_to_idx)
    logger.info(f"Total markets: {N_MARKETS}, Total wallets: {N_WALLETS}")

    # ── 2. Initialize node embeddings ─────────────────────────────────────────
    HIDDEN_DIM = 128
    logger.info("Initializing node embeddings...")
    
    market_init = torch.load(BASE_DIR / "market_features.pt", map_location="cpu") # (175292, 390)
    market_projector = nn.Linear(390, HIDDEN_DIM).to(DEVICE)
    
    with torch.no_grad():
        h_market = market_projector(market_init.to(DEVICE))
        h_market = torch.nan_to_num(h_market, nan=0.0, posinf=0.0, neginf=0.0)
        h_market = torch.clamp(h_market, min=-10.0, max=10.0)
    
    h_wallet = torch.randn(N_WALLETS, HIDDEN_DIM, device=DEVICE) * 0.01
    h_wallet = torch.nan_to_num(h_wallet, nan=0.0, posinf=0.0, neginf=0.0)
    h_wallet = torch.clamp(h_wallet, min=-10.0, max=10.0)
    
    # ── 3. Model setup ────────────────────────────────────────────────────────
    model = CausalDynamicGNN(lob_dim=52, hidden_dim=HIDDEN_DIM, edge_dim=EDGE_DIM).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    class_weights = torch.tensor([10.0, 1.0, 10.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    TBPTT_STEPS = 20
    ACCUMULATION_STEPS = 5  # Effective batch size = 20 * 5 = 100 blocks
    
    # Edge builder with pre-computed lookup structures
    edge_builder = EdgeBuilder(market_to_idx, wallet_to_idx)
    
    # 3 Hours of Training, 1 Hour of Validation
    hours_to_run = [
        (2026, 3, 23, 13, True),
        (2026, 3, 23, 14, True),
        (2026, 3, 23, 15, True),
        (2026, 3, 23, 16, False)
    ]
    
    # ── 5. Global Loop State ─────────────────────────────────
    block_count = 0
    total_lob_updates = 0
    total_edges = 0
    
    # Accumulated loss for TBPTT
    tbptt_loss = torch.tensor(0.0, device=DEVICE)
    loss_history = []
    val_history = []
    prev_block_ts = None
    
    # Initialize LOB buffer with the first hour
    first_hour = hours_to_run[0]
    first_lob, _ = get_hour_paths(BASE_DIR, first_hour[0], first_hour[1], first_hour[2], first_hour[3])
    
    logger.info("Loading initial LOB Window Buffer...")
    buffer = LOBWindowBuffer.from_parquet(first_lob, K=10)
    
    # Fast forward lifecycle to the start of the first hour
    hour_start_ns = pd.Timestamp(f"{first_hour[0]}-{first_hour[1]:02d}-{first_hour[2]:02d} {first_hour[3]:02d}:00:00").value
    buffer.reset_to_time(hour_start_ns)

    # ── 4. Outer loop over hours ──────────────────────────────────────────────
    for year, month, day, hour, is_train in hours_to_run:
        if is_train:
            model.train()
            logger.info(f"--- Starting TRAINING Hour: {year}-{month:02d}-{day:02d} {hour:02d}:00 ---")
        else:
            model.eval()
            tbptt_loss = torch.tensor(0.0, device=DEVICE)
            logger.info(f"--- Starting VALIDATION Hour: {year}-{month:02d}-{day:02d} {hour:02d}:00 ---")
        lob_path, txn_path = get_hour_paths(BASE_DIR, year, month, day, hour)
        
        # Load transactions for this hour
        t0 = time.time()
        try:
            # We use try/except in case there are no parquet files in this hour directory
            txn_df = duckdb.query(f"SELECT * FROM read_parquet('{txn_path}/**/*.parquet') ORDER BY block_number, timestamp").df()
            logger.info(f"Loaded {len(txn_df)} transactions for the hour in {time.time()-t0:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to load transactions for {txn_path}: {e}")
            continue
            
        if txn_df.empty:
            logger.info("No transactions found, skipping hour.")
            continue
            
        # Swap in the new hour's LOB data (skip for the very first hour since we already loaded it)
        if (year, month, day, hour) != hours_to_run[0]:
            buffer.reload_lob_data(lob_path)
            
        # Group transactions by exact block
        blocks = txn_df.groupby("block_number", sort=True)
        
        for block_number, block_txns in blocks:
            # Exact block timestamp
            t_ns = block_txns["timestamp"].iloc[0].value
            if prev_block_ts is None:
                prev_block_ts = pd.Timestamp(f"{year}-{month:02d}-{day:02d} {hour:02d}:00:00").value
            t_prev_ns = prev_block_ts
            
            if t_ns <= t_prev_ns:
                prev_block_ts = max(t_ns, t_prev_ns)
                continue

            # ── Step A: LOB Update ─────────────────────────────────────────────
            t0 = time.time()
            active_ids, windows = buffer.load_step(t_prev_ns, t_ns)
            
            if len(active_ids) > 0:
                active_idx = torch.tensor(
                    [market_to_idx[m] for m in active_ids if m in market_to_idx], 
                    dtype=torch.long, device=DEVICE
                )
                
                # Filter windows for markets we actually have mappings for
                mapped_mask = [m in market_to_idx for m in active_ids]
                windows_gpu = torch.tensor(windows[mapped_mask], dtype=torch.float32, device=DEVICE)
                
                # Replace any potential NaNs in the raw LOB data with 0.0 to prevent network corruption
                windows_gpu = torch.nan_to_num(windows_gpu, nan=0.0, posinf=0.0, neginf=0.0)
                # CLAMP huge values (like raw volumes in millions) so gradients don't explode to Inf
                windows_gpu = torch.clamp(windows_gpu, min=-100.0, max=100.0)
                
                total_lob_updates += len(active_idx)
            else:
                active_idx = torch.empty(0, dtype=torch.long, device=DEVICE)
                windows_gpu = torch.empty((0, 5, 52), dtype=torch.float32, device=DEVICE)
                
            lob_time = time.time() - t0

            # ── Step B: Transaction Edges ──────────────────────────────────────
            t1 = time.time()
            edge_index, edge_features = edge_builder(block_txns, device=DEVICE)
            # Remove NaNs and Infs, and clamp to a safe range
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
            edge_features = torch.clamp(edge_features, min=-100.0, max=100.0)
            edge_time = time.time() - t1
            
            # To avoid in-place modification issues in the autograd graph,
            # we clone the state tensors before they enter the model for this step.
            h_market_in = h_market.clone()
            h_wallet_in = h_wallet.clone()

            # ── Step C: Targets, Forward Pass, and Loss ────────────────────────
            HORIZON_NS = int(60 * 1e9)
            targets = buffer.get_future_targets(active_ids, t_ns, HORIZON_NS, device=DEVICE)
            
            valid_mask = ~torch.isnan(targets)
            valid_active_idx = active_idx[valid_mask]
            valid_targets = targets[valid_mask]

            # Check parameters before forward pass
            has_nan_weights = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan_weights:
                logger.error(f"NaNs found in model weights BEFORE forward pass at block {block_count}!")
                raise ValueError("Model weights are NaN")

            # Compute time elapsed since previous block (for all nodes)
            delta_t_global = 0.0
            if prev_block_ts is not None:
                delta_t_global = (t_ns - prev_block_ts) / 1e9 # seconds

            with torch.set_grad_enabled(is_train):
                if is_train:
                    # In order for checkpoint to pass gradients backward, at least one input must require_grad
                    h_market_in.requires_grad_(True)
                    dt_tensor = torch.tensor(delta_t_global, dtype=torch.float32, device=DEVICE)
                    
                    logits, h_market_out, h_wallet_out = checkpoint(
                        model,
                        windows_gpu,
                        active_idx,
                        h_market_in,
                        h_wallet_in,
                        edge_index,
                        edge_features,
                        dt_tensor,
                        dt_tensor,
                        use_reentrant=False
                    )
                else:
                    logits, h_market_out, h_wallet_out = model(
                        windows=windows_gpu,
                        active_idx=active_idx,
                        h_market=h_market_in,
                        h_wallet=h_wallet_in,
                        edge_index=edge_index,
                        edge_features=edge_features,
                        delta_t_market=delta_t_global,
                        delta_t_wallet=delta_t_global,
                    )
            
            if torch.isnan(logits).any():
                logger.error(f"NaNs detected in logits at block {block_count}!")
                raise ValueError("NaN found in logits")
                
            h_market = h_market_out
            h_wallet = h_wallet_out
            
            step_loss_val = 0.0
            if len(valid_active_idx) > 0:
                active_logits = logits[valid_active_idx] # (N, 3)
                valid_targets_long = valid_targets.long() # CrossEntropy requires long targets
                
                loss = criterion(active_logits, valid_targets_long)
                if is_train:
                    tbptt_loss = tbptt_loss + loss
                step_loss_val = loss.item()
                
                # Calculate Metrics for 3 classes
                preds = torch.argmax(active_logits, dim=1) # (N,)
                
                # We'll compute the confusion matrix counts for all 3x3 combinations
                # Matrix format: CM[true_class][pred_class]
                cm = [[0]*3 for _ in range(3)]
                for c_true in range(3):
                    for c_pred in range(3):
                        cm[c_true][c_pred] = ((valid_targets_long == c_true) & (preds == c_pred)).sum().item()
                
                correct = cm[0][0] + cm[1][1] + cm[2][2]
                accuracy = correct / len(valid_targets_long)
                
                target_history = loss_history if is_train else val_history
                target_history.append({
                    "block": block_count, 
                    "loss": step_loss_val, 
                    "accuracy": accuracy,
                    "cm": cm # 3x3 confusion matrix array
                })

            # ── Truncated Backpropagation Through Time (TBPTT) ─────────────────
            if block_count > 0 and block_count % TBPTT_STEPS == 0:
                if is_train:
                    if tbptt_loss.requires_grad and not torch.isnan(tbptt_loss):
                        # Scale loss by accumulation steps so gradients are properly averaged
                        (tbptt_loss / ACCUMULATION_STEPS).backward()
                        
                    # Only update weights every ACCUMULATION_STEPS cycles (e.g. 100 blocks)
                    if block_count % (TBPTT_STEPS * ACCUMULATION_STEPS) == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Detach state to prevent graph from growing infinitely
                h_market = h_market.detach()
                h_wallet = h_wallet.detach()
                
                # If embeddings became NaN (e.g. from a bad backward pass previously), zero them out
                h_market = torch.nan_to_num(h_market, nan=0.0)
                h_wallet = torch.nan_to_num(h_wallet, nan=0.0)
                
                if is_train:
                    tbptt_loss = torch.tensor(0.0, device=DEVICE)
            
            prev_block_ts = t_ns
            block_count += 1
            total_edges += edge_index.size(1)
            
            # Log occasionally to show progress
            if block_count % 100 == 0:
                logger.info(
                    f"Block {block_count:4d} | blk={block_number} | "
                    f"t={pd.Timestamp(t_ns)} | "
                    f"LOB={len(active_idx):3d} | edges={edge_index.size(1):4d} | "
                    f"loss={step_loss_val:.4f} | "
                    f"lob_t={lob_time*1000:.1f}ms edge_t={edge_time*1000:.1f}ms"
                )

        logger.info(
            f"Finished hour {hour:02d}:00 | blocks={block_count} total | "
            f"LOB_updates={total_lob_updates} | edges={total_edges}"
        )

    # Save loss history for plotting at the end of all hours
    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)
    with open("val_history.json", "w") as f:
        json.dump(val_history, f)

if __name__ == "__main__":
    main()
