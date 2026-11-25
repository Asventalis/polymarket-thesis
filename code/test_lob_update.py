"""
test_lob_update.py

Tests the LOBUpdateModule end-to-end:
  - Loads LOBWindowBuffer for one hour
  - Loads market_to_idx mapping
  - Initializes h_market on GPU
  - Runs 20 training steps, verifying h_market updates correctly
"""
import time
import logging
import torch
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

from polymarket_gnn.data.lob_window_buffer import LOBWindowBuffer
from polymarket_gnn.models.causal_gnn import LOBUpdateModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE        = Path("/data/v.pampinella/agy_project/data/derived")
TENSOR_PATH = BASE / "tensors/year=2026/month=3/day=23/hour=13/lob_features.parquet"
TXN_PATH    = "/data/v.pampinella/agy_project/data/derived/partitioned/transactions/year=2026/month=3/day=23/hour=13"
MAPPING_PATH = BASE / "id_mappings/market_to_idx.parquet"

K          = 5
HIDDEN_DIM = 128
DEVICE     = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def main():
    logger.info(f"Device: {DEVICE}")

    # ── 1. Load market mapping ────────────────────────────────────────────────
    logger.info("Loading market_to_idx mapping...")
    mapping_df    = pd.read_parquet(MAPPING_PATH)
    market_to_idx = dict(zip(mapping_df["market_id"], mapping_df["market_idx"]))
    N_MARKETS     = len(market_to_idx)
    logger.info(f"Total markets in mapping: {N_MARKETS}")

    # ── 2. Build LOBWindowBuffer ──────────────────────────────────────────────
    logger.info("Building LOBWindowBuffer...")
    t0  = time.time()
    buf = LOBWindowBuffer.from_parquet(TENSOR_PATH, K=K)
    logger.info(f"Buffer ready in {time.time()-t0:.2f}s")

    # ── 3. Load transaction timestamps (to define step boundaries) ────────────
    logger.info("Loading transaction buckets...")
    txn_df = duckdb.query(
        f"SELECT timestamp FROM read_parquet('{TXN_PATH}/**/*.parquet') ORDER BY timestamp"
    ).df()
    txn_df["bucket"] = txn_df["timestamp"].dt.floor("2s")
    buckets = sorted(txn_df["bucket"].unique())
    logger.info(f"Total 2s buckets: {len(buckets)}")

    # ── 4. Initialize global state ────────────────────────────────────────────
    h_market = torch.zeros(N_MARKETS, HIDDEN_DIM, device=DEVICE)
    model    = LOBUpdateModule(lob_dim=52, hidden_dim=HIDDEN_DIM).to(DEVICE)
    logger.info(f"h_market initialized: {h_market.shape} on {DEVICE}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 5. Run training steps ─────────────────────────────────────────────────
    logger.info("\n── Running 20 training steps ──")
    h_before = h_market.clone()

    for i, t in enumerate(buckets[:20]):
        t_ns      = t.value
        t_prev_ns = (t - pd.Timedelta("2s")).value

        # Get active markets and their windows
        active_market_ids, windows_np = buf.load_step(t_prev_ns, t_ns)

        if len(active_market_ids) == 0:
            logger.info(f"Step {i:2d} | t={t} | No active markets, skipping")
            continue

        # Convert market IDs → integer indices
        active_idx = torch.tensor(
            [market_to_idx[m] for m in active_market_ids if m in market_to_idx],
            dtype=torch.long, device=DEVICE
        )

        # Filter windows to only mapped markets
        mapped_mask = [m in market_to_idx for m in active_market_ids]
        windows_np  = windows_np[mapped_mask]

        if len(active_idx) == 0:
            continue

        # Push to GPU
        windows_gpu = torch.tensor(windows_np, dtype=torch.float32, device=DEVICE)

        # Forward pass: update h_market in place
        t0 = time.time()
        with torch.no_grad():
            h_market = model(windows_gpu, h_market, active_idx)
        elapsed = time.time() - t0

        # Check norm of updated rows
        updated_norm = h_market[active_idx].norm(dim=1).mean().item()

        logger.info(
            f"Step {i:2d} | t={t} | active={len(active_idx):3d} | "
            f"windows={windows_gpu.shape} | "
            f"avg_norm={updated_norm:.4f} | step_time={elapsed*1000:.2f}ms"
        )

    # ── 6. Verify h_market actually changed ───────────────────────────────────
    diff = (h_market - h_before).abs().sum().item()
    logger.info(f"\nTotal L1 change in h_market: {diff:.4f}")
    logger.info("✓ h_market updated successfully" if diff > 0 else "✗ h_market unchanged — something is wrong")

if __name__ == "__main__":
    main()
