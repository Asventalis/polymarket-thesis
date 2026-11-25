import os
import glob
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_partition_path(path_str: str) -> Dict[str, int]:
    """Extracts year, month, day, hour from a parquet partition path."""
    parts = path_str.split('/')
    result = {}
    for p in parts:
        if '=' in p:
            k, v = p.split('=')
            if k in ['year', 'month', 'day', 'hour']:
                result[k] = int(v)
    return result

def main():
    parser = argparse.ArgumentParser(description="Compute volume deltas for all orderbook events chronologically.")
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "derived" / "partitioned"))
    parser.add_argument("--test_mode", action="store_true", help="Only process the first 2 hours for testing.")
    args = parser.parse_args()

    ob_dir = os.path.join(args.base_dir, "orderbook")
    snap_dir = os.path.join(args.base_dir, "snapshots")
    checkpoint_file = os.path.join(args.base_dir, "pipeline_state_checkpoint.json")

    # 1. Discover all orderbook partitions
    ob_files = glob.glob(os.path.join(ob_dir, "**", "*.parquet"), recursive=True)
    if not ob_files:
        logging.error(f"No orderbook parquet files found in {ob_dir}")
        return

    # Parse and sort chronologically
    partitions = []
    for f in ob_files:
        p_info = parse_partition_path(f)
        p_info['filepath'] = f
        # Also construct the corresponding snapshot path
        snap_path = f.replace("/orderbook/", "/snapshots/")
        p_info['snap_filepath'] = snap_path if os.path.exists(snap_path) else None
        
        # Sort key (list so it can be JSON serialized easily)
        p_info['sort_key'] = [
            p_info.get('year', 0), 
            p_info.get('month', 0), 
            p_info.get('day', 0), 
            p_info.get('hour', 0)
        ]
        partitions.append(p_info)

    partitions.sort(key=lambda x: tuple(x['sort_key']))

    if args.test_mode:
        logging.info("TEST MODE: Only processing the first 2 hourly partitions.")
        partitions = partitions[:2]

    logging.info(f"Found {len(partitions)} chronological hour partitions to process.")

    # 2. Global State tracking & Checkpoint Loading
    market_states: Dict[str, Dict[str, Dict[float, float]]] = {}
    last_processed_sort_key = None
    
    if os.path.exists(checkpoint_file):
        logging.info(f"Checkpoint found at {checkpoint_file}. Loading state...")
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
            market_states = ckpt.get("market_states", {})
            
            # JSON dict keys are strings, we must cast price keys back to floats for math
            for m_id, sides in market_states.items():
                for side, prices in sides.items():
                    market_states[m_id][side] = {float(p): float(v) for p, v in prices.items()}
                    
            last_processed_sort_key = ckpt.get("last_processed_sort_key")
            logging.info(f"Successfully restored state for {len(market_states)} markets up to {last_processed_sort_key}.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return
            
    total_processed = 0
    total_missing_baseline = 0

    # 3. Process chronologically
    for p_idx, p in enumerate(partitions):
        # Skip if already processed according to checkpoint
        if last_processed_sort_key is not None:
            # Compare tuples for chronological skipping
            if tuple(p['sort_key']) <= tuple(last_processed_sort_key):
                logging.info(f"[{p_idx+1}/{len(partitions)}] Skipping {p['sort_key']} (Already processed in checkpoint).")
                continue
                
        logging.info(f"[{p_idx+1}/{len(partitions)}] Processing {p.get('year', 0)}-{p.get('month', 0):02d}-{p.get('day', 0):02d} hour={p.get('hour', 0):02d} ...")
        
        # Load orderbook
        ob_df = pd.read_parquet(p['filepath'])
        ob_df['event_type'] = 'tick'
        
        # Load snaps
        if p['snap_filepath']:
            snap_df = pd.read_parquet(p['snap_filepath'])
            snap_df['event_type'] = 'snap'
            combined = pd.concat([snap_df, ob_df], ignore_index=True)
        else:
            combined = ob_df

        # Sort chronologically
        combined.sort_values(by="timestamp_received", inplace=True)
        
        # Extract arrays for fast, memory-efficient iteration
        m_ids = combined['market_id'].values
        ev_types = combined['event_type'].values
        sides = combined['change_side'].astype(str).values
        prices = combined['change_price'].values
        sizes = combined['change_size'].values
        datas = combined['data'].values
        
        deltas = np.full(len(combined), np.nan, dtype=np.float32)
        is_tick_mask = (ev_types == 'tick')
        
        for i in range(len(combined)):
            m_id = m_ids[i]
            ev = ev_types[i]
            
            if ev == 'snap':
                raw_data = datas[i]
                if pd.notnull(raw_data) and raw_data:
                    try:
                        parsed = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                        bids = {float(k): float(v) for k, v in parsed.get('bids', [])}
                        asks = {float(k): float(v) for k, v in parsed.get('asks', [])}
                        market_states[m_id] = {'BUY': bids, 'SELL': asks}
                    except Exception as e:
                        logging.warning(f"Failed to parse snapshot for market {m_id}: {e}")
            elif ev == 'tick':
                if m_id not in market_states:
                    market_states[m_id] = {'BUY': {}, 'SELL': {}}
                    total_missing_baseline += 1
                
                side_str = str(sides[i]).upper()
                price_val = prices[i]
                size_val = sizes[i]
                
                if pd.notnull(price_val) and side_str in ['BUY', 'SELL']:
                    # Cast numpy types to native Python types for JSON serialization!
                    p_f = float(price_val)
                    s_f = float(size_val)
                    m_id_str = str(m_id)
                    
                    old_vol = market_states[m_id_str][side_str].get(p_f, 0.0)
                    delta = s_f - old_vol
                    deltas[i] = delta
                    
                    if s_f <= 1e-6:
                        market_states[m_id_str][side_str].pop(p_f, None)
                    else:
                        market_states[m_id_str][side_str][p_f] = s_f
                total_processed += 1
                
        # Assign deltas and filter out snapshot rows
        combined['delta'] = deltas
        final_df = combined[is_tick_mask].drop(columns=['event_type', 'data'])
        
        target_path = p['filepath']
        tmp_target_path = target_path + ".tmp"
        
        # Save augmented dataframe to a temp file, then atomic replace to prevent corruption on crash
        final_df.to_parquet(tmp_target_path, index=False)
        os.replace(tmp_target_path, target_path)
        logging.info(f"   -> Overwritten {target_path} successfully.")

        # Atomically dump checkpoint JSON
        ckpt_tmp = checkpoint_file + ".tmp"
        try:
            with open(ckpt_tmp, 'w') as f:
                json.dump({
                    "last_processed_sort_key": p['sort_key'],
                    "market_states": market_states
                }, f)
            os.replace(ckpt_tmp, checkpoint_file)
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    logging.info(f"Done! Processed {total_processed} total ticks.")
    logging.info(f"Fake empty baseline snapshots created: {total_missing_baseline}")


if __name__ == "__main__":
    main()

