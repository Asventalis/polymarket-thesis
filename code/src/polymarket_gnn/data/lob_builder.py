import numpy as np
import json
import logging
from typing import Dict, Any, List
import time
import pandas as pd
import duckdb
from pathlib import Path
import os

try:
    from polymarket_gnn.data.dataset_utils import load_and_interleave_hour
except ImportError:
    from dataset_utils import load_and_interleave_hour

logger = logging.getLogger(__name__)

class OrderbookState:
    def __init__(self, market_id: str, depth_levels: int = 10):
        self.market_id = market_id
        self.depth_levels = depth_levels
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.reset_accumulators()
        
    def reset_accumulators(self):
        self.events_processed = 0
        self.acc_ofi = 0.0
        self.acc_volume = 0.0
        self.acc_buy_count = 0
        self.acc_cancel_count = 0
        
    def apply_event(self, row: Any):
        row_type = getattr(row, 'type', 'event')
        if row_type == 'snapshot':
            self._apply_snapshot(getattr(row, 'data', None))
        else:
            self._apply_delta(row)
        self.events_processed += 1
        
    def _apply_snapshot(self, raw_json: str):
        if not raw_json or not isinstance(raw_json, str):
            return
        try:
            parsed = json.loads(raw_json)
            self.bids = {float(p): float(v) for p, v in parsed.get("bids", [])}
            self.asks = {float(p): float(v) for p, v in parsed.get("asks", [])}
        except Exception:
            pass
            
    def _apply_delta(self, row: Any):
        price = getattr(row, 'change_price', None)
        size = getattr(row, 'change_size', None)
        side = getattr(row, 'change_side', None)
        delta = getattr(row, 'delta', 0.0)
        
        if pd.isna(price) or pd.isna(size) or pd.isna(side):
            return
            
        price, size = float(price), float(size)
        side = str(side).upper()
        
        book = self.bids if side == "BUY" else self.asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size
            
        is_buy = 1.0 if side == "BUY" else 0.0
        delta_val = float(delta) if not pd.isna(delta) else 0.0
        is_cancel = 1.0 if delta_val < 0 else 0.0
        
        self.acc_volume += abs(delta_val)
        self.acc_buy_count += is_buy
        self.acc_cancel_count += is_cancel
        self.acc_ofi += (1.0 if is_buy else -1.0) * delta_val
        
    def extract_features(self, timestamp) -> dict:
        best_b = max(self.bids.keys()) if self.bids else 0.49
        best_a = min(self.asks.keys()) if self.asks else 0.51
        mid_price = (best_b + best_a) / 2.0
        spread = best_a - best_b
        
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        
        bid_prices, bid_sizes = np.zeros(self.depth_levels), np.zeros(self.depth_levels)
        ask_prices, ask_sizes = np.zeros(self.depth_levels), np.zeros(self.depth_levels)
        
        for i, (p, v) in enumerate(sorted_bids[:self.depth_levels]):
            bid_prices[i] = (p - mid_price) / mid_price
            bid_sizes[i] = v
            
        for i, (p, v) in enumerate(sorted_asks[:self.depth_levels]):
            ask_prices[i] = (p - mid_price) / mid_price
            ask_sizes[i] = v
            
        mean_b_depth = np.mean([v for _, v in sorted_bids[:10]]) if sorted_bids else 0.0
        mean_a_depth = np.mean([v for _, v in sorted_asks[:10]]) if sorted_asks else 0.0
        depth_skew = (mean_b_depth - mean_a_depth) / (mean_b_depth + mean_a_depth + 1e-5)
        
        k_events = self.events_processed or 1
        
        empirical_vector = np.array([
            mid_price, spread, best_b, best_a,
            self.acc_ofi, 
            self.acc_volume,
            self.events_processed,
            self.acc_buy_count / k_events,
            self.acc_cancel_count / k_events,
            mean_b_depth,
            mean_a_depth,
            depth_skew
        ], dtype=np.float32)
        
        self.reset_accumulators()
        
        full_tensor = np.concatenate([bid_prices, bid_sizes, ask_prices, ask_sizes, empirical_vector])
        
        return {
            "market_id": self.market_id,
            "timestamp": timestamp,
            "features": full_tensor
        }

def sort_by_numeric_value(paths):
    return sorted(paths, key=lambda p: int(p.name.split("=")[1]))

def process_all_hours(base_dir: Path, output_base: Path, K: int = 20):
    logger.info("Discovering hours to process...")
    snap_base = base_dir / "snapshots"
    ob_base = base_dir / "orderbook"
    
    # Trova tutte le cartelle orarie e ordinale cronologicamente (numericamente!)
    hours = []
    for y in sort_by_numeric_value(snap_base.glob("year=*")):
        for m in sort_by_numeric_value(y.glob("month=*")):
            for d in sort_by_numeric_value(m.glob("day=*")):
                for h in sort_by_numeric_value(d.glob("hour=*")):
                    hours.append(h)
    
    logger.info(f"Trovate {len(hours)} ore totali.")
    
    global_states = {}
    
    for i, snap_dir in enumerate(hours):
        rel_path = snap_dir.relative_to(snap_base)
        ob_dir = ob_base / rel_path
        
        if not ob_dir.exists():
            logger.warning(f"Skipping {rel_path}: No orderbook data.")
            continue
            
        out_dir = output_base / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "lob_features.parquet"
        
        logger.info(f"[{i+1}/{len(hours)}] Processing {rel_path}...")
        t0 = time.time()
        
        try:
            df = load_and_interleave_hour(ob_dir, snap_dir)
        except Exception as e:
            logger.error(f"Error loading {rel_path}: {e}")
            continue
            
        extracted_records = []
        
        for row in df.itertuples(index=False):
            m_id = row.market_id
            
            if m_id not in global_states:
                global_states[m_id] = OrderbookState(market_id=m_id)
                
            state = global_states[m_id]
            state.apply_event(row)
            
            if state.events_processed % K == 0:
                extracted_records.append(state.extract_features(row.timestamp_received))
                
        if extracted_records:
            out_df = pd.DataFrame(extracted_records)
            out_df.to_parquet(out_file, engine='pyarrow', index=False)
            
        t_tot = time.time() - t0
        logger.info(f"Finished {rel_path} in {t_tot:.2f}s. Extracted {len(extracted_records)} tensors. Active Markets in RAM: {len(global_states)}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("lob_extraction.log"),
            logging.StreamHandler()
        ]
    )
    
    base_dir = Path("/data/v.pampinella/agy_project/data/derived/partitioned")
    out_dir = Path("/data/v.pampinella/agy_project/data/derived/tensors")
    
    process_all_hours(base_dir, out_dir, K=20)
