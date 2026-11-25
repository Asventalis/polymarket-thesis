import time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import torch
from sklearn.preprocessing import StandardScaler

from polymarket_gnn.data.lob_encoder import ContinuousLOBMessageEngine

class CausalCoordinator:
    """
    Coordinates the temporal data flow for the auto-regressive Dynamic GNN.
    
    Loads RAW orderbook and snapshot data, streams them through the LOBMessageEngine
    causally step-by-step, and yields instantaneous LOB features and transaction edges.
    """
    def __init__(
        self,
        date_str: str,
        start_hour: int = 0,
        end_hour: int = 23,
        n_markets: int = 50,
        n_wallets: int = 2000,
        horizon_steps: int = 150,
    ):
        self.date_str = date_str
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.n_markets = n_markets
        self.n_wallets = n_wallets
        self.horizon_steps = horizon_steps
        
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.db_path = str(self.project_root / "data" / "preprocessing.duckdb")
        
        self.lob_engine = ContinuousLOBMessageEngine(min_events=20, k_slices=4, depth_levels=10)


    def _get_files_expr(self, table_name: str, start_time: str, end_time: str) -> str:
        import pandas as pd
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)
        base_dir = self.project_root / "data" / "derived" / "partitioned" / table_name
        files = []
        h = start.floor("h")
        while h <= end:
            part = base_dir / f"year={h.year}" / f"month={h.month}" / f"day={h.day}" / f"hour={h.hour}"
            files.extend(sorted(part.glob("*.parquet")))
            h += pd.Timedelta(hours=1)
        if not files:
            return f"(SELECT * FROM read_parquet('{base_dir}/**/*.parquet', hive_partitioning=1) WHERE 1=0)"
        return "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"

    def _get_markets_expr(self) -> str:
        markets_file = self.project_root / "data" / "universe_markets.parquet"
        return f"read_parquet('{markets_file}')"

    def _select_markets_and_wallets(self):
        import duckdb
        db = duckdb.connect(':memory:')
        start_time = f"{self.date_str} {self.start_hour:02d}:00:00"
        end_time = f"{self.date_str} {self.end_hour:02d}:59:59"
        tx_expr = self._get_files_expr("transactions", start_time, end_time)
        ob_expr = self._get_files_expr("orderbook", start_time, end_time)
        m_expr = self._get_markets_expr()
        
        # Select All Active Markets
        market_df = db.execute(f"""
            WITH tx_ranked AS (
                SELECT t.market_id, SUM(t.total_usdc) as total_usdc
                FROM {tx_expr} t
                WHERE t.timestamp >= TIMESTAMP '{start_time}'
                  AND t.timestamp <= TIMESTAMP '{end_time}'
                GROUP BY t.market_id
                ORDER BY total_usdc DESC
            ),
            ob_markets AS (
                SELECT DISTINCT market_id
                FROM {ob_expr}
                WHERE timestamp_received >= TIMESTAMP '{start_time}'
                  AND timestamp_received <  TIMESTAMP '{end_time}'
            )
            SELECT tx.market_id, tx.total_usdc,
                   m.token_id_1 as token_id_yes,
                   m.token_id_2 as token_id_no
            FROM tx_ranked tx
            JOIN ob_markets ob ON tx.market_id = ob.market_id
            JOIN {m_expr} m ON tx.market_id = m.market_id
            WHERE m.token_id_1 IS NOT NULL AND m.token_id_2 IS NOT NULL
            ORDER BY tx.total_usdc DESC
        """).df()
        
        self.market_ids = market_df['market_id'].tolist()
        self.market_to_idx = {m: i for i, m in enumerate(self.market_ids)}
        self.token_yes_map = dict(zip(market_df['market_id'], market_df['token_id_yes']))
        self.token_no_map  = dict(zip(market_df['market_id'], market_df['token_id_no']))

        # Select All Active Wallets
        mid_str = ",".join(f"'{m}'" for m in self.market_ids)
        wallets_df = db.execute(f"""
            SELECT wallet, SUM(total_usdc) as usdc
            FROM {tx_expr}
            WHERE market_id IN ({mid_str})
              AND timestamp >= TIMESTAMP '{start_time}'
              AND timestamp <= TIMESTAMP '{end_time}'
            GROUP BY wallet
            ORDER BY usdc DESC
        """).df()
        
        self.wallet_ids = wallets_df['wallet'].tolist()
        self.wallet_to_idx = {w: i for i, w in enumerate(self.wallet_ids)}
        db.close()
        
        self.n_markets = len(self.market_ids)
        self.n_wallets = len(self.wallet_ids)
        
        print(f"Selected ALL {self.n_markets} active markets and {self.n_wallets} active wallets.")

    def _load_raw_orderbook_and_snapshots(self):
        mid_str = ",".join(f"'{m}'" for m in self.market_ids)
        start_time = f"{self.date_str} {self.start_hour:02d}:00:00"
        end_time   = f"{self.date_str} {self.end_hour:02d}:59:59"
        
        import duckdb
        db = duckdb.connect(':memory:')
        ob_expr = self._get_files_expr("orderbook", start_time, end_time)
        snap_expr = self._get_files_expr("snapshots", start_time, end_time)
        
        print("Loading RAW LOB ticks...")
        # LOB ticks
        self.ob_df = db.execute(f"""
            SELECT *
            FROM {ob_expr}
            WHERE market_id IN ({mid_str})
              AND timestamp_received >= TIMESTAMP '{start_time}'
              AND timestamp_received <= TIMESTAMP '{end_time}'
            ORDER BY timestamp_received ASC
        """).df()
        
        print("Loading RAW LOB snapshots...")
        # LOB snapshots
        self.snap_df = db.execute(f"""
            SELECT *
            FROM {snap_expr}
            WHERE market_id IN ({mid_str})
              AND timestamp_received >= TIMESTAMP '{start_time}'
              AND timestamp_received <= TIMESTAMP '{end_time}'
            ORDER BY timestamp_received ASC
        """).df()
        
        # Build 2-second Grid
        self.ob_df['timestamp_received'] = pd.to_datetime(self.ob_df['timestamp_received'], utc=True)
        self.snap_df['timestamp_received'] = pd.to_datetime(self.snap_df['timestamp_received'], utc=True)
        
        all_ts = pd.concat([self.ob_df['timestamp_received'], self.snap_df['timestamp_received']])
        
        if all_ts.empty:
            self.grid_times = np.array([])
            print("WARNING: No orderbook data found for this day!")
            return
            
        min_ts = all_ts.min().ceil('2s')
        max_ts = all_ts.max().ceil('2s')
        
        self.grid_times = pd.date_range(start=min_ts, end=max_ts, freq='2s')
        
        # Precompute targets (price jumps). This requires knowing mid_price at the grid points.
        # Since we use the raw stream for features, we will just sample mid_price at the grid points.
        self.ob_df['grid_time'] = self.ob_df['timestamp_received'].dt.ceil('2s')
        grid_mid = self.ob_df.groupby(['market_id', 'grid_time'])['mid_price'].last().reset_index()
        grid_mid = grid_mid.set_index(['grid_time', 'market_id'])['mid_price'].unstack()
        grid_mid = grid_mid.reindex(self.grid_times, method='ffill')
        
        # Binary target: +1 if price goes up by 0.01 in horizon_steps
        targets = grid_mid.shift(-self.horizon_steps) - grid_mid
        self.Y = (targets >= 0.01).astype(float).values
        self.Y[targets.isna()] = np.nan
        
        print(f"Loaded Raw Streams. Grid: {len(self.grid_times)} steps.")

    def _load_transactions_by_step(self):
        mid_str = ",".join(f"'{m}'" for m in self.market_ids)
        wal_str = ",".join(f"'{w}'" for w in self.wallet_ids)
        
        start_time = f"{self.date_str} {self.start_hour:02d}:00:00"
        end_time   = f"{self.date_str} {self.end_hour:02d}:59:59"
        
        import duckdb
        db = duckdb.connect(':memory:')
        tx_expr = self._get_files_expr("transactions", start_time, end_time)
        tx_df = db.execute(f"""
            SELECT t.timestamp, t.wallet, t.market_id,
                   t.token_id, t.side, t.total_usdc
            FROM {tx_expr} t
            WHERE t.market_id IN ({mid_str})
              AND t.wallet    IN ({wal_str})
              AND t.timestamp >= TIMESTAMP '{start_time}'
              AND t.timestamp <= TIMESTAMP '{end_time}'
        """).df()
        db.close()
        
        # Determine relation
        yes_tokens = set(self.token_yes_map.values())
        no_tokens  = set(self.token_no_map.values())
        
        def get_relation(row):
            if row['token_id'] in yes_tokens:
                return 'yes_buy' if row['side'] == 'BUY' else 'yes_sell'
            elif row['token_id'] in no_tokens:
                return 'no_buy'  if row['side'] == 'BUY' else 'no_sell'
            return None
            
        tx_df['relation'] = tx_df.apply(get_relation, axis=1)
        tx_df = tx_df.dropna(subset=['relation'])
        
        # Snap to 2-second grid (ceiling so it lands on the next step)
        tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], utc=True)
        tx_df['grid_time'] = tx_df['timestamp'].dt.ceil('2s')
        
        tx_df['w_idx'] = tx_df['wallet'].map(self.wallet_to_idx)
        tx_df['m_idx'] = tx_df['market_id'].map(self.market_to_idx)
        
        # Pre-group by t_idx (map grid_time to t_idx)
        t_map = {t: i for i, t in enumerate(self.grid_times)}
        tx_df['t_idx'] = tx_df['grid_time'].map(t_map)
        tx_df = tx_df.dropna(subset=['t_idx'])
        tx_df['t_idx'] = tx_df['t_idx'].astype(int)
        
        self.tx_by_step = tx_df.groupby('t_idx')
        print(f"Loaded {len(tx_df)} valid transactions snapped to {len(self.tx_by_step)} active steps.")

    def build(self):
        t0 = time.time()
        self._select_markets_and_wallets()
        self._load_raw_orderbook_and_snapshots()
        self._load_transactions_by_step()
        print(f"Dataset completely built in {time.time() - t0:.1f}s")
        
        # Split chronological
        if not hasattr(self, 'grid_times') or len(self.grid_times) == 0:
            return
            
        T = len(self.grid_times)
        split = int(T * 0.7)
        self.train_indices = np.arange(0, split)
        self.val_indices = np.arange(split, T - self.horizon_steps)

    def get_step(self, t_idx: int, device: torch.device):
        """
        Yields instantaneous features and edges for EXACTLY step t.
        Returns:
            emp_feats: [N_markets, S, 12]
            spatial_feats: [N_markets, S, 5, k_slices, depth_levels]
            edge_index_dict, edge_weight_dict
            y: [N_markets]
            mask: [N_markets]
        Where S is the max number of LOB events that occurred in this 2s window across all markets.
        If S=0, returns empty tensors.
        """
        t_query = self.grid_times[t_idx]
        
        # 1. Run the LOB Engine to get rich orderbook events up to t_query
        lob_updates = self.lob_engine.step(self.market_ids, self.ob_df, self.snap_df, t_query)
        
        # Collate variable-length event sequences per market into a padded batch
        # shape needed: emp [N_m, S, 12], spatial [N_m, S, 5, K, D]
        N_m = len(self.market_ids)
        max_s = max([len(seq) for seq in lob_updates.values()]) if lob_updates else 0
        
        if max_s > 0:
            emp_feats = torch.zeros(N_m, max_s, 12, dtype=torch.float32, device=device)
            spatial_feats = torch.zeros(N_m, max_s, 5, self.lob_engine.k_slices, self.lob_engine.depth_levels, dtype=torch.float32, device=device)
            valid_seq_mask = torch.zeros(N_m, max_s, dtype=torch.bool, device=device)
            
            for m_id, seq in lob_updates.items():
                m_idx = self.market_to_idx[m_id]
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
            spatial_feats = torch.zeros(N_m, 0, 5, self.lob_engine.k_slices, self.lob_engine.depth_levels, dtype=torch.float32, device=device)
            valid_seq_mask = torch.zeros(N_m, 0, dtype=torch.bool, device=device)
            
        # Targets
        y_raw = self.Y[t_idx]
        mask = torch.tensor(~np.isnan(y_raw), dtype=torch.bool, device=device)
        y = torch.tensor(np.nan_to_num(y_raw, nan=0.0), dtype=torch.float32, device=device)
        
        # 2. Instantaneous Edges
        edge_index_dict = {}
        edge_weight_dict = {}
        RELATIONS = ['yes_buy', 'yes_sell', 'no_buy', 'no_sell']
        
        if hasattr(self, 'tx_by_step') and t_idx in self.tx_by_step.groups:
            tx_group = self.tx_by_step.get_group(t_idx)
            agg = tx_group.groupby(['relation', 'w_idx', 'm_idx'])['total_usdc'].sum().reset_index()
            
            for rel in RELATIONS:
                sub = agg[agg['relation'] == rel]
                if not sub.empty:
                    ei = torch.tensor(np.stack([sub['w_idx'].values, sub['m_idx'].values]), dtype=torch.long)
                    ew = torch.tensor(sub['total_usdc'].values, dtype=torch.float32)
                    edge_index_dict[rel] = ei.to(device)
                    edge_weight_dict[rel] = ew.to(device)
                else:
                    edge_index_dict[rel] = torch.zeros(2, 0, dtype=torch.long, device=device)
                    edge_weight_dict[rel] = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            for rel in RELATIONS:
                edge_index_dict[rel] = torch.zeros(2, 0, dtype=torch.long, device=device)
                edge_weight_dict[rel] = torch.zeros(0, dtype=torch.float32, device=device)
                
        return emp_feats, spatial_feats, valid_seq_mask, edge_index_dict, edge_weight_dict, y, mask
