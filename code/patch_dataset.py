import re
from pathlib import Path

path = Path("src/polymarket_gnn/data/causal_dataset.py")
content = path.read_text()

# 1. Remove the old _get_db
content = re.sub(r'    def _get_db\(self\):.*?            return db\n', '', content, flags=re.DOTALL)

# 2. Add the _get_files_expr helper
helper = """    def _get_files_expr(self, table_name: str, start_time: str, end_time: str) -> str:
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
"""
content = content.replace("    def _select_markets_and_wallets(self):", helper + "\n    def _select_markets_and_wallets(self):")

# 3. Fix _select_markets_and_wallets
old_select = """    def _select_markets_and_wallets(self):
        db = self._get_db()
        
        # Select Top Markets
        market_df = db.execute(f\"\"\"
            WITH tx_ranked AS (
                SELECT t.market_id, SUM(t.total_usdc) as total_usdc
                FROM universe.transactions t
                GROUP BY t.market_id
                ORDER BY total_usdc DESC
                LIMIT {self.n_markets * 10}
            ),
            ob_markets AS (
                SELECT DISTINCT market_id
                FROM universe.orderbook
                WHERE timestamp_received >= TIMESTAMP '{self.date_str} 00:00:00'
                  AND timestamp_received <  TIMESTAMP '{self.date_str} 23:59:59'
            )
            SELECT tx.market_id, tx.total_usdc,
                   m.token_id_1 as token_id_yes,
                   m.token_id_2 as token_id_no
            FROM tx_ranked tx
            JOIN ob_markets ob ON tx.market_id = ob.market_id
            JOIN universe.markets m ON tx.market_id = m.market_id
            WHERE m.token_id_1 IS NOT NULL AND m.token_id_2 IS NOT NULL
            ORDER BY tx.total_usdc DESC
            LIMIT {self.n_markets}
        \"\"\").df()"""

new_select = """    def _select_markets_and_wallets(self):
        import duckdb
        db = duckdb.connect(':memory:')
        start_time = f"{self.date_str} 00:00:00"
        end_time = f"{self.date_str} 23:59:59"
        tx_expr = self._get_files_expr("transactions", start_time, end_time)
        ob_expr = self._get_files_expr("orderbook", start_time, end_time)
        m_expr = self._get_markets_expr()
        
        # Select Top Markets
        market_df = db.execute(f\"\"\"
            WITH tx_ranked AS (
                SELECT t.market_id, SUM(t.total_usdc) as total_usdc
                FROM {tx_expr} t
                WHERE t.timestamp >= TIMESTAMP '{start_time}'
                  AND t.timestamp <= TIMESTAMP '{end_time}'
                GROUP BY t.market_id
                ORDER BY total_usdc DESC
                LIMIT {self.n_markets * 10}
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
            LIMIT {self.n_markets}
        \"\"\").df()"""
content = content.replace(old_select, new_select)

# 4. Fix Top Wallets
old_wallets = """        # Select Top Wallets
        mid_str = ",".join(f"'{m}'" for m in self.market_ids)
        wallets_df = db.execute(f\"\"\"
            SELECT wallet, SUM(total_usdc) as usdc
            FROM universe.transactions
            WHERE market_id IN ({mid_str})
            GROUP BY wallet
            ORDER BY usdc DESC
            LIMIT {self.n_wallets}
        \"\"\").df()"""

new_wallets = """        # Select Top Wallets
        mid_str = ",".join(f"'{m}'" for m in self.market_ids)
        wallets_df = db.execute(f\"\"\"
            SELECT wallet, SUM(total_usdc) as usdc
            FROM {tx_expr}
            WHERE market_id IN ({mid_str})
              AND timestamp >= TIMESTAMP '{start_time}'
              AND timestamp <= TIMESTAMP '{end_time}'
            GROUP BY wallet
            ORDER BY usdc DESC
            LIMIT {self.n_wallets}
        \"\"\").df()"""
content = content.replace(old_wallets, new_wallets)

# 5. Fix _load_raw_orderbook_and_snapshots
old_load1 = """        start_time = f"{self.date_str} 00:00:00"
        end_time   = f"{self.date_str} 23:59:59"
        
        db = self._get_db()
        
        print("Loading RAW LOB ticks...")
        # LOB ticks
        self.ob_df = db.execute(f\"\"\"
            SELECT *
            FROM universe.orderbook"""

new_load1 = """        start_time = f"{self.date_str} 00:00:00"
        end_time   = f"{self.date_str} 23:59:59"
        
        import duckdb
        db = duckdb.connect(':memory:')
        ob_expr = self._get_files_expr("orderbook", start_time, end_time)
        snap_expr = self._get_files_expr("snapshots", start_time, end_time)
        
        print("Loading RAW LOB ticks...")
        # LOB ticks
        self.ob_df = db.execute(f\"\"\"
            SELECT *
            FROM {ob_expr}"""
content = content.replace(old_load1, new_load1)

content = content.replace("FROM universe.snapshots", "FROM {snap_expr}")

# 6. Fix _load_transactions_by_step
old_load2 = """        start_time = f"{self.date_str} 00:00:00"
        end_time   = f"{self.date_str} 23:59:59"
        
        db = self._get_db()
        tx_df = db.execute(f\"\"\"
            SELECT t.timestamp, t.wallet, t.market_id,
                   t.token_id, t.side, t.total_usdc
            FROM universe.transactions t"""

new_load2 = """        start_time = f"{self.date_str} 00:00:00"
        end_time   = f"{self.date_str} 23:59:59"
        
        import duckdb
        db = duckdb.connect(':memory:')
        tx_expr = self._get_files_expr("transactions", start_time, end_time)
        tx_df = db.execute(f\"\"\"
            SELECT t.timestamp, t.wallet, t.market_id,
                   t.token_id, t.side, t.total_usdc
            FROM {tx_expr} t"""
content = content.replace(old_load2, new_load2)

path.write_text(content)
print("Patched causal_dataset.py successfully")
