import duckdb
import pandas as pd

class GraphSnapshotGenerator:
    def __init__(self, db_path="data/preprocessing.duckdb", events_path="data/derived/market_lifecycle_events.parquet"):
        self.con = duckdb.connect(db_path, read_only=True)
        self.events_path = events_path
        
        # In-memory dictionary tracking which nodes are currently alive in the graph
        self.active_markets = set()
        
        # Load the entire event queue into memory for instant chronological stepping
        print("Loading chronological market lifecycle queue...")
        self.event_queue = self.con.execute(f"SELECT * FROM read_parquet('{self.events_path}') ORDER BY event_timestamp ASC").df()
        
    def update_state(self, t1, t2):
        """
        Advances the event queue to t2, dynamically spawning and purging market nodes.
        """
        window_events = self.event_queue[
            (self.event_queue['event_timestamp'] > t1) & 
            (self.event_queue['event_timestamp'] <= t2)
        ]
        
        for _, row in window_events.iterrows():
            if row['event_type'] == 'OPEN':
                self.active_markets.add(row['market_id'])
            elif row['event_type'] == 'CLOSE':
                self.active_markets.discard(row['market_id'])
                
    def extract_edges(self, t1, t2):
        """
        Extracts the Transaction Edge tensors for all currently active markets.
        Computes the vectorized interaction effect.
        """
        if not self.active_markets:
            return pd.DataFrame()
            
        # Format the active markets for the SQL IN clause
        market_list = "('" + "','".join(self.active_markets) + "')"
        
        # DuckDB query to extract and vector-math the edges directly from the partitioned dataset
        query = f"""
        SELECT 
            wallet AS source_wallet,
            market_id AS target_market,
            timestamp,
            price,
            tokens AS size,
            fee_rate,
            -- Calculate the positive/negative predictive flow tensor
            token_direction * (CASE WHEN side = 'BUY' THEN 1 ELSE -1 END) AS effect_tensor
        FROM read_parquet('data/derived/partitioned/transactions/*/*/*/*/*.parquet')
        WHERE timestamp > '{t1}' AND timestamp <= '{t2}'
        AND market_id IN {market_list}
        """
        
        edges = self.con.execute(query).df()
        return edges

# ---- Phase 3: Verification Script ----
if __name__ == "__main__":
    generator = GraphSnapshotGenerator()
    
    # Let's simulate a chronological step-through over 6 hours on March 6, 2026
    start_time = pd.to_datetime("2026-03-06 00:00:00")
    
    # First, fast-forward the state dictionary up to the start time
    # (Applying all OPEN/CLOSE events that happened historically before our simulation starts)
    print(f"Fast-forwarding graph state to start time {start_time}...")
    generator.update_state(pd.to_datetime("2000-01-01"), start_time)
    print(f"Initial State: {len(generator.active_markets)} active markets.\n")
    
    # Now simulate generating 1-hour snapshots
    for hour in range(1, 7):
        t1 = start_time + pd.Timedelta(hours=hour-1)
        t2 = start_time + pd.Timedelta(hours=hour)
        
        generator.update_state(t1, t2)
        edges = generator.extract_edges(t1, t2)
        
        print(f"Snapshot [{t1} -> {t2}]:")
        print(f"  -> Active Market Nodes: {len(generator.active_markets)}")
        print(f"  -> Extracted Transaction Edges: {len(edges)}")
        
        if not edges.empty:
            print("  -> Sample Edge Vector:")
            print("     " + edges.iloc[0].to_string().replace('\n', '\n     '))
        print("-" * 50)
