import sys
import duckdb

def verify_transaction(target_hash):
    con = duckdb.connect("data/preprocessing.duckdb", read_only=True)

    print(f"Scanning partitioned dataset for tx_hash: {target_hash}...")
    
    query = f"""
    SELECT 
        t.timestamp,
        t.wallet,
        t.role,
        t.side,
        t.tokens,
        t.price,
        t.total_usdc,
        t.token_outcome AS newly_mapped_outcome,
        t.token_id AS traded_token_id,
        m.outcome_1 AS market_yes_name,
        m.token_id_1 AS market_yes_token,
        m.outcome_2 AS market_no_name,
        m.token_id_2 AS market_no_token,
        t.market_id,
        t.tx_hash
    FROM read_parquet('data/derived/partitioned/transactions/*/*/*/*/*.parquet') t
    LEFT JOIN universe.markets m ON t.market_id = m.market_id
    WHERE t.tx_hash = '{target_hash}'
    """
    
    df = con.execute(query).df()

    if df.empty:
        print(f"\n[Error] No transaction found with hash: {target_hash}")
    else:
        print("\n==== TRANSACTION INTEGRITY VERIFICATION ====")
        print("Here is the full row cross-referenced against the Market Table:")
        # Transpose the dataframe so it prints vertically like a clean key-value pair list
        print(df.iloc[0].to_frame(name="Value").to_string())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_transaction.py <tx_hash>")
        print("Example: python verify_transaction.py 0xe53c988a54dea75b565d99ccedd208953384af3d723d94d2100c0da092559b07")
        sys.exit(1)
        
    verify_transaction(sys.argv[1])
