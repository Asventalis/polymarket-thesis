"""
Precomputes the global ID → integer index mappings for markets and wallets.

Markets: loaded from the existing market_node_mapping.csv.
Wallets: extracted from all transaction parquet files (all unique wallet addresses seen).

Output:
  /data/v.pampinella/agy_project/data/derived/id_mappings/market_to_idx.parquet
  /data/v.pampinella/agy_project/data/derived/id_mappings/wallet_to_idx.parquet
"""

import duckdb
import pandas as pd
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/data/v.pampinella/agy_project/data")
OUT_DIR = BASE / "derived/id_mappings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_market_mapping():
    logger.info("Building market mapping from existing market_node_mapping.csv...")
    t0 = time.time()

    df = pd.read_csv(BASE / "derived/market_node_mapping.csv")
    # Keep just the two columns we need, rename for clarity
    df = df[["market_id", "node_index"]].rename(columns={"node_index": "market_idx"})
    df = df.sort_values("market_idx").reset_index(drop=True)

    out = OUT_DIR / "market_to_idx.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Market mapping: {len(df)} markets → saved to {out} ({time.time()-t0:.2f}s)")
    return df


def build_wallet_mapping():
    logger.info("Extracting all unique wallets from transactions (all hours)...")
    t0 = time.time()

    txn_glob = str(BASE / "derived/partitioned/transactions/**/*.parquet")

    # DuckDB can scan all parquet files at once with a glob - very fast
    result = duckdb.query(f"""
        SELECT DISTINCT wallet
        FROM read_parquet('{txn_glob}', hive_partitioning=true)
        WHERE wallet IS NOT NULL
        ORDER BY wallet
    """).df()

    result["wallet_idx"] = range(len(result))
    result = result[["wallet", "wallet_idx"]]

    out = OUT_DIR / "wallet_to_idx.parquet"
    result.to_parquet(out, index=False)
    logger.info(f"Wallet mapping: {len(result)} unique wallets → saved to {out} ({time.time()-t0:.2f}s)")
    return result


if __name__ == "__main__":
    markets = build_market_mapping()
    wallets = build_wallet_mapping()

    print("\n=== Summary ===")
    print(f"Markets: {len(markets)}")
    print(f"Wallets: {len(wallets)}")
    print("\nMarket sample:")
    print(markets.head(3))
    print("\nWallet sample:")
    print(wallets.head(3))
