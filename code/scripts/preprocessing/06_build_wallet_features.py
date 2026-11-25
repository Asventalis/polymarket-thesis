"""Build unified pre-sorted Parquet datasets with wallet flow features for all available days.

This replaces the old optimize_storage.py. It now includes:
- Orderbook features (best_bid, best_ask, spread, mid_price)
- Transaction features (volume_2s, num_trades_2s, buy_volume_2s, sell_volume_2s)
- Target (target_10s)
- NEW: Signed wallet order-flow features (net_signed_usdc, order_flow_imbalance, etc.)

The wallet flow features encode YES/NO × BUY/SELL direction from the transaction stream,
which is the primary signal identified in the thesis proposal.
"""

import time
from pathlib import Path
import duckdb


def build_wallet_enriched_dataset(date_str: str, db_path: str, out_dir: Path, num_markets: int = 500):
    out_file = out_dir / f"{date_str}.parquet"
    print(f"\nBuilding wallet-enriched unified Parquet for {date_str}...")
    start_t = time.time()

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"ATTACH '{db_path}' AS db (READ_ONLY);")

    # Load SQL template
    sql_dir = Path(__file__).resolve().parent.parent.parent.parent / "sql"
    wallet_sql = (sql_dir / "07_wallet_flow_features_2s.sql").read_text()

    # Build everything inline using COPY with a CTE chain
    query = f"""
    COPY (
        WITH selected_markets AS (
            SELECT DISTINCT market_id
            FROM read_parquet('{out_dir.parent / "targets" / date_str}.parquet')
            LIMIT {num_markets}
        ),

        -- Market-time grid (same as SQL pipeline)
        time_grid AS (
            SELECT t AS grid_time
            FROM generate_series(
                (SELECT MIN(grid_time) FROM read_parquet('{out_dir.parent / "targets" / date_str}.parquet')),
                (SELECT MAX(grid_time) FROM read_parquet('{out_dir.parent / "targets" / date_str}.parquet')),
                INTERVAL 2 SECOND
            ) AS t(t)
        ),

        market_time_grid AS (
            SELECT sm.market_id, tg.grid_time
            FROM selected_markets sm
            CROSS JOIN time_grid tg
        ),

        -- Wallet flow: signed YES/NO x BUY/SELL per market per 2s bucket
        tx_signed AS (
            SELECT
                t.wallet,
                t.timestamp,
                t.market_id,
                t.total_usdc,
                CASE
                    WHEN t.token_id = m.token_id_1 THEN 'YES'
                    WHEN t.token_id = m.token_id_2 THEN 'NO'
                    ELSE 'UNKNOWN'
                END AS token_outcome,
                CASE
                    WHEN t.token_id = m.token_id_1 AND t.side = 'BUY'  THEN  1.0
                    WHEN t.token_id = m.token_id_1 AND t.side = 'SELL' THEN -1.0
                    WHEN t.token_id = m.token_id_2 AND t.side = 'BUY'  THEN -1.0
                    WHEN t.token_id = m.token_id_2 AND t.side = 'SELL' THEN  1.0
                    ELSE 0.0
                END AS signed_direction
            FROM db.transactions_with_market t
            JOIN db.markets_own_cleaned m ON t.market_id = m.market_id
            WHERE
                t.timestamp >= (SELECT MIN(grid_time) FROM read_parquet('{out_dir.parent / "targets" / date_str}.parquet')) - INTERVAL 2 SECOND
                AND t.timestamp <= (SELECT MAX(grid_time) FROM read_parquet('{out_dir.parent / "targets" / date_str}.parquet'))
                AND t.market_id IN (SELECT market_id FROM selected_markets)
        ),

        wallet_flow AS (
            SELECT
                market_id,
                time_bucket(INTERVAL '2 seconds', timestamp) AS grid_time,
                SUM(signed_direction * total_usdc)  AS net_signed_usdc,
                SUM(total_usdc)                     AS total_flow_usdc,
                CASE WHEN SUM(total_usdc) > 0
                     THEN SUM(signed_direction * total_usdc) / SUM(total_usdc)
                     ELSE 0.0 END                   AS order_flow_imbalance,
                SUM(CASE WHEN token_outcome = 'YES' AND signed_direction > 0 THEN total_usdc ELSE 0 END) AS yes_buy_usdc,
                SUM(CASE WHEN token_outcome = 'YES' AND signed_direction < 0 THEN total_usdc ELSE 0 END) AS yes_sell_usdc,
                SUM(CASE WHEN token_outcome = 'NO'  AND signed_direction < 0 THEN total_usdc ELSE 0 END) AS no_buy_usdc,
                SUM(CASE WHEN token_outcome = 'NO'  AND signed_direction > 0 THEN total_usdc ELSE 0 END) AS no_sell_usdc,
                MAX(total_usdc)                     AS max_single_wallet_usdc,
                COUNT(DISTINCT wallet)              AS flow_n_unique_wallets,
                COUNT(DISTINCT CASE WHEN signed_direction > 0 THEN wallet END) AS n_bullish_wallets,
                COUNT(DISTINCT CASE WHEN signed_direction < 0 THEN wallet END) AS n_bearish_wallets
            FROM tx_signed
            WHERE token_outcome IN ('YES', 'NO')
            GROUP BY market_id, grid_time
        )

        SELECT
            o.grid_time,
            o.market_id,
            -- Orderbook features
            o.mid_price, o.spread, o.best_bid, o.best_ask,
            -- Transaction volume features
            tx.volume_2s, tx.num_trades_2s, tx.buy_volume_2s, tx.sell_volume_2s, tx.unique_wallets_2s,
            -- NEW: Signed wallet order-flow features
            COALESCE(w.net_signed_usdc,        0.0) AS net_signed_usdc,
            COALESCE(w.total_flow_usdc,        0.0) AS total_flow_usdc,
            COALESCE(w.order_flow_imbalance,   0.0) AS order_flow_imbalance,
            COALESCE(w.yes_buy_usdc,           0.0) AS yes_buy_usdc,
            COALESCE(w.yes_sell_usdc,          0.0) AS yes_sell_usdc,
            COALESCE(w.no_buy_usdc,            0.0) AS no_buy_usdc,
            COALESCE(w.no_sell_usdc,           0.0) AS no_sell_usdc,
            COALESCE(w.max_single_wallet_usdc, 0.0) AS max_single_wallet_usdc,
            COALESCE(w.flow_n_unique_wallets,  0)   AS flow_n_unique_wallets,
            COALESCE(w.n_bullish_wallets,      0)   AS n_bullish_wallets,
            COALESCE(w.n_bearish_wallets,      0)   AS n_bearish_wallets,
            -- Target
            t.target_10s
        FROM read_parquet('{out_dir.parent / "orderbook" / date_str}.parquet') o
        JOIN read_parquet('{out_dir.parent / "transactions" / date_str}.parquet') tx
          ON o.market_id = tx.market_id AND o.grid_time = tx.grid_time
        JOIN read_parquet('{out_dir.parent / "targets" / date_str}.parquet') t
          ON o.market_id = t.market_id AND o.grid_time = t.grid_time
        JOIN selected_markets sm ON o.market_id = sm.market_id
        LEFT JOIN wallet_flow w ON o.market_id = w.market_id AND o.grid_time = w.grid_time
        WHERE t.target_10s IS NOT NULL
        ORDER BY o.grid_time ASC, o.market_id ASC
    ) TO '{out_file}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'SNAPPY');
    """

    con.execute(query)
    elapsed = time.time() - start_t
    print(f"  Done: {out_file} in {elapsed:.1f}s")

    # Quick read benchmark
    bench_start = time.time()
    df = con.execute(f"SELECT * FROM read_parquet('{out_file}') LIMIT 7200").df()
    bench_elapsed = time.time() - bench_start
    n_cols = len(df.columns)
    print(f"  Benchmark: {len(df):,} rows × {n_cols} cols in {bench_elapsed*1000:.0f}ms")
    print(f"  Wallet flow columns: {[c for c in df.columns if c not in ['grid_time','market_id','mid_price','spread','best_bid','best_ask','volume_2s','num_trades_2s','buy_volume_2s','sell_volume_2s','unique_wallets_2s','target_10s']]}")


def build_all(num_markets: int = 500):
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    db_path = project_root / "data" / "preprocessing.duckdb"
    multi_day_dir = project_root / "data" / "derived" / "multi_day"
    unified_dir = multi_day_dir / "unified_v2"  # New version with wallet features
    unified_dir.mkdir(parents=True, exist_ok=True)

    target_files = sorted((multi_day_dir / "targets").glob("*.parquet"))
    print(f"Found {len(target_files)} daily datasets to enrich with wallet flow features.")
    
    # TEST SET LOCKED: Never process test set dates (March 24-26)
    LOCKED_TEST_DATES = {"2026-03-24", "2026-03-25", "2026-03-26"}

    for t_file in target_files:
        date_str = t_file.stem
        if date_str in LOCKED_TEST_DATES:
            print(f"  SKIPPING {date_str} — TEST SET LOCKED")
            continue
        build_wallet_enriched_dataset(date_str, str(db_path), unified_dir, num_markets)

    print(f"\nAll done. Wallet-enriched unified datasets at: {unified_dir}")


if __name__ == "__main__":
    build_all()
