-- Compute 2s transaction features
CREATE TEMP TABLE tx_2s_buckets AS
SELECT 
    market_id,
    time_bucket(INTERVAL '2 seconds', timestamp) as grid_time,
    SUM(total_usdc) as volume_2s,
    COUNT(tx_hash) as num_trades_2s,
    SUM(CASE WHEN side = 'BUY' THEN total_usdc ELSE 0 END) as buy_volume_2s,
    SUM(CASE WHEN side = 'SELL' THEN total_usdc ELSE 0 END) as sell_volume_2s,
    COUNT(DISTINCT wallet) as unique_wallets_2s
FROM db.transactions_with_market
WHERE 
    timestamp >= TIMESTAMP '$start_time' 
    AND timestamp <= TIMESTAMP '$end_time'
GROUP BY market_id, grid_time;

CREATE TABLE transaction_features_2s AS
SELECT 
    g.market_id,
    g.grid_time,
    COALESCE(t.volume_2s, 0.0) as volume_2s,
    COALESCE(t.num_trades_2s, 0) as num_trades_2s,
    COALESCE(t.buy_volume_2s, 0.0) as buy_volume_2s,
    COALESCE(t.sell_volume_2s, 0.0) as sell_volume_2s,
    COALESCE(t.unique_wallets_2s, 0) as unique_wallets_2s
FROM market_time_grid g
LEFT JOIN tx_2s_buckets t
    ON g.market_id = t.market_id
    AND g.grid_time = t.grid_time;
