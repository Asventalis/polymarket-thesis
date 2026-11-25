DROP TABLE IF EXISTS market_edges_2s;
DROP TABLE IF EXISTS wallet_degrees;
DROP TABLE IF EXISTS recent_trades;
DROP TABLE IF EXISTS dev_tx;

-- Create a small temp table of transactions in our dev window
CREATE TEMP TABLE dev_tx AS
SELECT market_id, timestamp, wallet
FROM db.transactions_with_market
WHERE timestamp >= TIMESTAMP '$lookback_start_time' 
  AND timestamp <= TIMESTAMP '$end_time';

-- Compute recent trades for co-trading graph construction
CREATE TEMP TABLE recent_trades AS
SELECT 
    g.market_id,
    g.grid_time,
    t.wallet
FROM market_time_grid g
JOIN dev_tx t
  ON t.market_id = g.market_id
  AND t.timestamp > g.grid_time - INTERVAL $memory_seconds SECOND
  AND t.timestamp <= g.grid_time;

-- Compute wallet degrees (how many unique markets a wallet traded in the 20s window)
CREATE TEMP TABLE wallet_degrees AS
SELECT 
    grid_time, 
    wallet, 
    COUNT(DISTINCT market_id) as degree
FROM recent_trades
GROUP BY grid_time, wallet;

-- Build inverse wallet-degree weighted co-trading graph
CREATE TABLE market_edges_2s AS
SELECT 
    r1.grid_time,
    r1.market_id AS market_id_1,
    r2.market_id AS market_id_2,
    SUM(1.0 / d.degree) as edge_weight,
    COUNT(DISTINCT r1.wallet) as co_trades
FROM recent_trades r1
JOIN recent_trades r2
  ON r1.grid_time = r2.grid_time
  AND r1.wallet = r2.wallet
  AND r1.market_id < r2.market_id
JOIN wallet_degrees d
  ON r1.grid_time = d.grid_time
  AND r1.wallet = d.wallet
GROUP BY r1.grid_time, r1.market_id, r2.market_id;
