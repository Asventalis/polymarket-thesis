-- Create 2-second time grid for 1-hour window
CREATE TEMP TABLE time_grid_2s AS
SELECT t AS grid_time
FROM generate_series(
    TIMESTAMP '$start_time',
    TIMESTAMP '$end_time',
    INTERVAL 2 SECOND
) AS t(t);

-- Create market time grid
CREATE TEMP TABLE market_time_grid AS
SELECT m.market_id, t.grid_time
FROM active_markets m
CROSS JOIN time_grid_2s t;

-- Clean orderbook events around the window
CREATE TEMP TABLE orderbook_dev_window AS
SELECT 
    market_id,
    to_timestamp(timestamp_received / 1000.0) as event_time,
    arg_max(mid_price, timestamp_received) as mid_price,
    arg_max(spread, timestamp_received) as spread,
    arg_max(best_bid, timestamp_received) as best_bid,
    arg_max(best_ask, timestamp_received) as best_ask
FROM db.orderbook
WHERE 
    to_timestamp(timestamp_received / 1000.0) >= TIMESTAMP '$start_time' - INTERVAL 1 HOUR 
    AND to_timestamp(timestamp_received / 1000.0) <= TIMESTAMP '$end_time'
    AND market_id IN (SELECT market_id FROM db.markets_universe)
GROUP BY market_id, event_time;

-- Compute 2s orderbook features using ASOF JOIN
CREATE TABLE orderbook_features_2s AS
SELECT 
    g.market_id,
    g.grid_time,
    o.mid_price,
    o.spread,
    o.best_bid,
    o.best_ask
FROM market_time_grid g
ASOF LEFT JOIN orderbook_dev_window o
    ON g.market_id = o.market_id
    AND g.grid_time >= o.event_time;
