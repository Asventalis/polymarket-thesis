-- 07_wallet_flow_features_2s.sql
-- Compute signed wallet order-flow features per market per 2-second grid step.
-- These features encode direction and conviction of trading activity:
--   - net_signed_usdc:        YES-side buying pressure minus selling pressure
--   - order_flow_imbalance:   net flow normalized by total volume (in [-1, 1])
--   - whale_buy_usdc:         USDC from top-10 wallets by volume doing YES-BUY
--   - n_informed_buyers:      wallets buying YES that are in the top-percentile by historical volume
--   - unique_yes_buyers:      count of distinct wallets taking YES-BUY positions
--   - unique_no_buyers:       count of distinct wallets taking NO-BUY positions (bearish)

-- Step 1: Identify YES vs NO token per transaction
CREATE TEMP TABLE tx_with_side AS
SELECT
    t.wallet,
    t.timestamp,
    t.market_id,
    t.side,       -- BUY or SELL in terms of the token they transacted
    t.total_usdc,
    t.tokens,
    t.price,
    CASE 
        WHEN t.token_id = m.token_id_1 THEN 'YES'
        WHEN t.token_id = m.token_id_2 THEN 'NO'
        ELSE 'UNKNOWN'
    END AS token_outcome
FROM db.transactions_with_market t
JOIN db.markets_own_cleaned m ON t.market_id = m.market_id
WHERE
    t.timestamp >= TIMESTAMP '$start_time'
    AND t.timestamp <= TIMESTAMP '$end_time'
    AND t.market_id IN (SELECT market_id FROM active_markets);

-- Step 2: Compute signed direction for each transaction
-- YES-BUY = +1 (bullish), YES-SELL = -1 (bearish)
-- NO-BUY  = -1 (bearish on YES), NO-SELL = +1 (bullish on YES)
CREATE TEMP TABLE tx_signed AS
SELECT
    wallet,
    timestamp,
    market_id,
    total_usdc,
    token_outcome,
    side,
    CASE
        WHEN token_outcome = 'YES' AND side = 'BUY'  THEN  1.0
        WHEN token_outcome = 'YES' AND side = 'SELL' THEN -1.0
        WHEN token_outcome = 'NO'  AND side = 'BUY'  THEN -1.0
        WHEN token_outcome = 'NO'  AND side = 'SELL' THEN  1.0
        ELSE 0.0
    END AS signed_direction
FROM tx_with_side
WHERE token_outcome IN ('YES', 'NO');

-- Step 3: Aggregate to 2-second grid buckets per market
CREATE TEMP TABLE wallet_flow_2s_buckets AS
SELECT
    market_id,
    time_bucket(INTERVAL '2 seconds', timestamp) AS grid_time,
    SUM(signed_direction * total_usdc)  AS net_signed_usdc,
    SUM(total_usdc)                     AS total_flow_usdc,
    -- Order flow imbalance: net signed / total (in [-1, 1])
    CASE WHEN SUM(total_usdc) > 0
         THEN SUM(signed_direction * total_usdc) / SUM(total_usdc)
         ELSE 0.0
    END AS order_flow_imbalance,
    -- Separate YES-BUY and NO-BUY pressure
    SUM(CASE WHEN token_outcome = 'YES' AND side = 'BUY'  THEN total_usdc ELSE 0.0 END) AS yes_buy_usdc,
    SUM(CASE WHEN token_outcome = 'YES' AND side = 'SELL' THEN total_usdc ELSE 0.0 END) AS yes_sell_usdc,
    SUM(CASE WHEN token_outcome = 'NO'  AND side = 'BUY'  THEN total_usdc ELSE 0.0 END) AS no_buy_usdc,
    SUM(CASE WHEN token_outcome = 'NO'  AND side = 'SELL' THEN total_usdc ELSE 0.0 END) AS no_sell_usdc,
    -- Whale signal: max single-wallet USDC in this bucket (concentration proxy)
    MAX(total_usdc)                     AS max_single_wallet_usdc,
    COUNT(DISTINCT wallet)              AS n_unique_wallets,
    COUNT(DISTINCT CASE WHEN signed_direction > 0 THEN wallet END) AS n_bullish_wallets,
    COUNT(DISTINCT CASE WHEN signed_direction < 0 THEN wallet END) AS n_bearish_wallets
FROM tx_signed
GROUP BY market_id, grid_time;

-- Step 4: Join to the full market-time grid (fill zeros for missing buckets)
CREATE TABLE wallet_flow_features_2s AS
SELECT
    g.market_id,
    g.grid_time,
    COALESCE(w.net_signed_usdc,        0.0) AS net_signed_usdc,
    COALESCE(w.total_flow_usdc,        0.0) AS total_flow_usdc,
    COALESCE(w.order_flow_imbalance,   0.0) AS order_flow_imbalance,
    COALESCE(w.yes_buy_usdc,           0.0) AS yes_buy_usdc,
    COALESCE(w.yes_sell_usdc,          0.0) AS yes_sell_usdc,
    COALESCE(w.no_buy_usdc,            0.0) AS no_buy_usdc,
    COALESCE(w.no_sell_usdc,           0.0) AS no_sell_usdc,
    COALESCE(w.max_single_wallet_usdc, 0.0) AS max_single_wallet_usdc,
    COALESCE(w.n_unique_wallets,       0)   AS flow_n_unique_wallets,
    COALESCE(w.n_bullish_wallets,      0)   AS n_bullish_wallets,
    COALESCE(w.n_bearish_wallets,      0)   AS n_bearish_wallets
FROM market_time_grid g
LEFT JOIN wallet_flow_2s_buckets w
    ON g.market_id = w.market_id
    AND g.grid_time = w.grid_time;
