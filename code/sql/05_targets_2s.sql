-- Compute logit return targets
CREATE TABLE targets_2s AS
WITH grid_ordered AS (
    SELECT 
        market_id,
        grid_time,
        mid_price,
        LEAD(mid_price, 1) OVER (PARTITION BY market_id ORDER BY grid_time) as price_plus_2s,
        LEAD(mid_price, 5) OVER (PARTITION BY market_id ORDER BY grid_time) as price_plus_10s,
        LEAD(mid_price, 10) OVER (PARTITION BY market_id ORDER BY grid_time) as price_plus_20s
    FROM orderbook_features_2s
)
SELECT
    market_id,
    grid_time,
    -- Logit = log(p / (1-p)). Return = Logit(p_future) - Logit(p_current)
    LN(GREATEST(price_plus_2s, 0.0001) / (1.0 - LEAST(price_plus_2s, 0.9999))) - 
    LN(GREATEST(mid_price, 0.0001) / (1.0 - LEAST(mid_price, 0.9999))) as target_2s,
    
    LN(GREATEST(price_plus_10s, 0.0001) / (1.0 - LEAST(price_plus_10s, 0.9999))) - 
    LN(GREATEST(mid_price, 0.0001) / (1.0 - LEAST(mid_price, 0.9999))) as target_10s,
    
    LN(GREATEST(price_plus_20s, 0.0001) / (1.0 - LEAST(price_plus_20s, 0.9999))) - 
    LN(GREATEST(mid_price, 0.0001) / (1.0 - LEAST(mid_price, 0.9999))) as target_20s
FROM grid_ordered;
