-- 01_time_coverage.sql: Inspect daily event counts and timestamp range across views

SELECT 
    CAST(timestamp AS DATE) AS day,
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT wallet) AS unique_wallets,
    COUNT(DISTINCT market_id) AS active_markets
FROM universe.transactions
GROUP BY day
ORDER BY day;
