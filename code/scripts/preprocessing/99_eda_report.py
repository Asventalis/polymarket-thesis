import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from polymarket_gnn.data.duckdb_client import get_duckdb_connection

def generate_eda_report():
    # Setup directories
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    figures_dir = project_root / "reports" / "data_audit" / "figures"
    reports_dir = project_root / "reports" / "data_audit"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    con = get_duckdb_connection()
    
    # 1. Wallet-level statistics
    print("Executing wallet-level aggregation query...")
    wallet_stats_query = """
    SELECT 
        wallet, 
        COUNT(tx_hash) as num_transactions, 
        SUM(total_usdc) as total_volume, 
        COUNT(DISTINCT market_id) as num_markets
    FROM transactions_with_market
    JOIN markets_universe USING (market_id)
    GROUP BY wallet
    """
    df_wallets = con.execute(wallet_stats_query).df()
    
    # 2. Save descriptive statistics
    stats_desc = df_wallets[['num_transactions', 'total_volume', 'num_markets']].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    
    # 3. Plotting
    sns.set_theme(style="whitegrid")
    
    # Volume distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_wallets, x='total_volume', bins=50, log_scale=True, color='blue', alpha=0.6)
    plt.title('Distribution of Total Volume per Wallet (Log Scale)')
    plt.xlabel('Total Volume (USDC)')
    plt.ylabel('Number of Wallets')
    vol_fig_path = figures_dir / 'volume_per_wallet.png'
    plt.savefig(vol_fig_path)
    plt.close()
    
    # Transactions distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_wallets, x='num_transactions', bins=50, log_scale=True, color='green', alpha=0.6)
    plt.title('Distribution of Transactions per Wallet (Log Scale)')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Wallets')
    tx_fig_path = figures_dir / 'transactions_per_wallet.png'
    plt.savefig(tx_fig_path)
    plt.close()
    
    # Markets distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_wallets, x='num_markets', bins=50, log_scale=True, color='orange', alpha=0.6)
    plt.title('Distribution of Number of Markets per Wallet (Log Scale)')
    plt.xlabel('Number of Markets')
    plt.ylabel('Number of Wallets')
    mkt_fig_path = figures_dir / 'markets_per_wallet.png'
    plt.savefig(mkt_fig_path)
    plt.close()
    
    # 4. Generate Markdown Report
    report_path = reports_dir / "eda_report.md"
    
    md_content = f"""# Exploratory Data Analysis: Wallet Behavior

This report provides a quick overview of the dataset features, specifically focusing on the distribution of trading behavior among wallets within our selected market universe (crypto up/down markets).

## Summary Statistics

The following table summarizes the trading behavior across `{len(df_wallets):,}` unique wallets:

| Metric | Total Volume (USDC) | Transactions | Markets Traded |
|---|---|---|---|
| **Mean** | {df_wallets['total_volume'].mean():.2f} | {df_wallets['num_transactions'].mean():.2f} | {df_wallets['num_markets'].mean():.2f} |
| **Median** | {df_wallets['total_volume'].median():.2f} | {df_wallets['num_transactions'].median():.2f} | {df_wallets['num_markets'].median():.2f} |
| **Max** | {df_wallets['total_volume'].max():.2f} | {df_wallets['num_transactions'].max():.2f} | {df_wallets['num_markets'].max():.2f} |
| **95th Pct** | {df_wallets['total_volume'].quantile(0.95):.2f} | {df_wallets['num_transactions'].quantile(0.95):.2f} | {df_wallets['num_markets'].quantile(0.95):.2f} |
| **99th Pct** | {df_wallets['total_volume'].quantile(0.99):.2f} | {df_wallets['num_transactions'].quantile(0.99):.2f} | {df_wallets['num_markets'].quantile(0.99):.2f} |

## Distributions

### Volume per Wallet
The vast majority of wallets trade small amounts, while a few "whale" wallets account for the majority of the volume.
![Volume per Wallet]({vol_fig_path.absolute()})

### Transactions per Wallet
Similarly, transaction frequency follows a heavy-tailed distribution. Many wallets transact only a few times, whereas algorithmic traders or market makers transact tens of thousands of times.
![Transactions per Wallet]({tx_fig_path.absolute()})

### Market Participation per Wallet
Most wallets specialize or only participate in a small number of markets. A small subset of users (likely automated) participate across a wide array of markets.
![Markets per Wallet]({mkt_fig_path.absolute()})
"""
    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    generate_eda_report()
