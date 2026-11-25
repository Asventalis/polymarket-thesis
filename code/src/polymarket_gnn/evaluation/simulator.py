"""Economic evaluation simulator for trading Polymarket contracts based on model predictions."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Trade:
    market_id: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    side: str  # "BUY_YES" or "SELL_YES"
    size: float
    pnl: float

class PolymarketSimulator:
    def __init__(
        self,
        threshold: float = 0.005,
        holding_period_seconds: int = 10,
        fee_rate: float = 0.000,
        slippage: float = 0.001,
        initial_capital: float = 10000.0,
        max_position_size: float = 100.0
    ):
        """
        Args:
            threshold: Minimum predicted return to trigger a trade.
            holding_period_seconds: How long to hold before exiting.
            fee_rate: Proportional fee per trade.
            slippage: Fixed slippage added to spread.
        """
        self.threshold = threshold
        self.holding_period_seconds = holding_period_seconds
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.times = []
        
    def run_simulation(
        self,
        df_predictions: pd.DataFrame,
    ):
        """
        df_predictions requires columns:
        - market_id
        - grid_time
        - mid_price
        - best_bid
        - best_ask
        - predicted_return (logit return)
        """
        # Sort chronologically
        df = df_predictions.sort_values('grid_time').copy()
        
        # To simulate holding period, we need to know the price exactly N seconds later
        # We can self-join or just group by market
        print("Pre-computing future prices for exits...")
        df['exit_time'] = df['grid_time'] + pd.Timedelta(seconds=self.holding_period_seconds)
        
        # Build an index to quickly lookup exit prices
        df_indexed = df.set_index(['market_id', 'grid_time'])
        
        def get_price(market_id, time, side):
            try:
                row = df_indexed.loc[(market_id, time)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row['best_ask'] if side == 'BUY' else row['best_bid']
            except KeyError:
                return None
                
        print("Simulating trades...")
        for _, row in df.iterrows():
            m_id = row['market_id']
            t_entry = row['grid_time']
            t_exit = row['exit_time']
            pred = row['predicted_return']
            
            # Simple strategy: If predicted return > threshold, buy YES.
            # If predicted return < -threshold, sell YES.
            
            if pred > self.threshold:
                entry_price = row['best_ask'] + self.slippage
                exit_price = get_price(m_id, t_exit, 'SELL')
                if exit_price is not None:
                    exit_price -= self.slippage
                    size = min(self.max_position_size, self.capital * 0.05)
                    shares = size / entry_price if entry_price > 0 else 0
                    
                    if shares > 0:
                        revenue = shares * exit_price
                        fees = (size + revenue) * self.fee_rate
                        pnl = revenue - size - fees
                        
                        self.capital += pnl
                        self.trades.append(Trade(m_id, t_entry, entry_price, t_exit, exit_price, "BUY_YES", size, pnl))
                        
            elif pred < -self.threshold:
                # Assuming we can short (sell YES)
                entry_price = row['best_bid'] - self.slippage
                exit_price = get_price(m_id, t_exit, 'BUY')
                if exit_price is not None:
                    exit_price += self.slippage
                    size = min(self.max_position_size, self.capital * 0.05)
                    # For selling YES, we collect entry_price. 
                    # Profit is entry_price - exit_price.
                    shares = size / (1 - entry_price) if entry_price < 1 else 0
                    if shares > 0:
                        cost_to_close = shares * exit_price
                        revenue = shares * entry_price
                        fees = (revenue + cost_to_close) * self.fee_rate
                        pnl = revenue - cost_to_close - fees
                        
                        self.capital += pnl
                        self.trades.append(Trade(m_id, t_entry, entry_price, t_exit, exit_price, "SELL_YES", size, pnl))
                        
            self.times.append(t_entry)
            self.equity_curve.append(self.capital)
            
        print("Simulation complete.")
        return self.compute_metrics()
        
    def compute_metrics(self) -> Dict:
        if not self.trades:
            return {"num_trades": 0, "total_pnl": 0.0}
            
        pnls = [t.pnl for t in self.trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        
        equity_series = pd.Series(self.equity_curve)
        drawdowns = (equity_series.cummax() - equity_series) / equity_series.cummax()
        max_drawdown = drawdowns.max()
        
        turnover = sum(t.size for t in self.trades)
        
        return {
            "num_trades": len(self.trades),
            "total_pnl": sum(pnls),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "final_capital": self.capital,
            "return_on_capital": (self.capital - self.initial_capital) / self.initial_capital
        }
