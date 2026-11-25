import pytest
import pandas as pd
from pathlib import Path

def get_fixture_path(filename: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "derived" / "dev_fixture" / filename

def test_orderbook_features():
    path = get_fixture_path("orderbook_features_2s.parquet")
    assert path.exists(), "Orderbook fixture not found"
    df = pd.read_parquet(path)
    
    # Basic checks
    assert not df.empty
    assert 'market_id' in df.columns
    assert 'grid_time' in df.columns
    assert 'mid_price' in df.columns
    assert 'spread' in df.columns
    
    # Check bounds
    assert df['grid_time'].min() >= pd.Timestamp('2026-03-10 14:00:00', tz='UTC').tz_localize(None)
    assert df['grid_time'].max() <= pd.Timestamp('2026-03-10 15:00:00', tz='UTC').tz_localize(None)
    
    # Check that mid_price is within [0, 1] range (Polymarket probabilities)
    valid_prices = df['mid_price'].dropna()
    assert (valid_prices >= 0.0).all()
    assert (valid_prices <= 1.0).all()

def test_transaction_features():
    path = get_fixture_path("transaction_features_2s.parquet")
    assert path.exists(), "Transaction fixture not found"
    df = pd.read_parquet(path)
    
    assert not df.empty
    assert 'volume_2s' in df.columns
    assert 'num_trades_2s' in df.columns
    assert 'buy_volume_2s' in df.columns
    
    # Test for zero-fill (no NaNs allowed in volume or trade counts)
    assert df['volume_2s'].isna().sum() == 0
    assert df['num_trades_2s'].isna().sum() == 0
    
    assert (df['volume_2s'] >= 0).all()
    assert (df['num_trades_2s'] >= 0).all()

def test_targets():
    path = get_fixture_path("targets_2s.parquet")
    assert path.exists(), "Targets fixture not found"
    df = pd.read_parquet(path)
    
    assert not df.empty
    assert 'target_2s' in df.columns
    assert 'target_10s' in df.columns
    assert 'target_20s' in df.columns

def test_market_edges():
    path = get_fixture_path("market_edges_2s.parquet")
    assert path.exists(), "Market edges fixture not found"
    df = pd.read_parquet(path)
    
    # Edge validation
    if not df.empty:
        assert 'market_id_1' in df.columns
        assert 'market_id_2' in df.columns
        assert 'co_trades' in df.columns
        assert (df['co_trades'] > 0).all()
