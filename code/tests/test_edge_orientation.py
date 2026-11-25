import numpy as np
import pandas as pd
import torch

from polymarket_gnn.data.graph_builder import EDGE_DIM, EdgeBuilder, load_market_orientation


def block():
    return pd.DataFrame({
        "wallet": ["w1", "w2", "w1", "w3", "zz"],
        "market_id": ["yesno", "updown", "updown", "unknown_orient", "yesno"],
        "total_usdc": [10.0, 5.0, 1.0, 2.0, 3.0], "tokens": [20.0, 10.0, 4.0, 4.0, 6.0],
        "price": [0.5, 0.3, 0.7, 0.5, 0.5], "fee_rate": [0.0, 0.01, 0.0, 0.0, 0.0],
        "side": ["BUY", "SELL", "BUY", "BUY", "BUY"], "token_direction": [1, 1, -1, -1, 1],
        "role": ["TAKER", "MAKER", "TAKER", "TAKER", "TAKER"],
    })


M = {"yesno": 0, "updown": 1, "unknown_orient": 2}
W = {"w1": 0, "w2": 1, "w3": 2}


def test_default_keeps_the_raw_direction():
    ei, ef = EdgeBuilder(M, W)(block())
    assert ei.tolist() == [[0, 1, 0, 2], [0, 1, 1, 2]] and ef.shape == (4, EDGE_DIM)      # unknown wallet dropped
    assert ef[:, 5].tolist() == [1.0, 1.0, -1.0, -1.0]


def test_book_orientation_flips_only_token2_markets_and_only_feature_5():
    raw_ei, raw = EdgeBuilder(M, W)(block())
    ei, ef = EdgeBuilder(M, W, {"yesno": 1, "updown": -1})(block())
    assert torch.equal(ei, raw_ei)
    # yes/no market unchanged; up/down market flipped; market missing from the orientation map keeps the raw value
    assert ef[:, 5].tolist() == [1.0, -1.0, 1.0, -1.0]
    keep = [c for c in range(EDGE_DIM) if c != 5]
    assert torch.equal(ef[:, keep], raw[:, keep])


def test_load_market_orientation(tmp_path):
    p = tmp_path / "universe.parquet"
    pd.DataFrame({"market_id": ["a", "b", "c", "d", "e"],
                  "outcome_1": ["Yes", "Up", " yes ", "Lakers", None],
                  "outcome_2": ["No", "Down", "No", "Celtics", None]}).to_parquet(p)
    assert load_market_orientation(p) == {"a": 1, "b": -1, "c": 1, "d": -1}
