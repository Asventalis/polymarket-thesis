import duckdb
import numpy as np
import pandas as pd
import pytest

from polymarket_gnn.data.graph_builder import EDGE_DIM, EdgeBuilder
from polymarket_gnn.data.wallet_skill import N_SKILL_FEATURES, build_running_skill, skill_edge_features


def run_skill(tr: pd.DataFrame, q: pd.DataFrame, horizon: int, seed: int = 0) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("tr_df", tr); con.register("q_df", q)
    con.execute("CREATE TABLE tr AS SELECT * FROM tr_df"); con.execute("CREATE TABLE q AS SELECT * FROM q_df")
    build_running_skill(con, horizon, seed)
    return con.execute("SELECT * FROM skill ORDER BY wallet, role, ts").df()


def test_running_skill_counts_only_outcomes_matured_strictly_before():
    H = 100
    tr = pd.DataFrame({
        "ts":     [0,     10,    20,    30,    0,     5],
        "wallet": ["a",   "a",   "a",   "a",   "a",   "b"],
        "role":   ["TAKER"] * 4 + ["MAKER", "TAKER"],
        "r":      [0.1,   -0.2,  0.0,   None,  0.3,   -0.1],     # win, loss, undecided, unknown | maker win | b loss
    })
    q = pd.DataFrame({"wallet": ["a"] * 5 + ["b", "c"], "role": ["TAKER"] * 4 + ["MAKER", "TAKER", "TAKER"],
                      "ts": [50, 100, 101, 500, 500, 106, 500]})
    s = run_skill(tr, q, H).set_index(["wallet", "role", "ts"])
    assert tuple(s.loc[("a", "TAKER", 50), ["wins", "losses"]]) == (0, 0)       # nothing matured yet
    assert tuple(s.loc[("a", "TAKER", 100), ["wins", "losses"]]) == (0, 0)      # matures AT 100: not strictly before
    assert tuple(s.loc[("a", "TAKER", 101), ["wins", "losses"]]) == (1, 0)
    assert tuple(s.loc[("a", "TAKER", 500), ["wins", "losses"]]) == (1, 1)      # r = 0 and r unknown never count
    assert tuple(s.loc[("a", "MAKER", 500), ["wins", "losses"]]) == (1, 0)      # roles are separate records
    assert tuple(s.loc[("b", "TAKER", 106), ["wins", "losses"]]) == (0, 1)
    assert tuple(s.loc[("c", "TAKER", 500), ["wins", "losses"]]) == (0, 0)      # wallet without history
    assert len(s) == len(q)


def test_running_skill_matches_brute_force_and_placebo_is_a_permutation():
    rng = np.random.default_rng(0)
    n, H = 3000, 500
    tr = pd.DataFrame({"ts": rng.integers(0, 5000, n), "wallet": rng.choice(list("abcdefgh"), n),
                       "role": rng.choice(["TAKER", "MAKER"], n),
                       "r": rng.choice([-1.0, 0.0, 1.0, np.nan], n)})
    q = tr[["wallet", "role", "ts"]].drop_duplicates().reset_index(drop=True)
    s = run_skill(tr, q, H, seed=3)

    def brute(w, role, t):
        m = (tr.wallet == w) & (tr.role == role) & (tr.ts + H < t)
        return int((tr.r[m] > 0).sum()), int((tr.r[m] < 0).sum())

    for row in s.sample(300, random_state=0).itertuples():
        assert (row.wins, row.losses) == brute(row.wallet, row.role, row.ts)
    # placebo = the record of ONE other wallet, the same for all the rows of a wallet, at the same time and role
    wallets = sorted(q.wallet.unique())
    mapping = {}
    for w in wallets:
        rows = s[s.wallet == w]
        cands = [p for p in wallets if all((r.p_wins, r.p_losses) == brute(p, r.role, r.ts) for r in rows.itertuples())]
        assert cands, w
        mapping[w] = cands
    assert any(mapping[w] != [w] and w not in mapping[w] for w in wallets)     # identity is actually shuffled


def test_skill_edge_features():
    f = skill_edge_features([0, 80, 20], [0, 20, 80], [1, -1, -1])
    assert f.shape == (3, N_SKILL_FEATURES) and f.dtype == np.float32
    assert f[0].tolist() == [0.0, 0.0, 0.0]
    assert f[1, 0] == pytest.approx(60 / 110) and f[1, 1] == pytest.approx(-60 / 110)
    assert f[2, 0] == pytest.approx(-60 / 110) and f[2, 1] == pytest.approx(60 / 110)    # a loser selling → points up
    assert f[1, 2] == pytest.approx(np.log1p(100) / 5)


def test_edge_builder_appends_skill_and_leaves_the_base_features_unchanged():
    df = pd.DataFrame({
        "wallet": ["w1", "w2", "zz"], "market_id": ["yesno", "updown", "yesno"],
        "total_usdc": [10.0, 5.0, 3.0], "tokens": [20.0, 10.0, 6.0], "price": [0.5, 0.3, 0.5], "fee_rate": [0.0, 0.01, 0.0],
        "side": ["BUY", "BUY", "BUY"], "token_direction": [1, 1, 1], "role": ["TAKER", "MAKER", "TAKER"],
        "skill_wins": [30, 30, 5], "skill_losses": [10, 10, 5],
    })
    M, W, O = {"yesno": 0, "updown": 1}, {"w1": 0, "w2": 1}, {"yesno": 1, "updown": -1}
    _, base = EdgeBuilder(M, W, O)(df)
    b = EdgeBuilder(M, W, O, skill=True)
    ei, ef = b(df)
    assert b.edge_dim == EDGE_DIM + N_SKILL_FEATURES and ef.shape == (2, b.edge_dim)
    assert np.allclose(ef[:, :EDGE_DIM].numpy(), base.numpy())
    assert ef[:, 7].tolist() == pytest.approx([0.4, 0.4])
    # buying token 1: bet on mid up in the Yes/No market, on mid down in the market whose book is on token 2
    assert ef[:, 8].tolist() == pytest.approx([0.4, -0.4])
    assert b(df.iloc[2:])[1].shape == (0, b.edge_dim)
    with pytest.raises(ValueError):
        EdgeBuilder(M, W, None, skill=True)


def test_running_behaviour_uses_only_strictly_earlier_trades():
    from polymarket_gnn.data.wallet_skill import BEHAVIOUR_COLUMNS, build_running_behaviour
    rng = np.random.default_rng(0)
    n = 2000
    tx = pd.DataFrame({"wallet": rng.choice(list("abcdef"), n), "ts": rng.integers(0, 300, n),      # many ties in time
                       "market_id": rng.choice([f"m{i}" for i in range(12)], n), "is_maker": rng.integers(0, 2, n),
                       "d": rng.choice([-1, 1], n), "usdc": rng.random(n) * 100})
    con = duckdb.connect()
    con.register("tx_df", tx); con.execute("CREATE TABLE tx AS SELECT * FROM tx_df")
    build_running_behaviour(con)
    b = con.execute("SELECT * FROM behaviour").df()
    assert len(b) == len(tx[["wallet", "ts"]].drop_duplicates())
    for row in b.sample(200, random_state=1).itertuples():
        past = tx[(tx.wallet == row.wallet) & (tx.ts < row.ts)]
        assert row.beh_n == len(past) and row.beh_maker == past.is_maker.sum() and row.beh_dir == past.d.sum()
        assert row.beh_usdc == pytest.approx(past.usdc.sum()) and row.beh_markets == past.market_id.nunique()
    first = b.loc[b.groupby("wallet").ts.idxmin()]
    assert (first[list(BEHAVIOUR_COLUMNS)].values == 0).all()                # nothing known at a wallet's first trade


def test_behaviour_edge_features_and_builder():
    from polymarket_gnn.data.wallet_skill import BEHAVIOUR_COLUMNS, N_BEHAVIOUR_FEATURES, behaviour_edge_features
    f = behaviour_edge_features([0, 100], [0, 25], [0, -60], [0.0, 5000.0], [0, 10])
    assert f.shape == (2, N_BEHAVIOUR_FEATURES) and (f[0] == 0).all()
    assert f[1].tolist() == pytest.approx([np.log1p(100) / 10, 0.25, 0.6, np.log1p(50) / 5, np.log1p(10) / 5, 0.1], rel=1e-6)
    df = pd.DataFrame({
        "wallet": ["w1", "w2"], "market_id": ["yesno", "updown"], "total_usdc": [10.0, 5.0], "tokens": [20.0, 10.0],
        "price": [0.5, 0.3], "fee_rate": [0.0, 0.01], "side": ["BUY", "BUY"], "token_direction": [1, 1], "role": ["TAKER", "MAKER"],
        "skill_wins": [30, 30], "skill_losses": [10, 10],
        "beh_n": [0, 100], "beh_maker": [0, 25], "beh_dir": [0, -60], "beh_usdc": [0.0, 5000.0], "beh_markets": [0, 10],
    })
    assert set(BEHAVIOUR_COLUMNS) <= set(df.columns)
    M, W, O = {"yesno": 0, "updown": 1}, {"w1": 0, "w2": 1}, {"yesno": 1, "updown": -1}
    _, base = EdgeBuilder(M, W, O)(df)
    b = EdgeBuilder(M, W, O, behaviour=True)
    _, ef = b(df)
    assert b.edge_dim == EDGE_DIM + N_BEHAVIOUR_FEATURES and np.allclose(ef[:, :EDGE_DIM].numpy(), base.numpy())
    assert np.allclose(ef[:, EDGE_DIM:].numpy(), f)
    both = EdgeBuilder(M, W, O, skill=True, behaviour=True)
    assert both(df)[1].shape == (2, EDGE_DIM + N_SKILL_FEATURES + N_BEHAVIOUR_FEATURES)
    assert np.allclose(both(df)[1][:, -N_BEHAVIOUR_FEATURES:].numpy(), f)
