"""
wallet_skill.py

Causal running track record of every (wallet, role), to be used as an edge
feature: at the time of a trade, how many of the wallet's PAST trades turned
out right or wrong at horizon H.

A past trade made at time s with direction d (oriented like the mid of its
market) has outcome r = d * (mid(s + H) - mid(s)), known at s + H.  It is
counted in the record read at time t only if s + H < t, so nothing that is
not observable at t enters the feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKILL_SHRINK = 10.0
N_SKILL_FEATURES = 3
N_BEHAVIOUR_FEATURES = 6
BEHAVIOUR_COLUMNS = ("beh_n", "beh_maker", "beh_dir", "beh_usdc", "beh_markets")


def build_running_skill(con, horizon_ns: int, seed: int = 0) -> None:
    """
    Needs the tables
        tr(ts BIGINT ns, wallet, role, r)   past trades; r NULL = outcome unknown, 0 = undecided
        q(wallet, role, ts BIGINT ns)       distinct (wallet, role, time) to look up
    and creates ``skill(wallet, role, ts, wins, losses, p_wins, p_losses)``: the decided trades
    of the (wallet, role) whose outcome matured strictly before ts.  p_* is the placebo: the
    record at the same time and role of another wallet, drawn by a fixed permutation of the
    wallets of ``q`` with a similar number of look-ups (similar records over the edges, identity destroyed).
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE skill_cum AS
        SELECT wallet, role, mt,
               sum(w) OVER (PARTITION BY wallet, role ORDER BY mt) AS wins,
               sum(l) OVER (PARTITION BY wallet, role ORDER BY mt) AS losses
        FROM (SELECT wallet, role, ts + {int(horizon_ns)} AS mt,
                     sum(CASE WHEN r > 0 THEN 1 ELSE 0 END) AS w, sum(CASE WHEN r < 0 THEN 1 ELSE 0 END) AS l
              FROM tr WHERE r IS NOT NULL AND r <> 0 GROUP BY 1, 2, 3)
    """)
    # permutation inside strata of activity (number of look-ups of the wallet, by powers of two): an
    # unrestricted permutation would hand to the few very active wallets, which make most of the
    # edges, the empty records of the many inactive ones
    wl = con.execute("SELECT wallet, floor(log2(count(*)))::INT AS stratum FROM q GROUP BY 1 ORDER BY 1").df()
    rng = np.random.default_rng(seed)
    wallets = wl["wallet"].values
    wallet_p = wallets.copy()
    for _, idx in wl.groupby("stratum").indices.items():
        wallet_p[idx] = wallets[rng.permutation(idx)]
    perm = pd.DataFrame({"wallet": wallets, "wallet_p": wallet_p})
    con.register("skill_perm_df", perm)
    con.execute("CREATE OR REPLACE TABLE skill_perm AS SELECT * FROM skill_perm_df")
    con.unregister("skill_perm_df")
    con.execute("""
        CREATE OR REPLACE TABLE skill AS
        WITH a AS (
            SELECT q.wallet, q.role, q.ts, p.wallet_p, coalesce(c.wins, 0) AS wins, coalesce(c.losses, 0) AS losses
            FROM (q JOIN skill_perm p ON q.wallet = p.wallet)
            ASOF LEFT JOIN skill_cum c ON q.wallet = c.wallet AND q.role = c.role AND q.ts > c.mt
        )
        SELECT a.wallet, a.role, a.ts, a.wins::BIGINT AS wins, a.losses::BIGINT AS losses,
               coalesce(c.wins, 0)::BIGINT AS p_wins, coalesce(c.losses, 0)::BIGINT AS p_losses
        FROM a ASOF LEFT JOIN skill_cum c ON a.wallet_p = c.wallet AND a.role = c.role AND a.ts > c.mt
    """)


def skill_edge_features(wins, losses, direction) -> np.ndarray:
    """
    (E, 3) float32:  shrunk skill (wins - losses) / (wins + losses + 10) in (-1, 1);
    skill * direction (the side the record of the wallet points to, oriented like the mid);
    log1p(decided trades) / 5 (how much the record can be trusted).
    """
    w = np.asarray(wins, dtype=np.float64)
    l = np.asarray(losses, dtype=np.float64)
    skill = (w - l) / (w + l + SKILL_SHRINK)
    return np.column_stack([skill, skill * np.asarray(direction, dtype=np.float64), np.log1p(w + l) / 5.0]).astype(np.float32)


def build_running_behaviour(con) -> None:
    """
    Needs the table ``tx(wallet, ts BIGINT ns, market_id, is_maker INT, d INT, usdc)`` with the trade
    sides of the history days, and creates ``behaviour(wallet, ts, beh_n, beh_maker, beh_dir,
    beh_usdc, beh_markets)``: for every (wallet, time) with a trade, what the wallet had done
    STRICTLY before that time: number of trade sides, how many as maker, sum of the directions,
    traded USDC, number of distinct markets.  No outcome of any trade is used, only behaviour.
    """
    con.execute("""
        CREATE OR REPLACE TABLE behaviour AS
        WITH first_seen AS (SELECT wallet, market_id, min(ts) AS ts FROM tx GROUP BY 1, 2),
        new_m AS (SELECT wallet, ts, count(*) AS new_markets FROM first_seen GROUP BY 1, 2),
        per_ts AS (SELECT wallet, ts, count(*) AS n, sum(is_maker) AS maker, sum(d) AS dir, sum(usdc) AS usdc
                   FROM tx GROUP BY 1, 2),
        j AS (SELECT p.*, coalesce(m.new_markets, 0) AS new_markets FROM per_ts p LEFT JOIN new_m m USING (wallet, ts))
        SELECT wallet, ts,
               coalesce(sum(n)           OVER w, 0)::BIGINT AS beh_n,
               coalesce(sum(maker)       OVER w, 0)::BIGINT AS beh_maker,
               coalesce(sum(dir)         OVER w, 0)::BIGINT AS beh_dir,
               coalesce(sum(usdc)        OVER w, 0)::DOUBLE AS beh_usdc,
               coalesce(sum(new_markets) OVER w, 0)::BIGINT AS beh_markets
        FROM j
        WINDOW w AS (PARTITION BY wallet ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)


def behaviour_edge_features(n, maker, direction_sum, usdc, markets) -> np.ndarray:
    """
    (E, 6) float32, all 0 for a wallet never seen before:
    log1p(trades)/10, maker share, one-sidedness |sum of directions| / trades,
    log1p(mean USDC per trade)/5, log1p(markets)/5, markets per trade.
    """
    n = np.asarray(n, dtype=np.float64)
    den = np.maximum(n, 1.0)
    return np.column_stack([
        np.log1p(n) / 10.0, np.asarray(maker, dtype=np.float64) / den, np.abs(np.asarray(direction_sum, dtype=np.float64)) / den,
        np.log1p(np.asarray(usdc, dtype=np.float64) / den) / 5.0, np.log1p(np.asarray(markets, dtype=np.float64)) / 5.0,
        np.asarray(markets, dtype=np.float64) / den,
    ]).astype(np.float32)
