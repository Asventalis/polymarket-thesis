"""
smart_money_test.py

Model-free test of the question: does the IDENTITY of the wallets trading a
market carry information about its future mid price?  This is the only thing
the wallet–market graph adds to the order book, so the answer bounds what any
graph model can gain on this target.

    python smart_money_test.py --horizon 3600      # also 21600, 43200

A trade is a bet on a direction d (+1 = bet that the market's mid price rises,
-1 = that it falls) and its outcome at horizon H is r = d * (mid(t + H) - mid(t)),
with the same "last snapshot strictly before" prices used for the labels.

Orientation.  The order book of a market, hence its mid price and the labels,
refers to ONE of the two outcome tokens: token 1 for Yes/No markets, token 2
for every other market (Up/Down, Over/Under, team A/team B).  The column
``token_direction`` of the transactions is +1 for token 1 in every market, so
for the second group it has the opposite sign of the mid.  The direction used
here is  d = (BUY ? +1 : -1) * token_direction * orientation(market),  with
orientation = +1 iff outcome_1 is "Yes" (static market metadata).  The script
checks the rule on the data (trade prices must sit at the oriented mid) and
stops if it does not hold.

Part A — persistence of wallet skill (trade level)
    Skill of a (wallet, role) = shrunk signed hit rate of its trades of the
    training days whose outcome was already realised before the first
    validation day.  Wallets are bucketed by skill; if skill is a property of
    the wallet, the buckets must keep their ranking on the validation day.
    Reported per role, because takers and makers are on opposite sides of
    every trade and one of the two sides wins on average by construction.

Part B — smart-money baseline (market level, same samples as the other baselines)
    signal(m, t) = Σ skill(wallet) · d · size  over the trades of market m in (t - W, t]
    prediction   = UP / DOWN by the sign, CONSTANT when there is no signal.
    Controls with the same structure and no identity information:
      taker flow       skill ≡ 1, takers only (plain signed order flow)
      placebo          skills permuted across wallets of the same role
    and, for additivity, the comparison with Mean Reversion H on the samples
    where the two disagree.

Causality: skills use only outcomes realised before the validation period
starts, and are frozen there; the signal at t reads trades up to block t, the
same information the model receives.  Nothing is fitted on validation data.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from baseline_evaluation import DEFAULT_BASE_DIR, MONTH, YEAR, iter_blocks
from polymarket_gnn.data.lob_layout import MID_IDX
from polymarket_gnn.data.price_index import LifecycleTracker, MidPriceIndex
from polymarket_gnn.evaluation.baselines import CONST, DedupTracker, mean_reversion_from_change, parse_days
from polymarket_gnn.evaluation.metrics import N_CLASSES, cm_to_metrics, confusion_matrix, format_metrics_row
from polymarket_gnn.evaluation.smart_money import (
    cluster_bootstrap_gap, cumulative, direction_from_signal, shrunk_skill, skill_buckets, window_sums,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROLES = ("TAKER", "MAKER")
VARIANTS = ("Taker flow", "Smart money (usdc)", "Smart money (count)", "Smart money placebo (usdc)")


def day_start_ns(day: int) -> int:
    return (pd.Timestamp(year=YEAR, month=MONTH, day=1) + pd.Timedelta(days=day - 1)).value


def sql_list(paths) -> str:
    return "[" + ", ".join(f"'{p}'" for p in paths) + "]"


def build_trade_table(con, base_dir: Path, days, horizon_ns: int, max_lag_ns: int) -> int:
    """
    Tables ``mids`` (mid price snapshots per market) and ``tr`` (one row per trade side with
    mid0 = mid strictly before t, midH = mid strictly before t + H, r = d * (midH - mid0),
    d oriented like the mid).  r is NULL when the outcome is unknown: t + H beyond the price
    coverage, or no book snapshot in the ``max_lag_ns`` before the trade (markets that trade
    while their book is not being recorded would get a stale mid(t)).
    Returns the end of the price coverage (ns).
    """
    price_days = list(range(min(days), max(days) + 2))                    # one extra day for the outcomes of the last day
    tensor_globs = [base_dir / f"tensors/year={YEAR}/month={MONTH}/day={d}/hour=*/lob_features.parquet" for d in price_days
                    if (base_dir / f"tensors/year={YEAR}/month={MONTH}/day={d}").exists()]
    last_price_day = max(d for d in price_days if (base_dir / f"tensors/year={YEAR}/month={MONTH}/day={d}").exists())
    coverage_end = day_start_ns(last_price_day + 1)
    t0 = time.time()
    con.execute(f"""
        CREATE TABLE mids AS
        SELECT market_id, epoch_ns(timestamp) AS ts, features[{MID_IDX + 1}] AS mid
        FROM read_parquet({sql_list(tensor_globs)})
    """)
    con.execute(f"""
        CREATE TABLE orient AS
        SELECT market_id, CASE WHEN lower(trim(any_value(outcome_1))) = 'yes' THEN 1 ELSE -1 END AS o
        FROM read_parquet('{base_dir.parent / "universe_markets.parquet"}') WHERE outcome_1 IS NOT NULL GROUP BY 1
    """)
    n_mid = con.execute("SELECT count(*) FROM mids").fetchone()[0]
    logger.info(f"mids: {n_mid} snapshots over days {price_days[0]}–{last_price_day} in {time.time() - t0:.0f}s")

    t0 = time.time()
    txn_globs = [base_dir / f"partitioned/transactions/year={YEAR}/month={MONTH}/day={d}/hour=*/**/*.parquet" for d in days]
    con.execute(f"""
        CREATE TABLE tr AS
        WITH t AS (
            SELECT epoch_ns(x.timestamp) AS ts, x.wallet, x.role, x.market_id, q.o AS orientation,
                   CASE WHEN x.token_direction * q.o = 1 THEN x.price ELSE 1 - x.price END AS price,   -- as price of the book token
                   (CASE WHEN x.side = 'BUY' THEN 1 ELSE -1 END) * x.token_direction * q.o AS d, x.total_usdc AS usdc
            FROM read_parquet({sql_list(txn_globs)}) x JOIN orient q ON x.market_id = q.market_id
            WHERE x.token_direction IN (1, -1) AND x.role IN ('TAKER', 'MAKER')
        ), a AS (
            SELECT t.*, m.mid AS mid0, t.ts - m.ts AS lag0 FROM t ASOF LEFT JOIN mids m ON t.market_id = m.market_id AND t.ts > m.ts
        ), b AS (
            SELECT a.*, m.mid AS midH_raw FROM a ASOF LEFT JOIN mids m ON a.market_id = m.market_id AND a.ts + {horizon_ns} > m.ts
        )
        SELECT ts, wallet, role, market_id, orientation, price, d, usdc, mid0, lag0,
               CASE WHEN ts + {horizon_ns} <= {coverage_end} AND lag0 <= {max_lag_ns} THEN d * (midH_raw - mid0) END AS r
        FROM b
    """)
    n, n_r = con.execute("SELECT count(*), count(r) FROM tr").fetchone()
    logger.info(f"tr: {n} trade sides, {n_r} with an outcome, in {time.time() - t0:.0f}s")
    return coverage_end


def convention_check(con) -> dict:
    """
    Checks the orientation rule on the data: with a fresh book, trade prices expressed as price of
    the book token must sit at the mid (and far from 1 - mid), in both orientation groups, and the
    markets where the opposite orientation fits better must be a negligible share.
    """
    df = con.execute("""
        SELECT orientation, count(*) AS n, count(DISTINCT market_id) AS markets,
               median(abs(price - mid0)) AS dist_to_mid, median(abs(price - (1 - mid0))) AS dist_to_one_minus_mid
        FROM tr WHERE role = 'TAKER' AND r IS NOT NULL GROUP BY 1 ORDER BY 1
    """).df()
    out = {("token1_yes_no" if r.orientation == 1 else "token2_other"):
           {"trades": int(r.n), "markets": int(r.markets), "median_dist_to_mid": float(r.dist_to_mid),
            "median_dist_to_one_minus_mid": float(r.dist_to_one_minus_mid)} for r in df.itertuples()}
    wrong = con.execute("""
        SELECT count(*) AS markets, sum(CASE WHEN d_mid > d_flip THEN 1 ELSE 0 END) AS wrong,
               sum(n) AS trades, sum(CASE WHEN d_mid > d_flip THEN n ELSE 0 END) AS wrong_trades
        FROM (SELECT market_id, count(*) AS n, avg(abs(price - mid0)) AS d_mid, avg(abs(price - (1 - mid0))) AS d_flip
              FROM tr WHERE role = 'TAKER' AND r IS NOT NULL GROUP BY 1 HAVING count(*) >= 20 AND abs(avg(mid0) - 0.5) > 0.1)
    """).fetchone()
    out["markets_checked"] = int(wrong[0]); out["markets_better_fit_by_opposite_orientation"] = int(wrong[1] or 0)
    out["share_of_trades_in_those_markets"] = float((wrong[3] or 0) / max(wrong[2] or 1, 1))
    ok = all(v["median_dist_to_mid"] < 0.25 * v["median_dist_to_one_minus_mid"] for k, v in out.items() if isinstance(v, dict)) \
        and out["share_of_trades_in_those_markets"] < 0.02
    logger.info(f"Orientation check: {out} → {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("the orientation rule does not hold on the data: the direction d would be wrong")
    return out


def main():
    p = argparse.ArgumentParser(description="Smart-money test: is wallet identity informative about the future mid?")
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--train-days", type=str, default="6-12", help="days whose realised trade outcomes define the wallet skill")
    p.add_argument("--val-days", type=str, default="13")
    p.add_argument("--min-trades", type=int, default=20, help="decided training trades needed for a wallet to enter the skill buckets")
    p.add_argument("--shrink", type=float, default=10.0)
    p.add_argument("--buckets", type=int, default=10)
    p.add_argument("--windows", type=str, default="600,3600", help="trade windows W in seconds for the market-level signal")
    p.add_argument("--max-lag", type=float, default=600.0,
                   help="a trade has an outcome only if its market has a book snapshot in the last N seconds")
    p.add_argument("--const-eps", type=float, default=0.0)
    p.add_argument("--dedup-minutes", type=float, default=10.0)
    p.add_argument("--close-filter", choices=["last_tx", "none"], default="last_tx")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--memory-limit", type=str, default="120GB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-part-b", action="store_true")
    p.add_argument("--max-hours", type=int, default=None, help="debug: stop part B after N validation hours")
    args = p.parse_args()

    H = args.horizon * 10**9
    horizon_hours = int(np.ceil(args.horizon / 3600))
    label = f"{args.horizon // 3600}h" if args.horizon >= 3600 else f"{args.horizon}s"
    train_days, val_days = parse_days(args.train_days), parse_days(args.val_days)
    if max(train_days) >= min(val_days):
        raise SystemExit("--train-days must strictly precede --val-days")
    windows_s = sorted({int(x) for x in args.windows.split(",") if x.strip()})
    windows_ns = [w * 10**9 for w in windows_s]
    val_start, val_end = day_start_ns(min(val_days)), day_start_ns(max(val_days) + 1)
    train_start = day_start_ns(min(train_days))
    out_path = args.out or Path(f"smart_money_{label}.json")
    rng = np.random.default_rng(args.seed)

    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    coverage_end = build_trade_table(con, args.base_dir, sorted(set(train_days) | set(val_days)), H, int(args.max_lag * 10**9))
    convention = convention_check(con)

    # ── Wallet skill from outcomes realised before the validation period ─────
    skill = con.execute(f"""
        SELECT wallet, role, count(*) AS n, sum(CASE WHEN r > 0 THEN 1 ELSE 0 END) AS wins,
               sum(CASE WHEN r < 0 THEN 1 ELSE 0 END) AS losses, avg(r) AS mean_r, sum(usdc) AS usdc
        FROM tr WHERE ts >= {train_start} AND ts + {H} <= {val_start} AND r IS NOT NULL
        GROUP BY 1, 2
    """).df()
    assert con.execute(f"SELECT coalesce(max(ts + {H}), 0) FROM tr WHERE ts >= {train_start} AND ts + {H} <= {val_start} AND r IS NOT NULL").fetchone()[0] <= val_start
    skill["skill"] = shrunk_skill(skill["wins"].values, skill["losses"].values, args.shrink)
    skill["bucket"] = -1
    skill["skill_placebo"] = 0.0
    for role in ROLES:
        sel = (skill["role"] == role).values
        decided = (skill["wins"] + skill["losses"]).values
        skill.loc[sel, "bucket"] = skill_buckets(skill["skill"].values[sel], decided[sel] >= args.min_trades, args.buckets)
        skill.loc[sel, "skill_placebo"] = rng.permutation(skill["skill"].values[sel])
    logger.info(f"Skill: {len(skill)} (wallet, role) pairs, {(skill['bucket'] >= 0).sum()} with ≥ {args.min_trades} decided trades")
    con.register("skill_df", skill[["wallet", "role", "skill", "skill_placebo", "bucket"]])
    con.execute("CREATE TABLE sk AS SELECT * FROM skill_df")

    # ── Part A: do the skill buckets keep their ranking on validation? ───────
    val_where = f"v.ts >= {val_start} AND v.ts < {val_end} AND v.r IS NOT NULL"
    by_mkt = con.execute(f"""
        SELECT v.market_id, v.role, coalesce(s.bucket, -1) AS bucket, count(*) AS n,
               sum(CASE WHEN v.r > 0 THEN 1 ELSE 0 END) AS wins, sum(CASE WHEN v.r < 0 THEN 1 ELSE 0 END) AS losses,
               sum(v.r) AS sum_r, sum(v.usdc * v.r) AS sum_usdc_r, sum(v.usdc) AS usdc, count(DISTINCT v.wallet) AS wallets_in_market
        FROM tr v LEFT JOIN sk s ON v.wallet = s.wallet AND v.role = s.role
        WHERE {val_where} GROUP BY 1, 2, 3
    """).df()
    n_wallets = con.execute(f"""
        SELECT v.role, coalesce(s.bucket, -1) AS bucket, count(DISTINCT v.wallet) AS wallets
        FROM tr v LEFT JOIN sk s ON v.wallet = s.wallet AND v.role = s.role WHERE {val_where} GROUP BY 1, 2
    """).df()
    val_wallet = con.execute(f"""
        SELECT v.wallet, v.role, sum(CASE WHEN v.r > 0 THEN 1 ELSE 0 END) AS v_wins, sum(CASE WHEN v.r < 0 THEN 1 ELSE 0 END) AS v_losses
        FROM tr v WHERE {val_where} GROUP BY 1, 2
    """).df()

    part_a = {}
    for role in ROLES:
        g = by_mkt[by_mkt["role"] == role]
        rows = []
        train_stats = skill[skill["role"] == role].groupby("bucket").agg(
            train_wallets=("wallet", "size"), train_skill=("skill", "mean"), tw=("wins", "sum"), tl=("losses", "sum"))
        for b, gb in g.groupby("bucket"):
            decided = gb["wins"].sum() + gb["losses"].sum()
            ts_ = train_stats.loc[b] if b in train_stats.index else None
            nw = n_wallets[(n_wallets["role"] == role) & (n_wallets["bucket"] == b)]["wallets"]
            rows.append({
                "bucket": int(b),
                "train_wallets": int(ts_["train_wallets"]) if ts_ is not None and b >= 0 else None,
                "train_hit_rate": float(ts_["tw"] / (ts_["tw"] + ts_["tl"])) if ts_ is not None and b >= 0 and (ts_["tw"] + ts_["tl"]) > 0 else None,
                "val_wallets": int(nw.iloc[0]) if len(nw) else 0,
                "val_trades": int(gb["n"].sum()), "val_decided": int(decided),
                "val_hit_rate": float(gb["wins"].sum() / decided) if decided else None,
                "val_mean_r_cents": float(100 * gb["sum_r"].sum() / gb["n"].sum()),
                "val_usdc_weighted_r_cents": float(100 * gb["sum_usdc_r"].sum() / gb["usdc"].sum()) if gb["usdc"].sum() > 0 else None,
                "val_markets": int(gb["market_id"].nunique()),
            })
        top, bot = args.buckets - 1, 0
        piv_w = g[g["bucket"].isin([top, bot])].pivot_table(index="market_id", columns="bucket", values="wins", aggfunc="sum", fill_value=0)
        piv_l = g[g["bucket"].isin([top, bot])].pivot_table(index="market_id", columns="bucket", values="losses", aggfunc="sum", fill_value=0)
        for piv in (piv_w, piv_l):
            for b in (top, bot):
                if b not in piv.columns:
                    piv[b] = 0
        gap = cluster_bootstrap_gap(piv_w[top].values, piv_l[top].values, piv_w[bot].values, piv_l[bot].values, n_boot=1000, seed=args.seed)

        # wallet-level persistence: rank correlation of training skill and validation hit rate
        m = skill[(skill["role"] == role) & (skill["bucket"] >= 0)].merge(val_wallet[val_wallet["role"] == role], on=["wallet", "role"])
        m = m[(m["v_wins"] + m["v_losses"]) >= args.min_trades]
        m["v_hit"] = m["v_wins"] / (m["v_wins"] + m["v_losses"])
        m["t_hit"] = m["wins"] / (m["wins"] + m["losses"])
        rho = float(m["t_hit"].rank().corr(m["v_hit"].rank())) if len(m) > 2 else None
        # equal weight per wallet, so that a few very active wallets do not decide the result
        by_bucket_wallet = m.groupby("bucket")["v_hit"].agg(["mean", "size"])
        part_a[role] = {
            "buckets": sorted(rows, key=lambda r: r["bucket"]),
            "top_minus_bottom_val_hit_rate": {"gap": gap[0], "ci95_market_bootstrap": [gap[1], gap[2]]},
            "wallet_level": {"n_wallets_both_periods": int(len(m)), "spearman_train_vs_val_hit_rate": rho,
                             "mean_val_hit_rate_per_wallet_by_bucket": {int(b): [float(r["mean"]), int(r["size"])] for b, r in by_bucket_wallet.iterrows()}},
        }

    print(f"\n{'=' * 110}\nSMART MONEY TEST — horizon {label} — skill from days {train_days} (outcomes realised before day {min(val_days)}), "
          f"tested on day(s) {val_days}\n{'=' * 110}")
    print("\nPART A — do wallets keep their skill?  (bucket 0 = worst training skill, "
          f"{args.buckets - 1} = best; -1 = fewer than {args.min_trades} decided training trades or new wallet)")
    for role in ROLES:
        print(f"\n[{role}]  bucket | train wallets | train hit | val wallets | val trades | val hit | val mean r (cents) | usdc-w. r (cents)")
        for r in part_a[role]["buckets"]:
            f = lambda x, fmt: ("   n/a" if x is None else format(x, fmt))
            print(f"          {r['bucket']:>4}  | {f(r['train_wallets'], '>12')}  | {f(r['train_hit_rate'], '>8.4f')}  | {r['val_wallets']:>10}  | "
                  f"{r['val_trades']:>9}  | {f(r['val_hit_rate'], '>6.4f')}  | {r['val_mean_r_cents']:>17.4f}  | {f(r['val_usdc_weighted_r_cents'], '>10.4f')}")
        g_ = part_a[role]["top_minus_bottom_val_hit_rate"]
        w_ = part_a[role]["wallet_level"]
        print(f"   top − bottom bucket, validation hit rate: {g_['gap']:+.4f}  (95% CI, markets resampled: {g_['ci95_market_bootstrap'][0]:+.4f} … {g_['ci95_market_bootstrap'][1]:+.4f})")
        print(f"   wallets with ≥ {args.min_trades} decided trades in both periods: {w_['n_wallets_both_periods']}, "
              f"Spearman(train hit rate, val hit rate) = {w_['spearman_train_vs_val_hit_rate']}")
        print("   mean validation hit rate per wallet, by bucket: " +
              "  ".join(f"{b}:{v[0]:.3f}" for b, v in sorted(w_["mean_val_hit_rate_per_wallet_by_bucket"].items())))

    results = {
        "meta": {"horizon_s": args.horizon, "label": label, "train_days": train_days, "val_days": val_days,
                 "min_trades": args.min_trades, "max_lag_s": args.max_lag, "shrink": args.shrink, "buckets": args.buckets, "windows_s": windows_s,
                 "const_eps": args.const_eps, "close_filter": args.close_filter, "seed": args.seed,
                 "price_coverage_end": str(pd.Timestamp(coverage_end)), "orientation_check": convention,
                 "n_skill_pairs": int(len(skill)), "generated_at": datetime.now().isoformat(timespec="seconds")},
        "part_a": part_a,
    }

    # ── Part B: market-level baseline on the samples of the other baselines ──
    if not args.skip_part_b:
        t0 = time.time()
        df = con.execute(f"""
            SELECT v.ts, v.market_id, v.role, v.d, v.usdc, coalesce(s.skill, 0) AS skill, coalesce(s.skill_placebo, 0) AS skill_placebo
            FROM tr v LEFT JOIN sk s ON v.wallet = s.wallet AND v.role = s.role
            WHERE v.ts > {val_start - max(windows_ns)} AND v.ts < {val_end}
            ORDER BY v.market_id, v.ts
        """).df()
        taker = (df["role"].values == "TAKER").astype(np.float64)
        d, usdc = df["d"].values.astype(np.float64), df["usdc"].values.astype(np.float64)
        contrib = np.column_stack([taker * d * usdc, df["skill"].values * d * usdc, df["skill"].values * d,
                                   df["skill_placebo"].values * d * usdc])
        ts_all, mk = df["ts"].values.astype(np.int64), df["market_id"].values
        bounds = np.flatnonzero(mk[1:] != mk[:-1]) + 1
        starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(mk)]])
        trades = {mk[s]: cumulative(ts_all[s:e], contrib[s:e]) for s, e in zip(starts, ends)} if len(mk) else {}
        logger.info(f"Part B: {len(df)} trade sides in {len(trades)} markets indexed in {time.time() - t0:.0f}s")
        del df, contrib

        lifecycle = LifecycleTracker(args.base_dir / "market_lifecycle_events.parquet")
        index = MidPriceIndex(args.base_dir, hours_back=horizon_hours, hours_ahead=horizon_hours, lifecycle=lifecycle,
                              close_filter=args.close_filter, const_eps=args.const_eps)
        known = set(pd.read_parquet(args.base_dir / "id_mappings/market_to_idx.parquet")["market_id"])
        names = ["Mean Reversion H (causal)"] + [f"{v} W={w}s" for w in windows_s for v in VARIANTS] + \
                [f"Faded taker flow W={w}s" for w in windows_s] + \
                [f"Mean Reversion H, else smart money W={w}s" for w in windows_s]
        cms = {s: {n: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for n in names} for s in ("all", "dedup")}
        # head-to-head with Mean Reversion on the samples where both take a side and the price moves
        h2h = {w: {v: np.zeros(4, dtype=np.int64) for v in VARIANTS} for w in windows_s}      # agree&right, agree&wrong, disagree&signal right, disagree&MR right
        coverage = {w: {v: 0 for v in VARIANTS} for w in windows_s}
        dedup = DedupTracker(int(args.dedup_minutes * 60 * 10**9))
        n_samples = n_hours = 0
        last_hour = None
        for prev, t_ns, active in iter_blocks(args.base_dir, val_days, index, lifecycle, known):
            if index._current_key != last_hour:
                last_hour = index._current_key
                n_hours += 1
                if args.max_hours and n_hours > args.max_hours:
                    break
                logger.info(f"Part B hour {pd.Timestamp(last_hour * 3_600_000_000_000)} | samples so far {n_samples}")
            if not active:
                continue
            y_all = index.labels(active, t_ns, H)
            valid = ~np.isnan(y_all)
            if not valid.any():
                continue
            act = [m for m, v in zip(active, valid) if v]
            y = y_all[valid].astype(np.int64)
            n = len(y)
            n_samples += n
            sig = np.zeros((len(windows_s), n, len(VARIANTS)))
            for j, m in enumerate(act):
                a = trades.get(m)
                if a is not None:
                    sig[:, j, :] = window_sums(a[0], a[1], t_ns, windows_ns)
            mr = mean_reversion_from_change(index.past_change(act, t_ns, H), args.const_eps)
            preds = {"Mean Reversion H (causal)": mr}
            for k, w in enumerate(windows_s):
                for i, v in enumerate(VARIANTS):
                    pr = direction_from_signal(sig[k, :, i])
                    preds[f"{v} W={w}s"] = pr
                    coverage[w][v] += int((pr != CONST).sum())
                    both = (pr != CONST) & (mr != CONST) & (y != CONST)
                    agree = both & (pr == mr)
                    dis = both & (pr != mr)
                    h2h[w][v] += np.array([(agree & (pr == y)).sum(), (agree & (pr != y)).sum(),
                                           (dis & (pr == y)).sum(), (dis & (mr == y)).sum()])
                preds[f"Faded taker flow W={w}s"] = 2 - preds[f"Taker flow W={w}s"]
                sm = preds[f"Smart money (usdc) W={w}s"]
                preds[f"Mean Reversion H, else smart money W={w}s"] = np.where(mr != CONST, mr, sm)
            keep = dedup.keep_mask(act, t_ns)
            for name, pr in preds.items():
                cms["all"][name] += confusion_matrix(y, pr)
                if keep.any():
                    cms["dedup"][name] += confusion_matrix(y[keep], pr[keep])
        logger.info(f"Part B done in {time.time() - t0:.0f}s: {n_samples} samples")

        results["part_b"] = {
            "n_samples": n_samples,
            "baselines": {n: {s: cm_to_metrics(cms[s][n]) for s in ("all", "dedup")} for n in names},
            "coverage": {str(w): {v: coverage[w][v] / max(n_samples, 1) for v in VARIANTS} for w in windows_s},
            "vs_mean_reversion": {str(w): {v: {"agree_right": int(h[0]), "agree_wrong": int(h[1]),
                                               "disagree_signal_right": int(h[2]), "disagree_mean_reversion_right": int(h[3])}
                                           for v, h in h2h[w].items()} for w in windows_s},
        }
        print(f"\nPART B — market-level baselines on the {n_samples} labelled validation samples")
        for s in ("all", "dedup"):
            print(f"\n[{s}]")
            for n_ in names:
                print(format_metrics_row(n_, results["part_b"]["baselines"][n_][s]))
        print("\nShare of samples with a signal, and head-to-head with Mean Reversion H where both take a side and the price moves")
        for w in windows_s:
            for v in VARIANTS:
                h = h2h[w][v]
                dis = h[2] + h[3]
                print(f"  {v + f' W={w}s':<38} coverage={coverage[w][v] / max(n_samples, 1):.3f}  agree: right {h[0]} wrong {h[1]}  "
                      f"disagree: signal right {h[2]} / Mean Reversion right {h[3]}"
                      + (f"  → signal wins {h[2] / dis:.3f}" if dis else ""))

    print()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
