"""
build_wallet_skill.py

Pre-computes the causal running track record of the wallets (see
polymarket_gnn/data/wallet_skill.py) for the transactions of the target days,
one parquet per hour:

    <out-dir>/h<label>/day=<d>/hour=<h>/*.parquet
        wallet, role, timestamp, wins, losses, p_wins, p_losses

read by ``train_full.py --edge-skill real|placebo``.  The outcomes of the past
trades are the ones of the smart-money test (same orientation rule, same
freshness filter on the book), taken from the trades of ``--history-days``.

    ./run_dev.sh build_wallet_skill.py --horizon 3600 --history-days 6-13 --target-days 11-13
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import duckdb

from polymarket_gnn.data.wallet_skill import BEHAVIOUR_COLUMNS, build_running_behaviour, build_running_skill
from polymarket_gnn.evaluation.baselines import parse_days
from smart_money_test import DEFAULT_BASE_DIR, MONTH, YEAR, build_trade_table, convention_check, sql_list

logger = logging.getLogger(__name__)


def build_behaviour(args):
    history, target = parse_days(args.history_days), parse_days(args.target_days)
    if not set(target) <= set(history):
        raise SystemExit("--target-days must be inside --history-days")
    out = args.out_dir if args.out_dir != Path("wallet_skill") else Path("wallet_behaviour")
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory_limit}'; SET preserve_insertion_order=false")
    t0 = time.time()
    globs = [args.base_dir / f"partitioned/transactions/year={YEAR}/month={MONTH}/day={d}/hour=*/**/*.parquet" for d in history]
    con.execute(f"""
        CREATE TABLE orient AS
        SELECT market_id, CASE WHEN lower(trim(any_value(outcome_1))) = 'yes' THEN 1 ELSE -1 END AS o
        FROM read_parquet('{args.base_dir.parent / "universe_markets.parquet"}') WHERE outcome_1 IS NOT NULL GROUP BY 1
    """)
    con.execute(f"""
        CREATE TABLE tx AS
        SELECT x.wallet, epoch_ns(x.timestamp) AS ts, x.timestamp, x.market_id, (x.role = 'MAKER')::INT AS is_maker,
               ((CASE WHEN x.side = 'BUY' THEN 1 ELSE -1 END) * x.token_direction * coalesce(q.o, 1))::INT AS d,
               x.total_usdc AS usdc, x.day, x.hour
        FROM read_parquet({sql_list(globs)}, hive_partitioning=true) x LEFT JOIN orient q ON x.market_id = q.market_id
    """)
    build_running_behaviour(con)
    n, seen = con.execute("SELECT count(*), avg((beh_n > 0)::INT) FROM behaviour").fetchone()
    logger.info(f"behaviour: {n} (wallet, time) rows in {time.time() - t0:.0f}s; wallet seen before in {seen:.3f}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cols = ", ".join(f"b.{c}" for c in BEHAVIOUR_COLUMNS)
    con.execute(f"""
        COPY (SELECT r.wallet, r.timestamp, {cols}, r.day, r.hour
              FROM (SELECT DISTINCT wallet, ts, timestamp, day, hour FROM tx WHERE day IN ({", ".join(str(d) for d in target)})) r
              JOIN behaviour b ON r.wallet = b.wallet AND r.ts = b.ts)
        TO '{out}' (FORMAT PARQUET, PARTITION_BY (day, hour))
    """)
    logger.info(f"written to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=3600)
    p.add_argument("--history-days", type=str, default="6-13", help="days whose trades build the track records")
    p.add_argument("--target-days", type=str, default="11-13", help="days whose transactions get the feature")
    p.add_argument("--max-lag", type=float, default=600.0)
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("wallet_skill"))
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--memory-limit", type=str, default="120GB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--behaviour", action="store_true",
                   help="build the behaviour features instead (no outcomes, no horizon): <out-dir>/day=<d>/hour=<h>/")
    args = p.parse_args()
    if args.behaviour:
        return build_behaviour(args)

    H = args.horizon * 10**9
    label = f"{args.horizon // 3600}h" if args.horizon >= 3600 else f"{args.horizon}s"
    history, target = parse_days(args.history_days), parse_days(args.target_days)
    if not set(target) <= set(history):
        raise SystemExit("--target-days must be inside --history-days")
    out = args.out_dir / f"h{label}"
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory_limit}'; SET preserve_insertion_order=false")
    build_trade_table(con, args.base_dir, history, H, int(args.max_lag * 10**9))
    convention_check(con)

    t0 = time.time()
    globs = [args.base_dir / f"partitioned/transactions/year={YEAR}/month={MONTH}/day={d}/hour=*/**/*.parquet" for d in target]
    con.execute(f"""
        CREATE TABLE rows AS
        SELECT DISTINCT wallet, role, timestamp, epoch_ns(timestamp) AS ts, day, hour
        FROM read_parquet({sql_list(globs)}, hive_partitioning=true)
    """)
    con.execute("CREATE TABLE q AS SELECT DISTINCT wallet, role, ts FROM rows")
    build_running_skill(con, H, args.seed)
    n, cov, pcov, mx = con.execute(
        "SELECT count(*), avg((wins + losses > 0)::INT), avg((p_wins + p_losses > 0)::INT), max(wins + losses) FROM skill").fetchone()
    logger.info(f"skill: {n} (wallet, role, time) rows in {time.time() - t0:.0f}s; with a record {cov:.3f} "
                f"(placebo {pcov:.3f}); largest record {mx}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    con.execute(f"""
        COPY (SELECT r.wallet, r.role, r.timestamp, s.wins, s.losses, s.p_wins, s.p_losses, r.day, r.hour
              FROM rows r JOIN skill s ON r.wallet = s.wallet AND r.role = s.role AND r.ts = s.ts)
        TO '{out}' (FORMAT PARQUET, PARTITION_BY (day, hour))
    """)
    logger.info(f"written to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
