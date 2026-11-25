#!/usr/bin/env python
"""Light economic simulation of the direction calls on the locked test days.

For every sample (market, block) of the test days at one horizon H, a model gives a
direction call UP / DOWN / CONST (argmax of the log-probabilities plus the two
calibration offsets chosen on the validation day).  Each UP or DOWN call is traded
as one share of the Yes token, ignoring the impact of our own orders on the book,
with three execution styles:

  TT  taker in, taker out: cross the spread at t and cross it again at t+H.
  MT  maker in, taker out: rest at the best quote at t (a bid for UP, an ask for
      DOWN); the order counts as filled if a later fill of the same market within
      (t, t+H] prints at or through our price; close by crossing at t+H.  Calls
      that are not filled are not traded.
  MM  maker in, maker out: as MT, but the closing order also rests at the best
      quote at t+H and is assumed filled: an upper bound for a pure maker.

Fees: takers pay base_m * p * (1 - p) per share, where base_m is the fee rate
observed on the market's own taker fills in March 2026 (zero on fee-free
markets); makers pay nothing; rebates are ignored.  A second scenario applies no
fee at all.

Inputs (per horizon): the control's feature table and probabilities on the test
days (data_audit/test_<H>/) and, optionally, the test predictions of deep models
(val_predictions_*_test.parquet, joined on (t, market_idx)).
"""
import argparse
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


HSEC = {"1m": 60, "15m": 900, "1h": 3600, "6h": 21600, "12h": 43200}
TICK = 0.001

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", required=True, choices=list(HSEC))
ap.add_argument("--audit-dir", default="/data/v.pampinella/data_audit")
ap.add_argument("--fills", default="/data/v.pampinella/agy_project/data/derived/partitioned/transactions/year=2026/month=3/day=*/hour=*/*.parquet")
ap.add_argument("--ctrl-offsets", default="0,0", help="down,up offsets of the control (validation choice)")
ap.add_argument("--deep", action="append", default=[], help="name=parquet:down,up")
ap.add_argument("--out-dir", default="/data/v.pampinella/data_audit/econ")
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--max-rows", type=int, default=0, help="debug: keep only the first N rows")
args = ap.parse_args()

H = args.horizon
HS = HSEC[H]
os.makedirs(args.out_dir, exist_ok=True)
tdir = os.path.join(args.audit_dir, f"test_{H}")

# ----------------------------------------------------------------------------- load
cols = ["t", "market_id", "market_idx", "y", "dedup", "mid", "delta", "spread", "best_bid",
        "best_ask", "last_trade_price", "since_trade", "category"]
log(f"reading {tdir}/val_features_{H}.parquet")
df = pq.read_table(os.path.join(tdir, f"val_features_{H}.parquet"), columns=cols).to_pandas()
df["category"] = df["category"].astype(str)
proba = np.load(os.path.join(tdir, f"proba_{H}_G5.npy")).astype(np.float64)
assert proba.shape[0] == len(df), (proba.shape, len(df))
if args.max_rows:
    df = df.iloc[: args.max_rows].copy()
    proba = proba[: args.max_rows]
n_all = len(df)
df["rid"] = np.arange(n_all, dtype=np.int64)


def calls(p, off):
    """argmax of log p + (off_down, 0, off_up); returns pred in {0,1,2} and the confidence."""
    lp = np.log(np.clip(p, 1e-12, 1.0)) + np.array([off[0], 0.0, off[1]])
    pred = lp.argmax(1).astype(np.int8)
    conf = p[np.arange(len(p)), pred]
    return pred, conf


off = [float(x) for x in args.ctrl_offsets.split(",")]
df["pred_ctrl"], df["conf_ctrl"] = calls(proba, off)
df["pred_oracle"] = df["y"].astype(np.int8)
df["conf_oracle"] = 1.0
signals = ["ctrl", "oracle"]
log(f"{n_all:,} rows; control calls: {np.bincount(df.pred_ctrl, minlength=3).tolist()} (down, const, up)")

tns = df["t"].values.astype("datetime64[ns]").astype(np.int64)
for spec in args.deep:
    name, rest = spec.split("=", 1)
    path, offs = rest.rsplit(":", 1)
    doff = [float(x) for x in offs.split(",")]
    d = pq.read_table(path, columns=["t", "market_idx", "p_down", "p_const", "p_up"]).to_pandas()
    d["_tns"] = d["t"].values.astype("datetime64[ns]").astype(np.int64)
    d = d.drop_duplicates(["_tns", "market_idx"])
    pr, cf = calls(d[["p_down", "p_const", "p_up"]].values.astype(np.float64), doff)
    d[f"pred_{name}"] = pr
    d[f"conf_{name}"] = cf
    key = pd.DataFrame({"_tns": tns, "market_idx": df["market_idx"].values})
    m = key.merge(d[["_tns", "market_idx", f"pred_{name}", f"conf_{name}"]], on=["_tns", "market_idx"], how="left")
    df[f"pred_{name}"] = m[f"pred_{name}"].fillna(-1).astype(np.int8).values
    df[f"conf_{name}"] = m[f"conf_{name}"].fillna(0.0).values
    signals.append(name)
    log(f"deep {name}: {len(d):,} prediction rows, {(df[f'pred_{name}'] >= 0).sum():,} matched")

# ----------------------------------------------------------------------------- duckdb
con = duckdb.connect()
con.execute(f"PRAGMA threads={args.threads}")
con.register("s0", df)

log("fee base rate per market from the taker fills of March 2026")
con.execute(f"""
create table fee_base as
select market_id,
       quantile_cont(fee_rate / (price * (1 - price)), 0.5) as fee_base,
       count(*) as n_taker_legs,
       avg((fee_rate > 0)::int) as share_fee_legs
from read_parquet('{args.fills}', hive_partitioning=true)
where role = 'TAKER' and price > 0 and price < 1
group by market_id
""")
fb = con.sql("select count(*), sum((fee_base>0)::int) from fee_base").fetchall()[0]
log(f"  {fb[0]:,} markets with taker fills, {fb[1]:,} charge a fee")

log("window features: new fills after each row, min/max fill price in (t, t+H]")
con.execute(f"""
create table w as
with a as (
  select s.*, lag(t) over (partition by market_id order by t, rid) as prev_t,
         case when since_trade is not null and since_trade >= 0 and since_trade < 1e9
              then t - to_seconds(since_trade) end as trade_ts
  from s0 s
), b as (
  select *, case when trade_ts > prev_t then last_trade_price end as fill_px from a
)
select *,
  min(fill_px) over (partition by market_id order by t
                     range between to_seconds(1) following and to_seconds({HS}) following) as lo_px,
  max(fill_px) over (partition by market_id order by t
                     range between to_seconds(1) following and to_seconds({HS}) following) as hi_px,
  t + to_seconds({HS}) as t_exit
from b
""")

log("exit book: last row of the same market at or before t+H (asof)")
con.execute("""
create table e as
select s.rid, q.spread as spread_x, q.mid as mid_x, q.t as t_x
from w s asof join w q on s.market_id = q.market_id and s.t_exit >= q.t
""")

con.execute(f"""
create table r as
select w.*, e.spread_x, e.mid_x, e.t_x, coalesce(f.fee_base, 0.0) as fee_base,
  (w.spread > 0 and w.best_bid > 0 and w.best_ask < 1 and w.best_bid < w.best_ask) as valid,
  w.mid + w.delta as exit_mid,
  case when e.spread_x > 0 then e.spread_x else w.spread end / 2 as hs_x,
  abs(e.mid_x - (w.mid + w.delta)) < 1e-6 as exit_seen
from w left join e using (rid) left join fee_base f using (market_id)
""")
stats = con.sql("select count(*), sum(valid::int), sum(exit_seen::int), sum((lo_px is not null)::int) from r").fetchall()[0]
log(f"  rows {stats[0]:,}; valid books {stats[1]:,}; exit book seen in table {stats[2]:,}; rows with a later fill in (t,t+H] {stats[3]:,}")

# ----------------------------------------------------------------------------- strategies
# One share.  d = +1 for UP, -1 for DOWN.  Prices: entry book (bid, ask) at t, exit book
# (exit_mid -/+ hs_x) at t+H.  Fees on taker legs: fee_base * p * (1 - p).
STRATS = {
    # strategy: (entry price expr, exit price expr, filled condition, taker legs for fees)
    "TT": ("case when d>0 then best_ask else best_bid end",
           "case when d>0 then exit_mid - hs_x else exit_mid + hs_x end",
           "true", "both"),
    "MT": ("case when d>0 then best_bid else best_ask end",
           "case when d>0 then exit_mid - hs_x else exit_mid + hs_x end",
           "case when d>0 then lo_px <= best_bid else hi_px >= best_ask end", "exit"),
    "MM": ("case when d>0 then best_bid else best_ask end",
           "case when d>0 then exit_mid + hs_x else exit_mid - hs_x end",
           "case when d>0 then lo_px <= best_bid else hi_px >= best_ask end", "none"),
}


def fee_expr(legs, scen):
    if scen == "nofee":
        return "0.0"
    ent = "fee_base * entry * (1 - entry)"
    ext = "fee_base * exit * (1 - exit)"
    return {"both": f"({ent} + {ext})", "exit": ext, "none": "0.0"}[legs]


rows = []
for sig in signals:
    con.execute(f"""
    create or replace table c as
    select rid, market_id, category, dedup, y, mid, delta, spread, best_bid, best_ask, lo_px, hi_px,
           exit_mid, hs_x, fee_base, valid,
           case when pred_{sig} = 2 then 1 when pred_{sig} = 0 then -1 else 0 end as d,
           conf_{sig} as conf
    from r where pred_{sig} in (0, 2) and valid
    """)
    n_calls = con.sql("select count(*) from c").fetchall()[0][0]
    if n_calls == 0:
        continue
    q90 = con.sql("select quantile_cont(conf, 0.9) from c").fetchall()[0][0]
    for strat, (entry, exit_, filled, legs) in STRATS.items():
        for scen in ("fee", "nofee"):
            con.execute(f"""
            create or replace table p as
            select *, entry, exit, filled,
              d * (exit - entry) - fee as pnl,
              case when d > 0 then entry else 1 - entry end as capital,
              d * (exit_mid - mid) as move,
              d * (exit - entry) - d * (exit_mid - mid) as spread_cost,
              fee,
              case when d * delta > 0 then 1 when delta = 0 then 0 else -1 end as outcome
            from (
              select c.*, {entry} as entry, {exit_} as exit, ({filled}) as filled
              from c
            ) c2
            cross join lateral (select {fee_expr(legs, scen)} as fee) f
            """)

            def agg(where, subset):
                r_ = con.sql(f"""
                select count(*) n_calls, coalesce(sum(filled::int), 0) n_trades,
                  avg(case when filled then pnl end) * 1000 mean_pnl_ticks,
                  avg(case when filled then pnl / capital end) * 100 mean_ret_pct,
                  avg(case when filled then (pnl > 0)::int end) hit_rate,
                  avg(case when filled then move end) * 1000 mean_move_ticks,
                  avg(case when filled then spread_cost end) * 1000 mean_spread_ticks,
                  avg(case when filled then fee end) * 1000 mean_fee_ticks,
                  sum(case when filled then pnl else 0 end) total_pnl_usd,
                  avg(case when filled then (outcome = 1)::int end) precision_dir,
                  avg(case when filled then (outcome = 0)::int end) share_const,
                  quantile_cont(case when filled then spread end, 0.5) * 1000 median_spread_ticks,
                  avg(case when filled and outcome = 1 then abs(delta) end) * 1000 move_when_right_ticks,
                  avg(case when filled and outcome = -1 then abs(delta) end) * 1000 move_when_wrong_ticks
                from p where {where}
                """).fetchall()[0]
                rows.append(dict(horizon=H, signal=sig, strategy=strat, fees=scen, subset=subset,
                                 **dict(zip(["n_calls", "n_trades", "mean_pnl_ticks", "mean_ret_pct", "hit_rate",
                                             "mean_move_ticks", "mean_spread_ticks", "mean_fee_ticks", "total_pnl_usd",
                                             "precision_dir", "share_const", "median_spread_ticks",
                                             "move_when_right_ticks", "move_when_wrong_ticks"], r_))))

            agg("true", "all")
            agg("dedup", "dedup")
            agg(f"conf >= {q90}", "top10")
            agg("d > 0", "up")
            agg("d < 0", "down")
            agg("fee_base > 0", "fee_markets")
            agg("fee_base = 0", "free_markets")
            for cat in con.sql("select distinct category from c order by 1").fetchall():
                agg(f"category = '{cat[0]}'", f"cat:{cat[0]}")
        log(f"  {sig} {strat} done")

out = pd.DataFrame(rows)
csv = os.path.join(args.out_dir, f"econ_{H}.csv")
out.to_csv(csv, index=False)

# ----------------------------------------------------------------------------- report
fmt = lambda x, nd=2: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
md = [f"# Economic simulation, horizon {H}, test days 24-26 March", "",
      f"Rows {n_all:,}; valid books {stats[1]:,}; control offsets {off}; deep signals {args.deep}.", "",
      "P&L per share in ticks of 0.001; return per unit of capital in %; one share per call.", ""]
for sig in signals:
    for scen in ("fee", "nofee"):
        md += [f"## {sig}, {'fees as charged in March 2026' if scen == 'fee' else 'no fees'}", "",
               "| strategy | subset | calls | trades | fill | P&L/share (ticks) | return % | hit | precision | const | move (ticks) | spread cost | fee | total $ | med. spread | move right | move wrong |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        sub = out[(out.signal == sig) & (out.fees == scen)]
        for _, r_ in sub.iterrows():
            if not r_.n_calls:
                continue
            fill = r_.n_trades / r_.n_calls
            md.append(f"| {r_.strategy} | {r_.subset} | {int(r_.n_calls):,} | {int(r_.n_trades):,} | {fmt(fill)} | "
                      f"{fmt(r_.mean_pnl_ticks)} | {fmt(r_.mean_ret_pct)} | {fmt(r_.hit_rate, 3)} | {fmt(r_.precision_dir, 3)} | "
                      f"{fmt(r_.share_const, 3)} | {fmt(r_.mean_move_ticks)} | {fmt(r_.mean_spread_ticks)} | {fmt(r_.mean_fee_ticks)} | "
                      f"{fmt(r_.total_pnl_usd, 0)} | {fmt(r_.median_spread_ticks, 1)} | {fmt(r_.move_when_right_ticks, 1)} | {fmt(r_.move_when_wrong_ticks, 1)} |")
        md.append("")
with open(os.path.join(args.out_dir, f"econ_{H}.md"), "w") as f:
    f.write("\n".join(md))
with open(os.path.join(args.out_dir, f"econ_{H}_meta.json"), "w") as f:
    json.dump(dict(horizon=H, rows=int(n_all), valid=int(stats[1]), exit_seen=int(stats[2]),
                   later_fill=int(stats[3]), ctrl_offsets=off, deep=args.deep,
                   fee_markets=int(fb[1]), markets_with_fills=int(fb[0])), f, indent=1)
log(f"ALL DONE -> {csv}")
