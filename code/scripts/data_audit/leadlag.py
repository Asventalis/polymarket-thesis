"""leadlag.py — minute-scale lead–lag between sibling contracts (a description of the data, not a model input).

For every pair of siblings (groups of sibling_features.py) and every mid-price change of market i at minute m, count
the changes of its sibling j in the minutes m+1 … m+K (and m−K … m−1), by lag and by sign product (+1: same direction,
−1: opposite).  Under independence the expected count at a lag is n_moves(i) × rate(j), where rate(j) is j's share of
minutes with a change over the window; the ratio observed / expected is the excess co-movement at that lag, and the
share of opposite signs says whether siblings move together (two sides of a game: opposite; one candidate of an
exclusive event up, the others down: opposite; two strikes of one ladder: same).  Days 6–13 March, hand-feature rows
(one per market every 20 book events; the first row of a minute with a new mid gives the minute's change).

Strata: game groups (sports / esports), exclusive template groups (sum of the members' last mids within [0.8, 1.2]),
other template groups.  Output: <out>/leadlag_1m.md and leadlag_counts.parquet.

    ./run_dev.sh scripts/data_audit/leadlag.py [--max-lag 15]
"""
import argparse, time
from pathlib import Path

import duckdb, numpy as np, pandas as pd

DATA = Path("/data/v.pampinella/agy_project/data")
HAND = DATA / "derived/hand_features"
GROUPS = DATA / "derived/sibling_features/groups.parquet"
ap = argparse.ArgumentParser()
ap.add_argument("--max-lag", type=int, default=15); ap.add_argument("--days", default="6-13")
ap.add_argument("--out", type=Path, default=Path("/data/v.pampinella/data_audit/leadlag"))
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)
lo, hi = (int(x) for x in a.days.split("-"))
files = ", ".join(f"'{HAND}/day=2026-03-{d:02d}.parquet'" for d in range(lo, hi + 1))
con = duckdb.connect(); con.sql("set threads=96; set memory_limit='600GB'; set preserve_insertion_order=false")
K = a.max_lag

# minute-level mid changes: the last mid of every (market, minute); a change = a minute whose last mid differs from the previous minute's
con.sql(f"""create table mins as
  select market_id, ts // 60000000000 as minute, arg_max(mid, ts) mid
  from read_parquet([{files}]) where mid is not null group by 1, 2""")
con.sql("""create table moves as
  select market_id, minute, sign(mid - lag(mid) over (partition by market_id order by minute)) s
  from mins qualify s <> 0 and s is not null""")
n_min = con.sql("select count(*) from mins").fetchone()[0]; n_mov = con.sql("select count(*) from moves").fetchone()[0]
log(f"{n_min:,} market-minutes with a quote, {n_mov:,} minute changes")
# per market: number of changes and the active window (first to last quoted minute) → rate
con.sql("""create table rate as
  select m.market_id, coalesce(v.n, 0) n_moves, (max(m.minute) - min(m.minute) + 1) span,
         coalesce(v.n, 0) / greatest(max(m.minute) - min(m.minute) + 1, 1) rate
  from mins m left join (select market_id, count(*) n from moves group by 1) v using (market_id) group by m.market_id, v.n""")

# strata of the groups
con.sql(f"""create table grp as select market_id, group_key, kind from read_parquet('{GROUPS}')""")
con.sql("""create table lastmid as select market_id, arg_max(mid, minute) mid from mins group by 1""")
con.sql("""create table stratum as
  select g.group_key, case when any_value(g.kind) = 'game' then 'game'
                           when sum(l.mid) between 0.8 and 1.2 then 'template, exclusive' else 'template, other' end stratum
  from grp g join lastmid l using (market_id) group by g.group_key""")
# ordered pairs with i the MORE ACTIVE market of the two (higher rate of changing minutes; ties by id): a positive lag is
# "the quieter sibling changes after the active one", a negative lag the reverse, so the asymmetry of the histogram is the lead
con.sql("""create table pairs as
  select a.market_id i, b.market_id j, s.stratum
  from grp a join grp b on a.group_key = b.group_key and a.market_id <> b.market_id join stratum s on s.group_key = a.group_key
  join rate ra on ra.market_id = a.market_id join rate rb on rb.market_id = b.market_id
  where ra.rate > rb.rate or (ra.rate = rb.rate and a.market_id < b.market_id)""")
log(f"pairs: {con.sql('select stratum, count(*) from pairs group by 1').fetchall()}")

# observed co-moves by lag (j's change at m + lag, lag in [-K, K] \ {0}) and sign product
con.sql(f"""create table co as
  select p.stratum, (mj.minute - mi.minute) lag, (mi.s * mj.s) sprod, count(*) n
  from pairs p join moves mi on mi.market_id = p.i join moves mj on mj.market_id = p.j
   and mj.minute between mi.minute - {K} and mi.minute + {K}
  group by 1, 2, 3""")
# expected count per (stratum, lag) under independence: sum over pairs of n_moves(i) * rate(j)  (j's changes are a Bernoulli per minute)
exp = con.sql("""select p.stratum, sum(ri.n_moves * rj.rate) expected, count(*) n_pairs, sum(ri.n_moves) n_moves_i
  from pairs p join rate ri on ri.market_id = p.i join rate rj on rj.market_id = p.j group by 1""").df().set_index("stratum")
co = con.sql("select * from co").df(); co.to_parquet(a.out / "leadlag_counts.parquet", index=False)
tab = co.pivot_table(index=["stratum", "lag"], columns="sprod", values="n", aggfunc="sum", fill_value=0).reset_index()
tab.columns = ["stratum", "lag"] + [f"s{int(c):+d}" for c in tab.columns[2:]]
tab["n"] = tab.get("s-1", 0) + tab.get("s+1", 0)
tab["expected"] = tab.stratum.map(exp.expected)
tab["ratio"] = tab.n / tab.expected
tab["opposite_share"] = tab.get("s-1", 0) / tab.n.replace(0, np.nan)

lines = ["# Minute-scale lead–lag between sibling contracts (6–13 March)\n",
         f"Minute changes of the mid (hand-feature rows, last mid of every minute): {n_mov:,} over {n_min:,} market-minutes. "
         "For every change of market i (the more active market of the pair), the changes of its quieter sibling j at lags −K…+K minutes (K = %d; lag 0 = the same minute), counted by lag and by the product of the two signs; "
         "`ratio` = observed / expected under independence (n_moves(i) × j's rate of changing minutes); `opposite` = share of pairs of changes with opposite signs. "
         "A ratio above 1 at positive lags means the quieter sibling tends to change after the active one; the asymmetry between +k and −k is the lead.\n" % K]
for st, g in tab.groupby("stratum"):
    e = exp.loc[st]
    lines += [f"## {st} — {int(e.n_pairs):,} ordered pairs, {int(e.n_moves_i):,} changes of i, expected {e.expected:,.0f} co-changes per lag\n",
              "| lag (min) | co-changes | ratio | opposite |", "|---|---|---|---|"]
    for r in g.sort_values("lag").itertuples():
        lines.append(f"| {int(r.lag):+d} | {int(r.n):,} | {r.ratio:.2f} | {r.opposite_share:.2f} |")
    pos = g[g.lag > 0]; neg = g[g.lag < 0]; r0 = g[g.lag == 0].ratio.iloc[0]
    for k in (1, 5, 15):
        a_, b_ = g[g.lag == k].n.iloc[0], g[g.lag == -k].n.iloc[0]
        lines.append(f"- lag ±{k}: quieter-after-active {a_:,} vs active-after-quieter {b_:,}, asymmetry {(a_ - b_) / (a_ + b_):+.3f}")
    lines.append(f"\nSame minute: ratio {r0:.2f}. Sum over lags +1…+{K}: ratio {pos.n.sum() / (e.expected * K):.2f}; −{K}…−1: {neg.n.sum() / (e.expected * K):.2f}; "
                 f"opposite-sign share at lag 0 / +1: {g[g.lag == 0].opposite_share.iloc[0]:.2f} / {g[g.lag == 1].opposite_share.iloc[0]:.2f}.\n")
    log(f"{st}: lag 0 {r0:.2f}, +1 {g[g.lag == 1].ratio.iloc[0]:.2f} vs −1 {g[g.lag == -1].ratio.iloc[0]:.2f}, +1..+{K} {pos.n.sum() / (e.expected * K):.2f} vs −{K}..−1 {neg.n.sum() / (e.expected * K):.2f}, opposite at +1 {g[g.lag == 1].opposite_share.iloc[0]:.2f}")
text = "\n".join(lines) + "\n"; (a.out / "leadlag_1m.md").write_text(text); print(text)
log("ALL DONE")
