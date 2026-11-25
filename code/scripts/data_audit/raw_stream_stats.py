"""Statistics of the raw order-book event stream (one day) that matter for the tensor construction.
Usage: python raw_stream_stats.py [day]   (default 12 = 12 March 2026, a training day)."""
import sys, duckdb, numpy as np, pandas as pd

import numpy as _np, pandas as _pd
def _md(self, index=True, floatfmt=".4g", **kw):
    df = self.to_frame() if isinstance(self, _pd.Series) else self
    df = df.reset_index() if index else df
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, _np.floating)): return "nan" if _np.isnan(v) else format(v, floatfmt)
        return str(v)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False): out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)
_pd.DataFrame.to_markdown = _md
_pd.Series.to_markdown = _md
D = "/data/v.pampinella/agy_project/data/derived/partitioned"
day = int(sys.argv[1]) if len(sys.argv) > 1 else 12
con = duckdb.connect()
con.sql("set threads=96; set memory_limit='500GB'")
ob = f"{D}/orderbook/year=2026/month=3/day={day}/hour=*/*.parquet"
sn = f"{D}/snapshots/year=2026/month=3/day={day}/hour=*/*.parquet"
Q = lambda s: con.sql(s).df()
P = lambda title, df: print(f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt=".4g"))

print(f"# Raw order-book stream, 2026-03-{day:02d}\n")
con.sql(f"""create table ev as
  select market_id, timestamp_received ts, timestamp_created_at tc, best_bid, best_ask, change_price, change_size,
         change_side side, delta, mid_price mid
  from read_parquet('{ob}')""")
con.sql("""create table ev2 as
  select *, row_number() over w rn,
         ts - lag(ts) over w dt,
         mid - lag(mid) over w dmid,
         case when side='BUY' then change_price - best_bid else change_price - best_ask end dist_best
  from ev window w as (partition by market_id order by ts)""")

P("Totals", Q("""select count(*) events, count(distinct market_id) markets,
   count(*) filter (where delta is null) delta_null, count(*) filter (where mid is null) mid_null,
   count(*) filter (where best_bid is null or best_ask is null) onesided,
   count(*) filter (where best_bid >= best_ask) crossed from ev"""))

P("Events per market per day (quantiles) and concentration", Q("""
 with c as (select market_id, count(*) n from ev group by 1), r as (select n, row_number() over (order by n desc) k, count(*) over () m, sum(n) over () tot from c)
 select quantile_cont(n, 0.5) p50, quantile_cont(n, 0.9) p90, quantile_cont(n, 0.99) p99, max(n) mx,
        sum(n) filter (where k <= m*0.01) / max(tot) share_top1pct_markets,
        sum(n) filter (where k <= m*0.10) / max(tot) share_top10pct_markets,
        count(*) filter (where n < 20) markets_lt20_events, count(*) filter (where n < 200) markets_lt200_events, count(*) m from r"""))

P("Event types", Q("""select side, count(*) n, avg((delta > 0)::int) frac_add, avg((delta < 0)::int) frac_remove, avg((delta = 0)::int) frac_zero,
   avg((change_size = 0)::int) frac_size0, avg((dist_best = 0)::int) frac_at_best, avg((abs(dist_best) <= 0.01)::int) frac_within_1cent,
   quantile_cont(abs(dist_best), 0.5) dist_best_p50, quantile_cont(abs(dist_best), 0.9) dist_best_p90
   from ev2 group by 1 order by 1"""))

P("Mid changes per event", Q("""select count(*) n, avg((dmid <> 0)::int) frac_mid_change,
   avg((abs(dmid) = 0.0005)::int) frac_half_tick, avg((abs(dmid) >= 0.0005 and abs(dmid) < 0.005)::int) frac_lt_half_cent,
   avg((abs(dmid) >= 0.005 and abs(dmid) < 0.0105)::int) frac_half_to_1cent, avg((abs(dmid) > 0.0105)::int) frac_gt_1cent
   from ev2 where dmid is not null"""))
P("Mid changes by price band", Q("""select case when mid < 0.05 or mid > 0.95 then 'extreme' else 'central' end band,
   count(*) n, avg((dmid <> 0)::int) frac_mid_change, quantile_cont(abs(dmid), 0.5) filter (where dmid<>0) med_abs_move,
   quantile_cont(abs(dmid), 0.9) filter (where dmid<>0) p90_abs_move from ev2 where dmid is not null group by 1"""))

P("Inter-event time within a market (seconds)", Q("""select
   quantile_cont(epoch(dt), 0.1) p10, quantile_cont(epoch(dt), 0.5) p50, quantile_cont(epoch(dt), 0.9) p90, quantile_cont(epoch(dt), 0.99) p99,
   avg((epoch(dt) <= 0.001)::int) frac_le_1ms, avg((epoch(dt) <= 1)::int) frac_le_1s, avg((epoch(dt) <= 60)::int) frac_le_1min, avg((epoch(dt) > 3600)::int) frac_gt_1h
   from ev2 where dt is not null"""))

for K, name in [(20, "one tensor snapshot (20 events)"), (200, "GRU window (10 snapshots = 200 events)"), (640, "GRU K=32 window (640 events)"), (1280, "time-first window (64 snapshots = 1280 events)")]:
    P(f"Wall-clock span of {name}", Q(f"""
      with b as (select market_id, (rn-1)//{K} bk, epoch(max(ts)-min(ts)) span, count(*) n, sum(dmid<>0) moves, sum(abs(coalesce(dmid,0))) path
                 from ev2 group by 1,2 having count(*) = {K}),
           act as (select market_id, count(*) n_day from ev group by 1),
           j as (select b.*, case when n_day < 200 then 'quiet (<200 ev/day)' when n_day < 5000 then 'medium' else 'busy (>=5000 ev/day)' end act from b join act using (market_id))
      select act, count(*) windows, quantile_cont(span,0.1) span_p10_s, quantile_cont(span,0.5) span_p50_s, quantile_cont(span,0.9) span_p90_s,
             avg((span > 3600)::int) frac_gt_1h, avg((span > 60)::int) frac_gt_1min, avg((moves = 0)::int) frac_no_mid_change, quantile_cont(moves,0.5) moves_p50, quantile_cont(moves,0.9) moves_p90
      from j group by 1 union all
      select 'ALL', count(*), quantile_cont(span,0.1), quantile_cont(span,0.5), quantile_cont(span,0.9), avg((span > 3600)::int), avg((span > 60)::int), avg((moves = 0)::int), quantile_cont(moves,0.5), quantile_cont(moves,0.9) from j
      order by 1"""))

P("Burstiness: events per (market, minute) with at least one event", Q("""
  with m as (select market_id, date_trunc('minute', ts) mn, count(*) n from ev group by 1,2),
       alive as (select market_id, epoch(max(ts)-min(ts))/60 alive_min from ev group by 1)
  select count(*) active_market_minutes, sum(alive_min) alive_market_minutes, count(*)/sum(alive_min) frac_minutes_with_event,
         quantile_cont(n,0.5) ev_per_active_min_p50, quantile_cont(n,0.9) p90, quantile_cont(n,0.99) p99, max(n) mx
  from m join alive using (market_id)"""))

P("Receive vs creation timestamp lag (seconds)", Q("""select quantile_cont(epoch(ts-tc),0.5) p50, quantile_cont(epoch(ts-tc),0.9) p90, quantile_cont(epoch(ts-tc),0.99) p99, avg((epoch(ts-tc) < 0)::int) frac_negative from ev"""))

P("Full-book snapshots", Q(f"""with s as (select market_id, timestamp_received ts, length(data) len from read_parquet('{sn}')),
   g as (select market_id, count(*) n, epoch(max(ts)-min(ts))/count(*) mean_gap_s from s group by 1)
   select count(*) markets_with_snapshot, sum(n) snapshots, quantile_cont(n,0.5) per_market_p50, quantile_cont(n,0.9) per_market_p90,
          quantile_cont(mean_gap_s,0.5) gap_p50_s, (select quantile_cont(len,0.5) from s) json_len_p50, (select max(len) from s) json_len_max from g"""))

P("Depth of a full snapshot (levels per side, from the JSON)", Q(f"""with s as (select data from read_parquet('{sn}') using sample 20000 rows),
   d as (select len(json_extract(data,'$.bids')) nb, len(json_extract(data,'$.asks')) na from s)
   select quantile_cont(nb,0.5) bids_p50, quantile_cont(nb,0.9) bids_p90, quantile_cont(na,0.5) asks_p50, quantile_cont(na,0.9) asks_p90, avg((nb>10)::int) frac_bids_gt10, avg((na>10)::int) frac_asks_gt10 from d"""))
print("\nDONE")
