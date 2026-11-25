"""
hand_features.py — causal hand-made features for EVERY tensor row of 6–13 March 2026.

The no-state control (scripts/data_audit/feature_audit.py) computed 61 features at the
validation samples and beat the deep models.  This script computes the same features at every
LOB tensor row (one per market every 20 book events) so that the training loop can feed them to
the deep model through ``HandFeatureIndex`` (``--hand-dir``).  Same definitions as the control,
with three differences stated here:
  * the trade features are taken as of the row's timestamp (the control: as of the sample time);
  * ``since_row`` is dropped (the model only ever reads the row of the current block);
  * the time-dependent features (hour, time since the last mid change, time since the last trade,
    hours to the end / since the start of the market) are stored as the ingredients
    (``t_last_move``, ``since_trade_s``, per-market start/end) and recomputed at query time.

Output (``OUT``):  day=YYYY-MM-DD.parquet (market_id, ts ns, feature columns, t_last_move ns),
markets.parquet (market_id, category code, is_yesno, start_ns, end_ns, duration_h),
meta.json (column list, robust location/scale of every model feature on the rows of 6–12 March,
category names).  CPU only; ~1 h with 110 threads.

    hand_features.py [OUT] [--days 6-13] [--history-from D] [--meta-from DIR] [--threads 110]

Other days (the test): ``--days 21-26 --history-from 19 --meta-from <reference dir>`` loads the rows from
19 March so that the rolling windows, the last mid change and the last trade have history, writes only
the day files of 21-26 March and copies meta.json from the reference directory: the robust location and
scale of every feature stay those of 6-12 March, as the trained models expect.
"""
import argparse, json, sys, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd

DATA = Path("/data/v.pampinella/agy_project/data")
ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", type=Path, default=DATA / "derived/hand_features")
ap.add_argument("--days", default="6-13", help="days of March 2026 whose files are written ('6-13' or '21,22,23')")
ap.add_argument("--history-from", type=int, default=None, help="first day loaded (default: the first of --days)")
ap.add_argument("--meta-from", type=Path, default=None, help="copy meta.json (the standardisation statistics) from this directory instead of computing it")
ap.add_argument("--stat-days", default="6-12", help="days of the statistics when meta.json is computed")
ap.add_argument("--threads", type=int, default=110)
args = ap.parse_args()
def _days(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return out
OUT = args.out; OUT.mkdir(parents=True, exist_ok=True)
DAYS = _days(args.days); STAT_DAYS = _days(args.stat_days)
FIRST = args.history_from if args.history_from is not None else DAYS[0]
if FIRST > DAYS[0]:
    raise SystemExit("--history-from must not be after the first of --days")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

CATS = [("esports", {"esports"}), ("sports", {"sports"}), ("crypto", {"crypto", "crypto-prices", "bitcoin", "ethereum", "solana"}),
        ("weather", {"weather"}), ("politics", {"politics", "elections", "geopolitics", "global-elections", "world-elections", "world", "trump", "midterms", "primaries", "us-presidential-election"}),
        ("economy_tech", {"finance", "economy", "business", "tech", "big-tech", "ai", "ipos", "macro-indicators", "pre-market", "economic-policy", "fdv"}),
        ("culture", {"pop-culture", "awards", "movies", "music", "celebrities", "oscars"})]
CAT_NAMES = [c for c, _ in CATS] + ["other"]
def category(tags):
    s = frozenset(t.strip().lower() for t in (tags or "").split(",") if t.strip())
    for name, keys in CATS:
        if s & keys: return name
    return "other"

con = duckdb.connect()
con.sql(f"set threads={args.threads}; set memory_limit='700GB'; set preserve_insertion_order=false")
log(f"days {DAYS[0]}-{DAYS[-1]} written, rows loaded from day {FIRST}, {args.threads} threads"
    + (f", meta.json copied from {args.meta_from}" if args.meta_from else f", statistics on days {STAT_DAYS[0]}-{STAT_DAYS[-1]}"))

# ── 1. tensor rows → per-row book columns and rolling features (as feature_audit.py) ─────────
glob = str(DATA / "derived/tensors/year=2026/month=3/day=*/hour=*/lob_features.parquet")
con.sql(f"""create table rows as
  select market_id, timestamp ts,
    features[41] mid, features[42] spread, features[43] best_bid, features[44] best_ask,
    features[45] ofi, features[46] volume, features[48] buy_ratio, features[49] cancel_ratio, features[52] depth_skew,
    features[11] bid_sz0, features[31] ask_sz0,
    features[11]+features[12]+features[13]+features[14]+features[15] bid_sz5,
    features[31]+features[32]+features[33]+features[34]+features[35] ask_sz5,
    list_sum(features[11:20]) bid_sz10, list_sum(features[31:40]) ask_sz10,
    length(list_filter(features[11:20], x -> x > 0)) bid_levels, length(list_filter(features[31:40], x -> x > 0)) ask_levels,
    features[2] bid_rel1, features[22] ask_rel1
  from read_parquet('{glob}') where day(timestamp) between {FIRST} and {DAYS[-1]}""")
log(f"rows: {con.sql('select count(*) from rows').fetchone()[0]:,}")
con.sql("""create table r1 as select *, epoch(ts - lag(ts) over w) dt_prev, mid - lag(mid) over w dmid
  from rows window w as (partition by market_id order by ts)""")
con.sql("drop table rows")
con.sql("""create table feat as
  select *,
    last_value(case when dmid <> 0 then sign(dmid) end ignore nulls) over w2 last_move_sign,
    last_value(case when dmid <> 0 then ts end ignore nulls) over w2 t_last_move,
    count(*) over (partition by market_id order by ts range between interval 1 minute preceding and current row) n_1m,
    count(*) over (partition by market_id order by ts range between interval 5 minute preceding and current row) n_5m,
    count(*) over (partition by market_id order by ts range between interval 15 minute preceding and current row) n_15m,
    count(*) over (partition by market_id order by ts range between interval 60 minute preceding and current row) n_60m,
    sum(ofi) over (partition by market_id order by ts rows between 4 preceding and current row) ofi_5,
    sum(ofi) over (partition by market_id order by ts rows between 19 preceding and current row) ofi_20,
    sum(ofi) over (partition by market_id order by ts range between interval 15 minute preceding and current row) ofi_15m,
    sum(volume) over (partition by market_id order by ts range between interval 15 minute preceding and current row) vol_15m,
    sum(case when dmid <> 0 then 1 else 0 end) over (partition by market_id order by ts range between interval 60 minute preceding and current row) moves_60m,
    sum(power(coalesce(dmid, 0), 2)) over (partition by market_id order by ts range between interval 60 minute preceding and current row) rv_60m,
    max(mid) over (partition by market_id order by ts range between interval 60 minute preceding and current row) max_60m,
    min(mid) over (partition by market_id order by ts range between interval 60 minute preceding and current row) min_60m,
    avg(mid) over (partition by market_id order by ts range between interval 60 minute preceding and current row) mean_60m
  from r1 window w2 as (partition by market_id order by ts rows between unbounded preceding and current row)""")
con.sql("drop table r1")
con.sql("create table pm as select market_id, ts, mid from feat")
log("rolling features done")

# ── 2. trades per market-minute, oriented to the book's token, cumulative ──────────────────────
tglob = str(DATA / "derived/partitioned/transactions/year=2026/month=3/day=*/hour=*/*.parquet")
con.sql(f"""create table trades as
  with o as (select market_id, case when outcome_1 = 'Yes' then 1 else -1 end orient from '{DATA}/universe_markets.parquet'),
  t as (select tr.market_id, timestamp tts, date_trunc('minute', timestamp) mn,
          orient * token_direction * (case when side = 'BUY' then 1 else -1 end) * total_usdc signed, total_usdc usd, price
        from read_parquet('{tglob}') tr join o using (market_id)
        where role = 'TAKER' and day(timestamp) between {FIRST} and {DAYS[-1]}),
  m as (select market_id, mn, sum(signed) s, sum(usd) u, count(*) n, arg_max(price, tts) last_price from t group by 1,2)
  select market_id, mn + interval 1 minute ts, last_price, sum(s) over w cs, sum(u) over w cu, sum(n) over w cn
  from m window w as (partition by market_id order by mn rows between unbounded preceding and current row)""")
log(f"trade minutes: {con.sql('select count(*) from trades').fetchone()[0]:,}")

# ── 3. market table ───────────────────────────────────────────────────────────────────────────
mk = con.sql(f"""select market_id, tags, outcome_1 = 'Yes' is_yesno, start_date, end_date,
                 epoch(end_date - start_date)/3600 duration_h from '{DATA}/universe_markets.parquet'""").df()
mk["category"] = mk.tags.map(category).map({c: i for i, c in enumerate(CAT_NAMES)}).astype(np.int8)
mk["start_ns"] = pd.to_datetime(mk.start_date).values.astype("datetime64[ns]").astype("int64")
mk["end_ns"] = pd.to_datetime(mk.end_date).values.astype("datetime64[ns]").astype("int64")
mk["is_yesno"] = mk.is_yesno.fillna(False).astype(np.float32)
mk["duration_h"] = mk.duration_h.astype(np.float32)
mk[["market_id", "category", "is_yesno", "start_ns", "end_ns", "duration_h"]].to_parquet(OUT / "markets.parquet", index=False)
log(f"markets: {len(mk):,}")

# ── 4. lagged mids and trades attached to every row, finished columns, one file per day ───────
con.sql("""create table allrows as
  select f.*, l1.mid mid_15m, l2.mid mid_1h, l3.mid mid_3h, l4.mid mid_6h, l5.mid mid_12h,
         x0.cs cs0, x0.cu cu0, x0.cn cn0, x0.last_price last_trade_price, epoch(f.ts - x0.ts) since_trade,
         x5.cs cs5, x15.cs cs15, x15.cu cu15, x15.cn cn15, x60.cs cs60, x60.cu cu60
  from feat f
  asof left join pm l1 on f.market_id = l1.market_id and l1.ts <= f.ts - interval 15 minute
  asof left join pm l2 on f.market_id = l2.market_id and l2.ts <= f.ts - interval 1 hour
  asof left join pm l3 on f.market_id = l3.market_id and l3.ts <= f.ts - interval 3 hour
  asof left join pm l4 on f.market_id = l4.market_id and l4.ts <= f.ts - interval 6 hour
  asof left join pm l5 on f.market_id = l5.market_id and l5.ts <= f.ts - interval 12 hour
  asof left join trades x0 on f.market_id = x0.market_id and x0.ts <= f.ts
  asof left join trades x5 on f.market_id = x5.market_id and x5.ts <= f.ts - interval 5 minute
  asof left join trades x15 on f.market_id = x15.market_id and x15.ts <= f.ts - interval 15 minute
  asof left join trades x60 on f.market_id = x60.market_id and x60.ts <= f.ts - interval 60 minute""")
con.sql("drop table feat; drop table pm; drop table trades")
log("lags and trades attached")

# stored columns (raw, float32); the index standardises them with meta.json
STORED = ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h",
          "spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10",
          "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1",
          "ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev",
          "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "last_move_sign",
          "flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "since_trade_s"]
con.sql("""create view fin as
  select market_id, epoch_ns(ts) ts, epoch_ns(t_last_move) t_last_move,
    mid, mid - mid_15m ret_15m, mid - mid_1h ret_1h, mid - mid_3h ret_3h, mid - mid_6h ret_6h, mid - mid_12h ret_12h,
    spread / tick spread_ticks,
    (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0 + 1e-6) imb1, (bid_sz5 - ask_sz5) / (bid_sz5 + ask_sz5 + 1e-6) imb5, (bid_sz10 - ask_sz10) / (bid_sz10 + ask_sz10 + 1e-6) imb10,
    ((best_ask * bid_sz0 + best_bid * ask_sz0) / (bid_sz0 + ask_sz0 + 1e-6) - mid) / tick micro_minus_mid_ticks,
    ln(1 + bid_sz0) log_bid_sz0, ln(1 + ask_sz0) log_ask_sz0, ln(1 + bid_sz10) log_bid_sz10, ln(1 + ask_sz10) log_ask_sz10,
    bid_levels, ask_levels, depth_skew, bid_rel1, ask_rel1,
    ofi, ofi_5, ofi_20, ofi_15m, ofi_20 / (abs(volume) * 20 + 1.0) ofi_20_z, ofi_15m / (vol_15m + 1.0) ofi_15m_z, ln(1 + vol_15m) log_vol_15m,
    buy_ratio, cancel_ratio, ln(1 + coalesce(dt_prev, 0)) log_dt_prev,
    n_1m, n_5m, n_15m, n_60m, moves_60m, sqrt(rv_60m) rv_60m,
    (mid - min_60m) / (max_60m - min_60m + 1e-6) - 0.5 range_pos_60m, mid - mean_60m mid_minus_mean60, coalesce(last_move_sign, 0) last_move_sign,
    case when since_trade <= 300 then cs0 - coalesce(cs5, 0) else 0 end flow_5m,
    case when since_trade <= 900 then cs0 - coalesce(cs15, 0) else 0 end flow_15m,
    case when since_trade <= 3600 then cs0 - coalesce(cs60, 0) else 0 end flow_60m,
    ln(1 + case when since_trade <= 900 then cu0 - coalesce(cu15, 0) else 0 end) usd_15m,
    ln(1 + case when since_trade <= 3600 then cu0 - coalesce(cu60, 0) else 0 end) usd_60m,
    case when since_trade <= 900 then cn0 - coalesce(cn15, 0) else 0 end ntr_15m,
    (case when since_trade <= 900 then cs0 - coalesce(cs15, 0) else 0 end) / ((case when since_trade <= 900 then cu0 - coalesce(cu15, 0) else 0 end) + 1.0) flow_ratio_15m,
    case when since_trade <= 3600 then (last_trade_price - mid) / tick end last_trade_minus_mid_ticks,
    since_trade since_trade_s
  from (select *, case when mid < 0.04 or mid > 0.96 then 0.001 else 0.01 end tick from allrows)""")
for d in DAYS:
    path = OUT / f"day=2026-03-{d:02d}.parquet"
    con.sql(f"""copy (select market_id, ts, t_last_move, {', '.join(f'{c}::float {c}' for c in STORED)}
                      from fin where day(make_timestamp(ts // 1000)) = {d} order by market_id, ts)
                to '{path}' (format parquet, compression zstd)""")
    log(f"wrote {path.name}: {con.sql(f'select count(*) from read_parquet({str(path)!r})').fetchone()[0]:,} rows")

# ── 5. robust location / scale of every MODEL feature on the rows of 6–12 March (sample 1 in 20) ─
if args.meta_from is not None:
    ref = json.loads((args.meta_from / "meta.json").read_text())
    ref["days"] = DAYS; ref["meta_from"] = str(args.meta_from); ref["history_from"] = FIRST
    (OUT / "meta.json").write_text(json.dumps(ref, indent=1))
    log(f"meta.json copied from {args.meta_from} (statistics of days {ref.get('stat_days')}); ALL DONE")
    sys.exit(0)
sample = con.sql(f"""select * from fin where day(make_timestamp(ts // 1000)) between {STAT_DAYS[0]} and {STAT_DAYS[-1]}
                     and hash(market_id, ts) % 20 = 0""").df()
sample = sample.merge(mk[["market_id", "start_ns", "end_ns", "duration_h"]], on="market_id", how="left")
# the index derives these at query time from (t, row); here with t = ts
derived = {
    "log_since_move": np.log1p(np.where(sample.t_last_move.isna(), 14 * 86400.0, (sample.ts - sample.t_last_move.fillna(0)) / 1e9)),
    "log_since_trade": np.log1p(sample.since_trade_s.fillna(30 * 86400.0).clip(lower=0)),
    "log_hours_to_end": np.log1p(((sample.end_ns - sample.ts) / 3.6e12).clip(lower=0).fillna(0)),
    "log_hours_since_start": np.log1p(((sample.ts - sample.start_ns) / 3.6e12).clip(lower=0).fillna(0)),
    "log_duration_h": np.log1p(sample.duration_h.clip(lower=0).fillna(0)),
}
stats = {}
num_cols = [c for c in STORED if c != "since_trade_s"]
for c in num_cols:
    x = sample[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
    q25, q50, q75 = np.percentile(x, [25, 50, 75]) if len(x) else (0, 0, 0)
    scale = (q75 - q25) / 1.349 if q75 > q25 else (x.std() if len(x) and x.std() > 0 else 1.0)
    stats[c] = {"loc": float(q50), "scale": float(scale), "finite": float(len(x) / len(sample))}
for c, x in derived.items():
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    scale = (q75 - q25) / 1.349 if q75 > q25 else (x.std() if x.std() > 0 else 1.0)
    stats[c] = {"loc": float(q50), "scale": float(scale), "finite": 1.0}
meta = {"stored": STORED, "derived": list(derived), "categories": CAT_NAMES, "stat_days": STAT_DAYS, "days": DAYS, "history_from": FIRST,
        "stats": stats, "never_moved_seconds": 14 * 86400.0, "never_traded_seconds": 30 * 86400.0}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
log(f"meta written ({len(sample):,} rows in the statistics); ALL DONE")
