"""No-state control and feature diagnostics from the clean data.

Builds, for every validation sample of the 1h model (same t, market, label as
val_predictions_1h_lb_hm_sg_lb5_tf64_ablation_pred_s0.parquet) and for a sample of training
rows of 7-12 March, a set of hand-made causal features from (a) the precomputed LOB tensor
rows, (b) the on-chain trades, (c) the static market table; then
  1. univariate diagnostics of every feature on the validation samples,
  2. LightGBM models on cumulative feature groups (train 7-11, early stop 12, evaluate 13),
     with the same subsets and hour-block bootstrap as consolidated_report.py.
Outputs in OUT (markdown + csv).  CPU only.
"""
import sys, time, json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, lightgbm as lgb

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

DATA = Path("/data/v.pampinella/agy_project/data")
ORIENT = Path("/data/v.pampinella/agy_project_orient")
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("/data/v.pampinella/data_audit"); OUT.mkdir(exist_ok=True)   # argv[5]: output dir (e.g. a control on another run's samples)
H = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
HL = {60: "1m", 900: "15m", 3600: "1h", 21600: "6h", 43200: "12h"}[H]
DHL = {60: "60s", 900: "900s"}.get(H, HL)   # horizon label in the deep runs' file names (train_full.py: <s>s below one hour)
PRED = sys.argv[2] if len(sys.argv) > 2 else "lb_hm_sg_lb5_tf64_ablation_pred_s0"
BASE = sys.argv[3] if len(sys.argv) > 3 else "lb_hm_sg_lb5_ablation_pred_s0"
TRAIN_DAYS = [7, 8, 9, 10, 11]; ES_DAY = 12; VAL_DAY = 13
TRAIN_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 10   # keep one train row in N
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

con = duckdb.connect()
con.sql("set threads=110; set memory_limit='600GB'; set preserve_insertion_order=false")

# ── 1. tensor rows 6-13 March → one row per (market, snapshot) with the needed columns ────────
glob = str(DATA / "derived/tensors/year=2026/month=3/day=*/hour=*/lob_features.parquet")
con.sql(f"""create table rows as
  select market_id, timestamp ts,
    features[41] mid, features[42] spread, features[43] best_bid, features[44] best_ask,
    features[45] ofi, features[46] volume, features[48] buy_ratio, features[49] cancel_ratio,
    features[50] mean_bid_depth, features[51] mean_ask_depth, features[52] depth_skew,
    features[11] bid_sz0, features[31] ask_sz0,
    features[11]+features[12]+features[13]+features[14]+features[15] bid_sz5,
    features[31]+features[32]+features[33]+features[34]+features[35] ask_sz5,
    list_sum(features[11:20]) bid_sz10, list_sum(features[31:40]) ask_sz10,
    length(list_filter(features[11:20], x -> x > 0)) bid_levels, length(list_filter(features[31:40], x -> x > 0)) ask_levels,
    features[2] bid_rel1, features[22] ask_rel1
  from read_parquet('{glob}')
  where day(timestamp) between 6 and 13""")
log(f"rows: {con.sql('select count(*) from rows').fetchone()[0]:,}")

con.sql("""create table r1 as
  select *, epoch(ts - lag(ts) over w) dt_prev, mid - lag(mid) over w dmid
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

# ── 2. trades: per market per minute, oriented to the book's token, cumulative sums ────────────
tglob = str(DATA / "derived/partitioned/transactions/year=2026/month=3/day=*/hour=*/*.parquet")
con.sql(f"""create table trades as
  with o as (select market_id, case when outcome_1 = 'Yes' then 1 else -1 end orient from '{DATA}/universe_markets.parquet'),
  t as (select tr.market_id, timestamp tts, date_trunc('minute', timestamp) mn,
          orient * token_direction * (case when side = 'BUY' then 1 else -1 end) * total_usdc signed, total_usdc usd, price,
          orient * token_direction * (case when side = 'BUY' then 1 else -1 end) sgn
        from read_parquet('{tglob}') tr join o using (market_id)
        where role = 'TAKER' and day(timestamp) between 6 and 13),
  m as (select market_id, mn, sum(signed) s, sum(usd) u, count(*) n, arg_max(price, tts) last_price from t group by 1,2)
  select market_id, mn + interval 1 minute ts, last_price,
         sum(s) over w cs, sum(u) over w cu, sum(n) over w cn
  from m window w as (partition by market_id order by mn rows between unbounded preceding and current row)""")
log(f"trade minutes: {con.sql('select count(*) from trades').fetchone()[0]:,}")

# ── 3. samples ────────────────────────────────────────────────────────────────────────────────
con.sql(f"""create table mk as select m.market_id, i.market_idx, tags, question, outcome_1, start_date, end_date, last_event_timestamp,
  epoch(end_date - start_date)/3600 duration_h, outcome_1 = 'Yes' is_yesno
  from '{DATA}/universe_markets.parquet' m join '{DATA}/derived/id_mappings/market_to_idx.parquet' i using (market_id)""")
if PRED != "none":
    pred = ORIENT / f"val_predictions_{DHL}_{PRED}.parquet"; base = ORIENT / f"val_predictions_{DHL}_{BASE}.parquet"
    con.sql(f"""create table val_s as
      select make_timestamp(p.t // 1000) t, mk.market_id, p.market_idx, p.y_true y, p.dedup, p.mid mid_model, p.past_delta, p.delta, p.y_pred pred_tf64, b.y_pred pred_gru10
      from '{pred}' p join '{base}' b using (t, market_idx) join mk using (market_idx)""")
else:   # no deep-model predictions at this horizon: samples = one tensor row in 20 of 13 March, labels like the training ones
    con.sql(f"""create table val_s as
      with r as (select market_id, ts t, mid from feat where day(ts) = {VAL_DAY} and hash(market_id, ts) % 20 = 0),
      r2 as (select r.market_id, mk.market_idx, r.t, r.mid mid_model from r join mk using (market_id)
             where mk.last_event_timestamp >= r.t + interval {H} second and r.t + interval {H} second <= timestamp '2026-03-13 23:59:59'),
      r3 as (select r2.*, f.mid mid_fwd, b.mid mid_back from r2
             asof join pm f on r2.market_id = f.market_id and f.ts <= r2.t + interval {H} second
             asof left join pm b on r2.market_id = b.market_id and b.ts <= r2.t - interval {H} second)
      select market_id, market_idx, t, mid_model, case when mid_fwd > mid_model then 2 when mid_fwd < mid_model then 0 else 1 end y,
             row_number() over (partition by market_id, date_trunc('minute', t) - interval (minute(t) % 10) minute order by t) = 1 dedup,
             mid_model - mid_back past_delta, mid_fwd - mid_model delta,
             case when mid_back is null or mid_model = mid_back then 1 when mid_model > mid_back then 0 else 2 end pred_tf64,
             case when mid_back is null or mid_model = mid_back then 1 when mid_model > mid_back then 0 else 2 end pred_gru10
      from r3""")
# train samples: tensor rows of the train days, one in TRAIN_EVERY, label from the last row <= t+H
con.sql(f"""create table tr_s as
  with r as (select market_id, ts t, mid, hash(market_id, ts) % {TRAIN_EVERY} k from feat where day(ts) between {TRAIN_DAYS[0]} and {ES_DAY}),
  r2 as (select r.market_id, r.t, r.mid mid_model from r join mk using (market_id) where k = 0 and mk.last_event_timestamp >= r.t + interval {H} second and r.t + interval {H} second <= timestamp '2026-03-13 23:59:59')
  select r2.*, f.mid mid_fwd from r2 asof join pm f on r2.market_id = f.market_id and f.ts <= r2.t + interval {H} second""")
con.sql("alter table tr_s add column y tinyint; update tr_s set y = case when mid_fwd > mid_model then 2 when mid_fwd < mid_model then 0 else 1 end")
log(f"val samples {con.sql('select count(*) from val_s').fetchone()[0]:,}, train samples {con.sql('select count(*) from tr_s').fetchone()[0]:,}")

def attach(src, dst):
    con.sql(f"""create table {dst} as
      select s.*, f.* exclude (market_id),
             l1.mid mid_15m, l2.mid mid_1h, l3.mid mid_3h, l4.mid mid_6h, l5.mid mid_12h,
             x0.cs cs0, x0.cu cu0, x0.cn cn0, x0.last_price last_trade_price, epoch(s.t - x0.ts) since_trade,
             x5.cs cs5, x15.cs cs15, x15.cu cu15, x15.cn cn15, x60.cs cs60, x60.cu cu60,
             mk.tags, mk.duration_h, mk.is_yesno, epoch(mk.end_date - s.t)/3600 hours_to_end, epoch(s.t - mk.start_date)/3600 hours_since_start
      from {src} s
      join mk using (market_id)
      asof join feat f on s.market_id = f.market_id and f.ts <= s.t
      asof left join pm l1 on s.market_id = l1.market_id and l1.ts <= s.t - interval 15 minute
      asof left join pm l2 on s.market_id = l2.market_id and l2.ts <= s.t - interval 1 hour
      asof left join pm l3 on s.market_id = l3.market_id and l3.ts <= s.t - interval 3 hour
      asof left join pm l4 on s.market_id = l4.market_id and l4.ts <= s.t - interval 6 hour
      asof left join pm l5 on s.market_id = l5.market_id and l5.ts <= s.t - interval 12 hour
      asof left join trades x0 on s.market_id = x0.market_id and x0.ts <= s.t
      asof left join trades x5 on s.market_id = x5.market_id and x5.ts <= s.t - interval 5 minute
      asof left join trades x15 on s.market_id = x15.market_id and x15.ts <= s.t - interval 15 minute
      asof left join trades x60 on s.market_id = x60.market_id and x60.ts <= s.t - interval 60 minute""")

attach("val_s", "val_f"); log("val features attached")
attach("tr_s", "tr_f"); log("train features attached")

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

def finish(df):
    df = df.copy()
    tick = np.where((df.mid < 0.04) | (df.mid > 0.96), 0.001, 0.01)
    df["mid_consistent"] = (np.abs(df.mid - df.mid_model) < 1e-6)
    df["hour"] = pd.to_datetime(df.t).dt.hour
    df["since_row"] = (pd.to_datetime(df.t) - pd.to_datetime(df.ts)).dt.total_seconds()
    df["since_move"] = (pd.to_datetime(df.t) - pd.to_datetime(df.t_last_move)).dt.total_seconds()
    df["log_dt_prev"] = np.log1p(df.dt_prev)
    for L in ["15m", "1h", "3h", "6h", "12h"]:
        df[f"ret_{L}"] = df.mid - df[f"mid_{L}"]
        df[f"sgn_{L}"] = np.sign(df[f"ret_{L}"])
    df["imb1"] = (df.bid_sz0 - df.ask_sz0) / (df.bid_sz0 + df.ask_sz0 + 1e-6)
    df["imb5"] = (df.bid_sz5 - df.ask_sz5) / (df.bid_sz5 + df.ask_sz5 + 1e-6)
    df["imb10"] = (df.bid_sz10 - df.ask_sz10) / (df.bid_sz10 + df.ask_sz10 + 1e-6)
    micro = (df.best_ask * df.bid_sz0 + df.best_bid * df.ask_sz0) / (df.bid_sz0 + df.ask_sz0 + 1e-6)
    df["micro_minus_mid_ticks"] = (micro - df.mid) / tick
    df["spread_ticks"] = df.spread / tick
    df["log_bid_sz0"] = np.log1p(df.bid_sz0); df["log_ask_sz0"] = np.log1p(df.ask_sz0)
    df["log_bid_sz10"] = np.log1p(df.bid_sz10); df["log_ask_sz10"] = np.log1p(df.ask_sz10)
    df["log_vol_15m"] = np.log1p(df.vol_15m)
    df["range_pos_60m"] = (df.mid - df.min_60m) / (df.max_60m - df.min_60m + 1e-6) - 0.5
    df["mid_minus_mean60"] = df.mid - df.mean_60m
    df["rv_60m"] = np.sqrt(df.rv_60m)
    df["flow_5m"] = (df.cs0 - df.cs5.fillna(0)).where(df.since_trade <= 300, 0.0)
    df["flow_15m"] = (df.cs0 - df.cs15.fillna(0)).where(df.since_trade <= 900, 0.0)
    df["flow_60m"] = (df.cs0 - df.cs60.fillna(0)).where(df.since_trade <= 3600, 0.0)
    df["usd_15m"] = np.log1p((df.cu0 - df.cu15.fillna(0)).where(df.since_trade <= 900, 0.0))
    df["usd_60m"] = np.log1p((df.cu0 - df.cu60.fillna(0)).where(df.since_trade <= 3600, 0.0))
    df["ntr_15m"] = (df.cn0 - df.cn15.fillna(0)).where(df.since_trade <= 900, 0.0)
    df["flow_ratio_15m"] = df.flow_15m / (np.expm1(df.usd_15m) + 1.0)
    df["last_trade_minus_mid_ticks"] = ((df.last_trade_price - df.mid) / tick).where(df.since_trade <= 3600)
    df["log_since_trade"] = np.log1p(df.since_trade)
    df["log_hours_to_end"] = np.log1p(df.hours_to_end.clip(lower=0))
    df["log_duration_h"] = np.log1p(df.duration_h.clip(lower=0))
    df["log_hours_since_start"] = np.log1p(df.hours_since_start.clip(lower=0))
    df["category"] = pd.Categorical(df.tags.map(category), categories=CAT_NAMES)
    df["is_yesno"] = df.is_yesno.astype(float)
    df["ofi_15m_z"] = df.ofi_15m / (df.vol_15m + 1.0)
    df["ofi_20_z"] = df.ofi_20 / (df.volume.abs() * 20 + 1.0)
    return df

GROUPS = {
  "G1 price history (direct-path inputs)": ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "sgn_15m", "sgn_1h", "sgn_3h", "sgn_6h", "sgn_12h", "hour"],
  "G2 + book shape": ["spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10", "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1"],
  "G3 + flow, activity, clock": ["ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev", "since_row", "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "since_move", "last_move_sign"],
  "G4 + on-chain trades": ["flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "log_since_trade"],
  "G5 + static market fields": ["category", "is_yesno", "log_hours_to_end", "log_duration_h", "log_hours_since_start"],
}

val = finish(con.sql("select * from val_f").df()); log(f"val df {val.shape}")
tr = finish(con.sql("select * from tr_f").df()); log(f"train df {tr.shape}")
val.to_parquet(OUT / f"val_features_{HL}.parquet"); tr.to_parquet(OUT / f"train_features_{HL}.parquet")
lines = [f"# No-state control and feature diagnostics ({HL}), built 2026-10-15\n",
         f"Validation samples: {len(val):,} (the {HL} samples of 13 March; mid of the last tensor row equals the model's mid in {val.mid_consistent.mean():.1%} of them). "
         f"Training samples: {len(tr):,} tensor rows of 7–12 March (one in {TRAIN_EVERY}), labelled with the last tensor row at or before t+H; class shares {np.bincount(tr.y, minlength=3)/len(tr)}.\n"]
log(f"mid consistency {val.mid_consistent.mean():.4f}; val class shares {np.bincount(val.y, minlength=3)/len(val)}")

# ── 4. univariate diagnostics on validation ────────────────────────────────────────────────────
def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
    return np.nanmean(np.diag(cm) / cm.sum(1))
yv = val.y.values; moved = yv != 1
rng_u = np.random.default_rng(1)
rows_u = []
allf = [f for g in GROUPS.values() for f in g if f != "category"]
for f in allf:
    x = val[f].values.astype(float); ok = np.isfinite(x)
    sub = rng_u.choice(np.flatnonzero(ok), size=min(1_000_000, ok.sum()), replace=False)
    rho = pd.Series(x[sub]).corr(pd.Series(val.delta.values[sub]), method='spearman')
    # directional hit rate of sign(x) on moved samples; both signs of the feature tried, the better one reported
    m = ok & moved & (x != 0)
    hit = np.mean(np.sign(x[m]) == np.where(yv[m] == 2, 1, -1)) if m.sum() else np.nan
    sgn = 1 if hit >= 0.5 else -1; hit = max(hit, 1 - hit) if m.sum() else np.nan
    p = np.where(x == 0, 1, np.where(sgn * x > 0, 2, 0)); p[~ok] = 1
    # CONST separation: AUC of |x| for y == CONST
    ax = np.abs(x[ok]); yc = (yv[ok] == 1)
    r = pd.Series(ax).rank().values; auc = (r[yc].sum() - yc.sum() * (yc.sum() + 1) / 2) / (yc.sum() * (~yc).sum())
    rows_u.append(dict(feature=f, finite=ok.mean(), nonzero=np.mean(x[ok] != 0), spearman_vs_delta=rho, hit_rate_on_moves=hit, n_moves_nonzero=int(m.sum()), sign=sgn,
                       ba_sign_rule=ba(yv, p), auc_abs_vs_const=max(auc, 1 - auc)))
U = pd.DataFrame(rows_u).sort_values("hit_rate_on_moves", ascending=False)
U.to_csv(OUT / f"univariate_{HL}.csv", index=False)
mr = np.where(np.isnan(val.past_delta) | (val.past_delta == 0), 1, np.where(val.past_delta > 0, 0, 2))
lines += ["\n## Univariate diagnostics on the validation samples\n",
          f"Mean Reversion (past_delta): balanced accuracy {ba(yv, mr):.4f}; deep model `{PRED}` {ba(yv, val.pred_tf64.values):.4f}; `{BASE}` {ba(yv, val.pred_gru10.values):.4f} (at 1 min both columns are Mean Reversion itself).\n",
          "`hit_rate_on_moves` = P(sign of the feature, with the better of the two orientations, equals the direction) among samples that moved and where the feature is non-zero; "
          "`ba_sign_rule` = balanced accuracy of the 3-class rule sign(feature) with zero → CONST; `auc_abs_vs_const` = AUC of |feature| for separating CONST from moves (either direction).\n",
          U.to_markdown(index=False, floatfmt=".4f")]
log("univariate done")

# ── 5. LightGBM on cumulative groups ───────────────────────────────────────────────────────────
def hourly_cms(t, y, p):
    h = pd.factorize(t.astype("int64") // 3_600_000_000_000, sort=True)[0]
    return np.bincount(h * 9 + y.astype(int) * 3 + p.astype(int), minlength=(h.max() + 1) * 9).reshape(-1, 3, 3).astype(float)
def ba_cm(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(-1), axis=-1)
rng = np.random.default_rng(0)
tv = pd.to_datetime(val.t).values.astype("datetime64[ns]").astype("int64")
mid_v = val.mid_model.values
subsets = {"all": np.ones(len(val), bool), "dedup": val.dedup.values.astype(bool), "extreme": (mid_v < 0.05) | (mid_v > 0.95), "central": (mid_v >= 0.05) & (mid_v <= 0.95)}
def evaluate(name, p, ref):
    out = {}
    for sn, msk in subsets.items():
        cm_p = hourly_cms(tv[msk], yv[msk], p[msk]); cm_r = hourly_cms(tv[msk], yv[msk], ref[msk])
        nh = len(cm_p); boot = rng.integers(0, nh, size=(2000, nh))
        d = ba_cm(cm_p[boot].sum(1)) - ba_cm(cm_r[boot].sum(1)); lo, hi = np.percentile(d, [2.5, 97.5])
        out[sn] = (ba_cm(cm_p.sum(0)), 100 * (ba_cm(cm_p.sum(0)) - ba_cm(cm_r.sum(0))), 100 * lo, 100 * hi)
    return out

tr_days = pd.to_datetime(tr.t).dt.day.values
fit_m, es_m = np.isin(tr_days, TRAIN_DAYS), tr_days == ES_DAY
w = 1.0 / np.bincount(tr.y, minlength=3)[tr.y.values]; w = w / w.mean()
results = []; feats = []
lines += ["\n## LightGBM no-state control, cumulative feature groups\n",
          "Trained on 7–11 March (early stopping on 12 March, class-balanced weights, 63 leaves, lr 0.05, ≤ 600 rounds), evaluated on the validation samples. "
          f"Balanced accuracy; Δ = difference from Mean Reversion in points with 95% hour-block bootstrap intervals; last column the difference from the deep model `{PRED}`.\n",
          "| group | #feat | all | Δ MR | 95% CI | dedup | Δ MR | extreme | Δ MR | central | Δ MR | Δ vs tf64 (all) | 95% CI | rounds |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for gname, cols in GROUPS.items():
    feats = feats + cols
    Xtr = tr[feats]; Xv = val[feats]
    ds = lgb.Dataset(Xtr[fit_m], tr.y.values[fit_m], weight=w[fit_m], categorical_feature=["category"] if "category" in feats else "auto", free_raw_data=False)
    es = lgb.Dataset(Xtr[es_m], tr.y.values[es_m], weight=w[es_m], reference=ds)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=100, verbose=-1, seed=0)
    bst = lgb.train(params, ds, num_boost_round=600, valid_sets=[es], callbacks=[lgb.early_stopping(40, verbose=False)])
    p = bst.predict(Xv, num_iteration=bst.best_iteration).argmax(1)
    r = evaluate(gname, p, mr); r2 = evaluate(gname, p, val.pred_tf64.values)
    lines.append(f"| {gname} | {len(feats)} | {r['all'][0]:.4f} | {r['all'][1]:+.2f} | [{r['all'][2]:+.2f}, {r['all'][3]:+.2f}] | {r['dedup'][0]:.4f} | {r['dedup'][1]:+.2f} | {r['extreme'][0]:.4f} | {r['extreme'][1]:+.2f} | {r['central'][0]:.4f} | {r['central'][1]:+.2f} | {r2['all'][1]:+.2f} | [{r2['all'][2]:+.2f}, {r2['all'][3]:+.2f}] | {bst.best_iteration} |")
    imp = pd.Series(bst.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(OUT / f"importance_{HL}_{gname.split()[0]}.csv")
    results.append((gname, p, imp))
    log(f"{gname}: all {r['all'][0]:.4f} (ΔMR {r['all'][1]:+.2f}), dedup {r['dedup'][0]:.4f}, rounds {bst.best_iteration}")
    np.save(OUT / f"pred_{HL}_{gname.split()[0]}.npy", p)
gname, p, imp = results[-1]
lines += ["\n### Gain importance of the full model (top 25, share of total gain)\n", (imp / imp.sum()).head(25).to_frame("share").to_markdown(floatfmt=".3f")]
# agreement / simple combination with the deep model
agree = np.mean(p == val.pred_tf64.values)
vote = np.where(p == val.pred_tf64.values, p, np.where(p == mr, p, np.where(val.pred_tf64.values == mr, mr, val.pred_tf64.values)))
rv = evaluate("vote", vote, val.pred_tf64.values)
lines += [f"\n### Relation to the time-first model\n\nThe full LightGBM agrees with the time-first model on {agree:.1%} of the samples. "
          f"A majority vote of LightGBM, time-first and Mean Reversion (ties → time-first) scores {rv['all'][0]:.4f} on all samples ({rv['all'][1]:+.2f} [{rv['all'][2]:+.2f}, {rv['all'][3]:+.2f}] vs time-first), {rv['dedup'][0]:.4f} dedup, {rv['central'][0]:.4f} central, {rv['extreme'][0]:.4f} extreme.\n"]
# per-category view of the dynamics
cat = val.category.values
tab = []
for c in CAT_NAMES:
    m = cat == c
    if m.sum() < 1000: continue
    tab.append(dict(category=c, samples=int(m.sum()), markets=val.market_id[m].nunique(), p_move=float(moved[m].mean()), p_up_given_move=float((yv[m & moved] == 2).mean()),
                    MR=ba(yv[m], mr[m]), lightgbm=ba(yv[m], p[m]), tf64=ba(yv[m], val.pred_tf64.values[m])))
lines += ["\n### By market category (validation samples; balanced accuracy inside the group)\n", pd.DataFrame(tab).to_markdown(index=False, floatfmt=".4f")]
(OUT / f"feature_audit_{HL}.md").write_text("\n".join(lines) + "\n")
log("ALL DONE")
