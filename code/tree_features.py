"""
tree_features.py — cross-fitted LightGBM log-probabilities at every tensor row of 6–13 March 2026.

The deep model as the residual of the no-state control: the trees' class log-probabilities are
given to the deep model as three more input columns and (``train_full.py --tree-offset``) added to
its output logits, so that the network starts from the trees' solution and learns what they miss
(the author's "boosting chain" idea in one stage, 17 October 2026).  Post hoc, the argmax of
0.7 log p(LightGBM) + 0.3 log p(deep model) was +0.19 over the control (learner audit of the same day).

Inputs: the hand-feature directory of ``hand_features.py`` (the control's features at every tensor
row), so the trees see what the deep model sees through the hand features: the 47 numeric stored
columns (mid, raw returns, book shape, flow, activity, trades), the five time-derived columns at the
row's own time, the hour, is_yesno and the category.  Labels as the control's: direction of
mid(last row <= ts + H) - mid(row), |delta| <= eps -> CONSTANT, NULL when ts + H is beyond the rows.

Fit rows as the control's: markets of the model's universe (``market_to_idx``) whose last event is at or
after ts + H, and ts + H within 13 March (the deep model's samples obey the same rule; without the
filter the fit rows have 30–38% CONSTANT labels against the control's 50%, and the trees scored 0.5852
on the control's samples against its 0.6015, build of 17 October 16:18).

Cross-fitting by day: the rows of training day d are predicted by a model fitted on the other
training days (6–12 without d); the rows of 13 March by the model fitted on all of 6–12.  No row is
predicted by a model that saw its own day.  Fit rows: one in EVERY (hash) of the labelled rows;
class-balanced weights; the control's hyper-parameters at a fixed number of rounds (the control
early-stopped at 536).

Output (OUT): day=YYYY-MM-DD.parquet (market_id, ts ns, lp_down, lp_const, lp_up float32: natural
log-probabilities), meta.json (settings, features, per-day out-of-fold balanced accuracy, and the
check against the control: balanced accuracy of the 13 March model on the control's own validation
samples and its agreement with the control's saved predictions).  CPU only.

  ./run_dev.sh tree_features.py OUT [--hand-dir D] [--audit-dir D] [--horizon 3600] [--every 10]
                                    [--rounds 536] [--threads 96] [--days 6,...,13] [--train-days 6,...,12]
"""
import argparse, json, time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
DERIVED = ["log_since_move", "log_since_trade", "log_hours_to_end", "log_hours_since_start", "log_duration_h"]
LP_COLS = ["lp_down", "lp_const", "lp_up"]
DOWN, CONST, UP = 0, 1, 2
INT64_MIN = -2**63          # NaT start / end dates were stored as this by hand_features.py

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--hand-dir", type=Path, default=Path("/data/v.pampinella/agy_project/data/derived/hand_features"))
ap.add_argument("--audit-dir", type=Path, default=Path("/data/v.pampinella/data_audit"),
                help="the control's val_features.parquet and pred_G5.npy (the check on 13 March); '' = skip")
ap.add_argument("--horizon", type=int, default=3600)
ap.add_argument("--control-suffix", default="", help="suffix of the control files in --audit-dir, e.g. _1m → val_features_1m.parquet, pred_1m_G5.npy")
ap.add_argument("--const-eps", type=float, default=0.0)
ap.add_argument("--every", type=int, default=10, help="fit on one labelled row in EVERY (hash of market, ts)")
ap.add_argument("--universe", type=Path, default=Path("/data/v.pampinella/agy_project/data/universe_markets.parquet"))
ap.add_argument("--market-idx", type=Path, default=Path("/data/v.pampinella/agy_project/data/derived/id_mappings/market_to_idx.parquet"))
ap.add_argument("--no-row-filter", action="store_true",
                help="fit on every market and row; default: as the control, only markets of the model's universe whose last event "
                     "is at or after ts + H (the deep model's samples obey the same rule), and ts + H within 13 March")
ap.add_argument("--rounds", type=int, default=536)
ap.add_argument("--threads", type=int, default=96)
ap.add_argument("--days", default="6,7,8,9,10,11,12,13", help="days of March 2026 to predict")
ap.add_argument("--train-days", default="6,7,8,9,10,11,12", help="days whose rows fit the models (cross-fitted)")
ap.add_argument("--end-day", type=int, default=13, help="last day of the labels (rows whose ts + H falls after its end are not fitted)")
ap.add_argument("--hand-dirs", default="", help="comma-separated extra directories searched for day files (e.g. the test days); --hand-dir first")
args = ap.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
H = int(args.horizon) * 10**9
DAYS = [int(x) for x in args.days.split(",")]
TRAIN_DAYS = [int(x) for x in args.train_days.split(",")]
t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

meta_hand = json.loads((args.hand_dir / "meta.json").read_text())
NUMERIC = [c for c in meta_hand["stored"] if c != "since_trade_s"]
CATS = list(meta_hand["categories"])
FEATURES = NUMERIC + DERIVED + ["hour", "is_yesno", "category"]
CAT_IDX = [FEATURES.index("category")]
log(f"{len(FEATURES)} features: {FEATURES}")

con = duckdb.connect()
con.sql(f"set threads={min(args.threads, 64)}; set memory_limit='400GB'; set preserve_insertion_order=false")
MARKETS = f"read_parquet('{args.hand_dir / 'markets.parquet'}')"
END_NS = int(pd.Timestamp(f"2026-03-{args.end_day:02d} 23:59:59").value)
HAND_DIRS = [args.hand_dir] + [Path(x) for x in args.hand_dirs.split(",") if x.strip()]
if args.no_row_filter:
    FIT_FILTER = "true"
    con.sql("create table uni as select null::varchar market_id, null::bigint last_ev_ns where false")
else:
    con.sql(f"""create table uni as select m.market_id, epoch_ns(m.last_event_timestamp) last_ev_ns
                from read_parquet('{args.universe}') m join read_parquet('{args.market_idx}') i using (market_id)""")
    FIT_FILTER = f"u.market_id is not null and u.last_ev_ns >= r.ts + {H} and r.ts + {H} <= {END_NS}"
    log(f"row filter as the control: {con.sql('select count(*) from uni').fetchone()[0]:,} markets of the model's universe, last event >= ts + H")


def day_path(d: int) -> Path:
    """The day file, searched in --hand-dir then --hand-dirs (the first that exists; else the first path, for .exists())."""
    cands = [h / f"day=2026-03-{d:02d}.parquet" for h in HAND_DIRS]
    return next((c for c in cands if c.exists()), cands[0])


def load_day(d: int):
    """ids, ts, X (n, F) float32, y (n,) int8 with -1 = no label, fit (n,) bool."""
    src = [f"select market_id, ts, mid from read_parquet('{day_path(d)}')"]
    if day_path(d + 1).exists():
        src.append(f"select market_id, ts, mid from read_parquet('{day_path(d + 1)}')")
    con.sql(f"create or replace table f as {' union all '.join(src)}")
    cov_end = con.sql("select max(ts) from f").fetchone()[0]
    q = f"""
    select r.market_id, r.ts, {', '.join(f'r.{c}' for c in NUMERIC)},
      ln(1 + case when r.t_last_move is null then {14 * 86400.0} else greatest((r.ts - r.t_last_move) / 1e9, 0) end) log_since_move,
      ln(1 + coalesce(greatest(r.since_trade_s, 0), {30 * 86400.0})) log_since_trade,
      ln(1 + greatest(coalesce((nullif(m.end_ns, {INT64_MIN})::double - r.ts) / 3.6e12, 0), 0)) log_hours_to_end,
      ln(1 + greatest(coalesce((r.ts - nullif(m.start_ns, {INT64_MIN})::double) / 3.6e12, 0), 0)) log_hours_since_start,
      ln(1 + greatest(coalesce(m.duration_h, 0), 0)) log_duration_h,
      ((r.ts // {HOUR_NS}) % 24) + (r.ts % {HOUR_NS}) / {HOUR_NS}.0 as "hour",
      coalesce(m.is_yesno, 0)::double is_yesno, coalesce(m.category, {len(CATS) - 1})::int category,
      case when r.ts + {H} <= {cov_end} then f.mid - r.mid end delta,
      (hash(r.market_id, r.ts) % {args.every} = 0 and {FIT_FILTER}) fit
    from read_parquet('{day_path(d)}') r
    left join {MARKETS} m using (market_id)
    left join uni u on r.market_id = u.market_id
    asof left join f on r.market_id = f.market_id and f.ts <= r.ts + {H}
    order by r.market_id, r.ts"""
    df = con.sql(q).df()
    con.sql("drop table f")
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan) for c in FEATURES]).astype(np.float32)
    delta = pd.to_numeric(df["delta"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    y = np.full(len(df), -1, dtype=np.int8)
    ok = ~np.isnan(delta)
    y[ok] = np.where(delta[ok] > args.const_eps, UP, np.where(delta[ok] < -args.const_eps, DOWN, CONST))
    fit = df["fit"].to_numpy(dtype=bool) & ok
    return df["market_id"].to_numpy(), df["ts"].to_numpy(dtype=np.int64), X, y, fit


def balanced_accuracy(y, pred):
    rec = [(pred[y == k] == k).mean() for k in range(3) if (y == k).any()]
    return float(np.mean(rec)) if rec else float("nan")


# ── 1. load every day once ────────────────────────────────────────────────────────────────────
days = {}
for d in sorted(set(DAYS) | set(TRAIN_DAYS)):       # training days are loaded even when not predicted
    ids, ts, X, y, fit = load_day(d)
    days[d] = (ids, ts, X, y, fit)
    shares = np.bincount(y[y >= 0], minlength=3) / max(1, (y >= 0).sum())
    fit_shares = np.bincount(y[fit], minlength=3) / max(1, fit.sum())
    log(f"day {d}: {len(ts):,} rows, {int((y >= 0).sum()):,} labelled, {int(fit.sum()):,} fit rows; class shares all {np.round(shares, 4)}, fit rows {np.round(fit_shares, 4)}")

params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=args.threads, verbose=-1, seed=0)


def fit_model(fit_days):
    Xf = np.concatenate([days[d][2][days[d][4]] for d in fit_days])
    yf = np.concatenate([days[d][3][days[d][4]] for d in fit_days]).astype(np.int32)
    counts = np.bincount(yf, minlength=3).astype(np.float64)
    w = (len(yf) / (3.0 * counts))[yf]
    ds = lgb.Dataset(Xf, yf, weight=w, feature_name=FEATURES, categorical_feature=CAT_IDX, free_raw_data=True)
    t1 = time.time()
    bst = lgb.train(params, ds, num_boost_round=args.rounds)
    log(f"  fitted on days {fit_days}: {len(yf):,} rows, {bst.num_trees()} trees in {time.time() - t1:.0f}s")
    return bst


# ── 2. one model per predicted day, fitted on the other training days ────────────────────────
per_day = {}
fitted = {}          # days outside the training days share the model fitted on all of them
for d in DAYS:
    fit_days = [x for x in TRAIN_DAYS if x != d]
    if not fit_days:
        raise SystemExit(f"no training days left for day {d}")
    if tuple(fit_days) not in fitted:
        log(f"day {d}: fitting on {fit_days}")
        fitted[tuple(fit_days)] = fit_model(fit_days)
    else:
        log(f"day {d}: model fitted on {fit_days} reused")
    bst = fitted[tuple(fit_days)]
    ids, ts, X, y, _ = days[d]
    t1 = time.time()
    p = bst.predict(X, num_threads=args.threads)
    lp = np.log(np.clip(p, 1e-7, 1.0)).astype(np.float32)
    pred = p.argmax(1)
    ok = y >= 0
    ba = balanced_accuracy(y[ok], pred[ok])
    per_day[str(d)] = {"rows": int(len(ts)), "labelled": int(ok.sum()), "fit_days": fit_days,
                       "oof_balanced_accuracy": ba, "pred_mix": np.bincount(pred, minlength=3).tolist()}
    pd.DataFrame({"market_id": ids, "ts": ts, "lp_down": lp[:, 0], "lp_const": lp[:, 1], "lp_up": lp[:, 2]}).to_parquet(
        OUT / f"day=2026-03-{d:02d}.parquet", index=False, compression="zstd")
    log(f"  day {d}: predicted {len(ts):,} rows in {time.time() - t1:.0f}s; out-of-fold balanced accuracy {ba:.4f}")
    if d == max(DAYS):
        imp = dict(zip(FEATURES, bst.feature_importance("gain").round(1).tolist()))
        per_day[str(d)]["gain_top15"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])

# ── 3. check against the control on 13 March ──────────────────────────────────────────────────
check = {}
SFX = args.control_suffix
if str(args.audit_dir) and 13 in DAYS and (args.audit_dir / f"val_features{SFX}.parquet").exists():
    val = pd.read_parquet(args.audit_dir / f"val_features{SFX}.parquet", columns=["market_id", "t", "y", "dedup"])
    val["rn"] = np.arange(len(val))
    val["t_ns"] = val["t"].values.astype("datetime64[ns]").astype(np.int64)
    pred_g5 = np.load(args.audit_dir / f"pred{SFX}_G5.npy") if (args.audit_dir / f"pred{SFX}_G5.npy").exists() else None
    lp_src = " union all ".join(f"select * from read_parquet('{OUT / f'day=2026-03-{d:02d}.parquet'}')" for d in DAYS if d in (12, 13))
    con.register("val", val)
    j = con.sql(f"""select v.rn, v.y, v.dedup, l.lp_down, l.lp_const, l.lp_up
                    from val v asof left join ({lp_src}) l on v.market_id = l.market_id and l.ts <= v.t_ns""").df().sort_values("rn")
    have = j["lp_down"].notna().to_numpy()
    pred = j[LP_COLS].to_numpy().argmax(1)
    yv = j["y"].to_numpy().astype(int); dd = j["dedup"].to_numpy(dtype=bool)
    check = {"control_samples": int(len(j)), "with_tree_row": int(have.sum()),
             "balanced_accuracy_all": balanced_accuracy(yv[have], pred[have]),
             "balanced_accuracy_dedup": balanced_accuracy(yv[have & dd], pred[have & dd])}
    if pred_g5 is not None:
        g5 = pred_g5[j["rn"].to_numpy()]
        check["control_balanced_accuracy_all"] = balanced_accuracy(yv, g5)
        check["agreement_with_control"] = float((pred[have] == g5[have]).mean())
    log(f"check vs the control: {check}")

meta = {"horizon_s": args.horizon, "const_eps": args.const_eps, "every": args.every, "rounds": args.rounds, "row_filter": not args.no_row_filter,
        "params": {k: v for k, v in params.items() if k != "num_threads"}, "features": FEATURES, "columns": LP_COLS,
        "hand_dir": str(args.hand_dir), "days": DAYS, "train_days": TRAIN_DAYS, "per_day": per_day, "check_vs_control": check}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
log("meta written; ALL DONE")
