"""news_audit.py — do GDELT news-volume features add information to the control's LightGBM?

Control protocol (learner_audit.py): the control's 61 features (G5) and samples, fit on 7–11 March, early
stopping on 12 March, evaluation on the 9,864,125 validation samples of 13 March; balanced accuracy with 95%
hour-block bootstrap intervals.  Two LightGBMs with the control's hyper-parameters and seed: G5 and G5 + the 12
news features of news_features.py (rows stage on the control's train / val rows).  Δ per subset (all, dedup,
extreme, central), per category and on the rows whose market has news terms; gain importance of every feature.

    ./run_dev.sh scripts/data_audit/news_audit.py --news /data/v.pampinella/data_audit/news_rows.parquet
"""
import argparse, json, time
from pathlib import Path

import numpy as np, pandas as pd, lightgbm as lgb

AUDIT = Path("/data/v.pampinella/data_audit")
HOUR_NS = 3_600_000_000_000
ap = argparse.ArgumentParser()
ap.add_argument("--news", type=Path, default=AUDIT / "news_rows.parquet")
ap.add_argument("--out", type=Path, default=AUDIT / "news")
ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--threads", type=int, default=100)
ap.add_argument("--tag", default="")
ap.add_argument("--prefix", default="news_", help="prefix of the extra feature columns in --news")
ap.add_argument("--title", default="News-volume features")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

GROUPS = {   # feature_audit.py, G5 = all 61
  "G1": ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "sgn_15m", "sgn_1h", "sgn_3h", "sgn_6h", "sgn_12h", "hour"],
  "G2": ["spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10", "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1"],
  "G3": ["ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev", "since_row", "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "since_move", "last_move_sign"],
  "G4": ["flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "log_since_trade"],
  "G5": ["category", "is_yesno", "log_hours_to_end", "log_duration_h", "log_hours_since_start"],
}
FEATS = [f for g in GROUPS.values() for f in g]
CAT_NAMES = ["esports", "sports", "crypto", "weather", "politics", "economy_tech", "culture", "other"]

news = pd.read_parquet(a.news)
NEWS = [c for c in news.columns if c.startswith(a.prefix)]
HAS = [c for c in NEWS if c.endswith("has_terms") or c.endswith("has_sib")][0]
N24 = [c for c in NEWS if c.endswith("log_n_24h") or c.endswith("log_n_sib")][0]
news["t"] = pd.to_datetime(news.t)
tr = pd.read_parquet(AUDIT / "train_features.parquet", columns=["market_id", "t", "y"] + FEATS)
val = pd.read_parquet(AUDIT / "val_features.parquet", columns=["market_id", "t", "market_idx", "y", "dedup", "mid_model"] + FEATS)
tr["t"] = pd.to_datetime(tr.t); val["t"] = pd.to_datetime(val.t)
n_tr, n_val = len(tr), len(val)
tr = tr.merge(news.drop_duplicates(["market_id", "t"]), on=["market_id", "t"], how="left")
val = val.merge(news.drop_duplicates(["market_id", "t"]), on=["market_id", "t"], how="left")
assert len(tr) == n_tr and len(val) == n_val
log(f"train {tr.shape}, val {val.shape}; news features {NEWS}; unmatched rows train {tr[NEWS[0]].isna().mean():.4%}, val {val[NEWS[0]].isna().mean():.4%}")
for X in (tr, val):
    X["category"] = pd.Categorical(X.category.astype(str), categories=CAT_NAMES)
    X[NEWS] = X[NEWS].fillna(0.0)
tr_days = tr.t.dt.day.values; fit_m = np.isin(tr_days, [7, 8, 9, 10, 11]); es_m = tr_days == 12
ytr = tr.y.values.astype(int); yv = val.y.values.astype(int)
w = 1.0 / np.bincount(ytr, minlength=3)[ytr]; w = w / w.mean()
tv = val.t.values.astype("datetime64[ns]").astype("int64")
mid_v = val.mid_model.values
subsets = {"all": np.ones(len(val), bool), "dedup": val.dedup.values.astype(bool),
           "extreme": (mid_v < 0.05) | (mid_v > 0.95), "central": (mid_v >= 0.05) & (mid_v <= 0.95),
           "has extra": val[HAS].values > 0, "no extra": val[HAS].values == 0,
           "extra active": val[N24].values > 0}
for c in CAT_NAMES:
    subsets[f"cat {c}"] = (val.category.astype(str) == c).values
log("has extra: train %.3f, val %.3f; val rows with the extra active %.3f" % ((tr[HAS] > 0).mean(), (val[HAS] > 0).mean(), (val[N24] > 0).mean()))

def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.diag(cm) / cm.sum(1))
def hourly_cms(t, y, p):
    h = pd.factorize(t // HOUR_NS, sort=True)[0]
    return np.bincount(h * 9 + y.astype(int) * 3 + p.astype(int), minlength=(h.max() + 1) * 9).reshape(-1, 3, 3).astype(float)
def ba_cm(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(-1), axis=-1)
brng = np.random.default_rng(0)
def delta(t, y, p, ref):
    cp, cr = hourly_cms(t, y, p), hourly_cms(t, y, ref)
    boot = brng.integers(0, len(cp), size=(a.n_boot, len(cp)))
    d = ba_cm(cp[boot].sum(1)) - ba_cm(cr[boot].sum(1)); lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return 100 * (ba_cm(cp.sum(0)) - ba_cm(cr.sum(0))), 100 * lo, 100 * hi

params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=0)
def fit(name, feats):
    f = a.out / f"proba_1h_{name}{a.tag}.npy"
    if f.exists():
        P = np.load(f); info = json.loads(f.with_suffix(".json").read_text()); log(f"{name}: cached ({info['rounds']} rounds)"); return P, info
    ds = lgb.Dataset(tr.loc[fit_m, feats], ytr[fit_m], weight=w[fit_m], categorical_feature=["category"], free_raw_data=False)
    es = lgb.Dataset(tr.loc[es_m, feats], ytr[es_m], weight=w[es_m], reference=ds)
    bst = lgb.train(params, ds, num_boost_round=600, valid_sets=[es], callbacks=[lgb.early_stopping(40, verbose=False)])
    P = bst.predict(val[feats], num_iteration=bst.best_iteration).astype(np.float32)
    gain = dict(zip(feats, bst.feature_importance("gain").tolist())); tot = sum(gain.values())
    info = {"rounds": bst.best_iteration, "gain": {k: v / tot for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}}
    np.save(f, P); f.with_suffix(".json").write_text(json.dumps(info, indent=1))
    log(f"{name}: {bst.best_iteration} rounds, validation BA {ba(yv, P.argmax(1)):.4f}")
    return P, info

saved = np.load(AUDIT / "pred_G5.npy")
P_base, i_base = fit("G5", FEATS)
P_news, i_news = fit("G5_" + a.prefix.rstrip("_"), FEATS + NEWS)
p_base, p_news = P_base.argmax(1), P_news.argmax(1)
rank = {k: i + 1 for i, k in enumerate(i_news["gain"])}
lines = [f"# {a.title} in the control's LightGBM (1h){' — ' + a.tag if a.tag else ''}\n",
         f"Control protocol: fit on 7–11 March ({fit_m.sum():,} rows), early stopping on 12 March, evaluation on the {len(val):,} validation samples of "
         f"13 March; both models with the control's hyper-parameters and seed. Extra features ({len(NEWS)}): {', '.join(NEWS)}. G5 retrained: {i_base['rounds']} rounds, G5 + news: {i_news['rounds']} rounds. "
         f"Rows with the extra features: train {(tr[HAS] > 0).mean():.1%}, validation {(val[HAS] > 0).mean():.1%}. "
         "Balanced accuracy; Δ in points with 95% hour-block bootstrap intervals.\n",
         "| subset | n | saved control | G5 retrained | G5 + news | Δ news − retrained | 95% CI | Δ news − saved | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
for sn, msk in subsets.items():
    if msk.sum() == 0: continue
    d1, l1, h1 = delta(tv[msk], yv[msk], p_news[msk], p_base[msk]); d2, l2, h2 = delta(tv[msk], yv[msk], p_news[msk], saved[msk])
    lines.append(f"| {sn} | {msk.sum():,} | {ba(yv[msk], saved[msk]):.4f} | {ba(yv[msk], p_base[msk]):.4f} | **{ba(yv[msk], p_news[msk]):.4f}** | "
                 f"{d1:+.2f} | [{l1:+.2f}, {h1:+.2f}] | {d2:+.2f} | [{l2:+.2f}, {h2:+.2f}] |")
    log(f"{sn}: saved {ba(yv[msk], saved[msk]):.4f} base {ba(yv[msk], p_base[msk]):.4f} news {ba(yv[msk], p_news[msk]):.4f} Δ {d1:+.2f} [{l1:+.2f}, {h1:+.2f}]")
lines += ["\nAgreement of G5 + news with the retrained G5: %.1f%%; class mix %s vs %s.\n" % (100 * np.mean(p_news == p_base), np.bincount(p_news, minlength=3) / len(p_news), np.bincount(p_base, minlength=3) / len(p_base)),
          "## Gain importance (share of the total gain), G5 + news\n", "| rank | feature | share |", "|---|---|---|"]
for k, v in list(i_news["gain"].items())[:25]:
    lines.append(f"| {rank[k]} | {k} | {v:.3%} |")
lines += ["", "Extra features: " + ", ".join(f"{k} (rank {rank[k]}, {i_news['gain'][k]:.3%})" for k in NEWS), ""]
text = "\n".join(lines) + "\n"
(a.out / f"{a.prefix}audit_1h{a.tag}.md").write_text(text); print(text)
log("ALL DONE")
