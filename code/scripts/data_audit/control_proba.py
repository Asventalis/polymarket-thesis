"""control_proba.py — the control's LightGBM G5 probabilities at a short horizon (proba_<H>_G5.npy).

feature_audit.py saves only the hard predictions (pred_<H>_G5.npy); the same-calibration comparison (offset_fair.py)
needs the class probabilities.  Refit G5 with the control's protocol and hyper-parameters (fit on 7–11 March, early
stopping on 12 March, seed 0) on train_features_<H>.parquet, predict val_features_<H>.parquet, save the probabilities
and report the agreement of their argmax with the saved control predictions.

    ./run_dev.sh scripts/data_audit/control_proba.py --horizon 1m
"""
import argparse, time
from pathlib import Path

import numpy as np, pandas as pd, lightgbm as lgb

AUDIT = Path("/data/v.pampinella/data_audit")
ap = argparse.ArgumentParser()
ap.add_argument("--horizon", default="1m"); ap.add_argument("--threads", type=int, default=100)
ap.add_argument("--audit-dir", type=Path, default=AUDIT, help="dir of the control tables (feature_audit.py argv[5])")
a = ap.parse_args(); AUDIT = a.audit_dir
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
sfx = "" if a.horizon == "1h" and (AUDIT / "val_features.parquet").exists() else f"_{a.horizon}"   # 1h: unsuffixed in the older audits
tr = pd.read_parquet(AUDIT / f"train_features{sfx}.parquet", columns=["t", "y"] + FEATS)
val = pd.read_parquet(AUDIT / f"val_features{sfx}.parquet", columns=["y"] + FEATS)
for X in (tr, val):
    X["category"] = pd.Categorical(X.category.astype(str), categories=CAT_NAMES)
tr_days = pd.to_datetime(tr.t).dt.day.values; fit_m = np.isin(tr_days, [7, 8, 9, 10, 11]); es_m = tr_days == 12
ytr = tr.y.values.astype(int); yv = val.y.values.astype(int)
w = 1.0 / np.bincount(ytr, minlength=3)[ytr]; w = w / w.mean()
log(f"train {tr.shape} (fit {fit_m.sum():,}, es {es_m.sum():,}), val {val.shape}")
params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=0)
ds = lgb.Dataset(tr.loc[fit_m, FEATS], ytr[fit_m], weight=w[fit_m], categorical_feature=["category"], free_raw_data=False)
es = lgb.Dataset(tr.loc[es_m, FEATS], ytr[es_m], weight=w[es_m], reference=ds)
bst = lgb.train(params, ds, num_boost_round=600, valid_sets=[es], callbacks=[lgb.early_stopping(40, verbose=False)])
P = bst.predict(val[FEATS], num_iteration=bst.best_iteration).astype(np.float32)
def ba(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    return np.nanmean(np.diag(cm) / cm.sum(1))
saved = np.load(AUDIT / f"pred{sfx}_G5.npy")
p = P.argmax(1)
np.save(AUDIT / f"proba_{a.horizon}_G5.npy", P)
log(f"{bst.best_iteration} rounds; BA refit {ba(yv, p):.4f}, saved control {ba(yv, saved):.4f}, agreement {np.mean(p == saved):.4f}; wrote proba_{a.horizon}_G5.npy")
log("ALL DONE")
