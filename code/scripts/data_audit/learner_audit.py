"""Same inputs, different learners.

The no-state control's 61 hand-made features (group G5 of feature_audit.py) on the control's own
samples (train rows of 7-11 March, early stopping on 12 March, the validation samples of 13 March)
under
  1. LightGBM, retrained exactly as the control (and its probabilities saved, which the control did
     not keep),
  2. an MLP of the shape of the deep predictor's wide head (256 x 2, SiLU, class-balanced
     cross-entropy, AdamW 1e-3, shuffled mini-batches, early stopping),
  3. the same MLP on piecewise-linear quantile encodings of the numeric features (Gorishniy et al.,
     "On embeddings for numerical features in tabular deep learning", 2022: each feature becomes T
     values, one per quantile bin, so that an MLP can place thresholds where the trees do),
  4. the same MLP in the regime of the deep model: one pass in time order, AdamW 1e-4,
  5. the deep model's probabilities combined with LightGBM's (geometric mean, weight chosen on the
     even validation hours and applied to the odd ones and vice versa).
Balanced accuracy on the control's subsets with hour-block bootstrap intervals against the retrained
LightGBM.  CPU by default.

    ./run_dev.sh scripts/data_audit/learner_audit.py [--deep LABEL ...] [--device cpu] [--epochs 8] [--bins 16]
"""
import argparse, copy, time
from pathlib import Path

import numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn

AUDIT = Path("/data/v.pampinella/data_audit"); ORIENT = Path("/data/v.pampinella/agy_project_orient")
HOUR_NS = 3_600_000_000_000
ap = argparse.ArgumentParser()
ap.add_argument("--deep", nargs="*", default=["lb_hm_sg_lb5_tf64_hand_xp_ablation_pred_s0", "lb_hm_sg_lb5_tf64_hand_xp_bce_ablation_pred_s0"])
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=64)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bins", type=int, default=16)
ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=AUDIT)
ap.add_argument("--learners", default="mlp_std,mlp_ple,mlp_1pass,mlp_1pass_lr1e-3,mlp_1pass_shuffled,mlp_2pass",
                help="MLP variants to report; a variant whose proba_1h_<key>.npy exists in --out is loaded, not retrained")
a = ap.parse_args()
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = torch.device(a.device)

GROUPS = {   # feature_audit.py, G5 = all 61
  "G1": ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "sgn_15m", "sgn_1h", "sgn_3h", "sgn_6h", "sgn_12h", "hour"],
  "G2": ["spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10", "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1"],
  "G3": ["ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev", "since_row", "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "since_move", "last_move_sign"],
  "G4": ["flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "log_since_trade"],
  "G5": ["category", "is_yesno", "log_hours_to_end", "log_duration_h", "log_hours_since_start"],
}
FEATS = [f for g in GROUPS.values() for f in g]
NUMERIC = [f for f in FEATS if f not in ("category", "hour")]      # the hour enters the MLPs as sin / cos
CAT_NAMES = ["esports", "sports", "crypto", "weather", "politics", "economy_tech", "culture", "other"]

# ── data ──────────────────────────────────────────────────────────────────────────────────────
vf = AUDIT / "val_features.parquet"; tf = AUDIT / "train_features.parquet"
tr = pd.read_parquet(tf, columns=["t", "y"] + FEATS); val = pd.read_parquet(vf, columns=["t", "market_idx", "y", "dedup", "mid_model", "past_delta"] + FEATS)
log(f"train {tr.shape}, val {val.shape}")
tr_days = pd.to_datetime(tr.t).dt.day.values; fit_m = np.isin(tr_days, [7, 8, 9, 10, 11]); es_m = tr_days == 12
ytr = tr.y.values.astype(int); yv = val.y.values.astype(int)
w = 1.0 / np.bincount(ytr, minlength=3)[ytr]; w = w / w.mean()
tv = pd.to_datetime(val.t).values.astype("datetime64[ns]").astype("int64")
mid_v = val.mid_model.values
subsets = {"all": np.ones(len(val), bool), "dedup": val.dedup.values.astype(bool), "extreme": (mid_v < 0.05) | (mid_v > 0.95), "central": (mid_v >= 0.05) & (mid_v <= 0.95)}
mr = np.where(np.isnan(val.past_delta) | (val.past_delta == 0), 1, np.where(val.past_delta > 0, 0, 2))
for X in (tr, val):
    X["category"] = pd.Categorical(X.category.astype(str), categories=CAT_NAMES)

def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
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
    d = ba_cm(cp[boot].sum(1)) - ba_cm(cr[boot].sum(1)); lo, hi = np.percentile(d, [2.5, 97.5])
    return 100 * (ba_cm(cp.sum(0)) - ba_cm(cr.sum(0))), 100 * lo, 100 * hi

# ── 1. LightGBM, as the control ───────────────────────────────────────────────────────────────
Xtr = tr[FEATS]; Xv = val[FEATS]
saved = np.load(AUDIT / "pred_G5.npy")
if (a.out / "proba_1h_G5.npy").exists():   # an earlier call of this script (536 rounds; argmax identical to the saved control)
    P_lgb = np.load(a.out / "proba_1h_G5.npy"); lgb_note = "probabilities of an earlier call"
else:
    ds = lgb.Dataset(Xtr[fit_m], ytr[fit_m], weight=w[fit_m], categorical_feature=["category"], free_raw_data=False)
    es = lgb.Dataset(Xtr[es_m], ytr[es_m], weight=w[es_m], reference=ds)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=100, verbose=-1, seed=0)
    bst = lgb.train(params, ds, num_boost_round=600, valid_sets=[es], callbacks=[lgb.early_stopping(40, verbose=False)])
    P_lgb = bst.predict(Xv, num_iteration=bst.best_iteration).astype(np.float32)
    np.save(a.out / "proba_1h_G5.npy", P_lgb); lgb_note = f"{bst.best_iteration} rounds"
p_lgb = P_lgb.argmax(1)
log(f"LightGBM ({lgb_note}): BA {ba(yv, p_lgb):.4f}, agreement with the saved control {np.mean(p_lgb == saved):.4f} (saved BA {ba(yv, saved):.4f})")

# ── 2-4. MLPs ─────────────────────────────────────────────────────────────────────────────────
def num(df):   # some columns are nullable (pd.NA): coerce column by column
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan) for c in NUMERIC]).astype(np.float32)
Rtr = num(tr); Rv = num(val)
sub = np.flatnonzero(fit_m)[::5]
S = Rtr[sub].astype(np.float64)
loc = np.nanmedian(S, 0); q75, q25 = np.nanpercentile(S, [75, 25], 0); scale = (q75 - q25) / 1.349
sd = np.nanstd(S, 0); scale = np.where(scale > 0, scale, sd); scale = np.where(scale > 0, scale, 1.0)
loc = loc.astype(np.float32); scale = scale.astype(np.float32)
has_nan = np.flatnonzero(np.isnan(Rtr[fit_m]).any(0) | np.isnan(Rv).any(0))
def extras(df, R):
    """standardised numerics (clipped ±10, NaN → 0) | NaN indicators | hour sin, cos | category one-hot"""
    z = np.clip(np.nan_to_num((R - loc) / scale, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
    ind = np.isnan(R[:, has_nan]).astype(np.float32)
    h = df.hour.values.astype(np.float32) / 24.0
    hs = np.stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h)], 1).astype(np.float32)
    oh = np.eye(len(CAT_NAMES), dtype=np.float32)[df.category.cat.codes.values]
    return z, np.concatenate([ind, hs, oh], 1)
Ztr, Etr = extras(tr, Rtr); Zv, Ev = extras(val, Rv)
log(f"MLP inputs: {Ztr.shape[1]} standardised numerics + {Etr.shape[1]} indicators / hour / category")

# piecewise-linear encoding: T quantile bins per numeric feature with at least 3 distinct edges
qs = np.nanquantile(S, np.linspace(0, 1, a.bins + 1), axis=0)
edges = []
for f in range(S.shape[1]):
    e = np.unique(qs[:, f][np.isfinite(qs[:, f])]); edges.append(e if len(e) >= 3 else None)
ple_cols = [f for f, e in enumerate(edges) if e is not None]; raw_cols = [f for f, e in enumerate(edges) if e is None]
Tmax = max(len(e) - 1 for e in edges if e is not None)
lo = np.full((len(ple_cols), Tmax), np.inf, np.float32); wd = np.ones((len(ple_cols), Tmax), np.float32); keep = np.zeros((len(ple_cols), Tmax), bool)
for i, f in enumerate(ple_cols):
    e = edges[f]; k = len(e) - 1
    lo[i, :k] = e[:-1]; wd[i, :k] = np.diff(e); keep[i, :k] = True
lo_t, wd_t, keep_t, ple_idx = torch.tensor(lo), torch.tensor(wd), torch.tensor(keep.ravel()), torch.tensor(ple_cols)
n_ple = int(keep.sum())
log(f"PLE: {len(ple_cols)} features × ≤{Tmax} bins = {n_ple} inputs; {len(raw_cols)} features kept standardised")

def make_std(idx, which):
    Z, E = (Ztr, Etr) if which == "tr" else (Zv, Ev)
    return torch.from_numpy(np.concatenate([Z[idx], E[idx]], 1))
def make_ple(idx, which):
    R, Z, E = (Rtr, Ztr, Etr) if which == "tr" else (Rv, Zv, Ev)
    x = torch.from_numpy(R[idx])[:, ple_idx]
    e = torch.nan_to_num(((x.unsqueeze(-1) - lo_t) / wd_t).clamp(0, 1), nan=0.0).flatten(1)[:, keep_t]
    return torch.cat([e, torch.from_numpy(Z[idx][:, raw_cols]), torch.from_numpy(E[idx])], 1)
IN_STD = Ztr.shape[1] + Etr.shape[1]; IN_PLE = n_ple + len(raw_cols) + Etr.shape[1]

def predict(net, make, idx, which, batch=65536):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            out.append(torch.softmax(net(make(idx[s:s + batch], which).to(dev)), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def fit(name, make, in_dim, lr, batch, epochs, shuffle, patience=2):
    """shuffled variants: early stopping on 12 March (best epoch kept); time-ordered variants mimic the
    deep model's regime and keep the state at the end of the last pass"""
    torch.manual_seed(a.seed)
    net = nn.Sequential(nn.Linear(in_dim, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    cw = torch.tensor(1.0 / np.bincount(ytr[fit_m], minlength=3), dtype=torch.float32, device=dev); cw = cw / cw.mean()
    fit_idx = np.flatnonzero(fit_m); es_idx = np.flatnonzero(es_m); y_t = torch.tensor(ytr, dtype=torch.long)
    if not shuffle:
        fit_idx = fit_idx[np.argsort(tr.t.values[fit_idx], kind="stable")]
    best = (-1.0, None, 0)
    for ep in range(epochs):
        net.train(); order = rng.permutation(fit_idx) if shuffle else fit_idx; tot = 0.0; n = 0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            loss = nn.functional.cross_entropy(net(make(idx, "tr").to(dev)), y_t[idx].to(dev), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx); n += len(idx)
        ba_es = ba(ytr[es_idx], predict(net, make, es_idx, "tr").argmax(1))
        log(f"{name} epoch {ep + 1}: train loss {tot / n:.4f}, early-stopping BA (12 March) {ba_es:.4f}")
        if not shuffle or ba_es > best[0]:
            best = (ba_es, copy.deepcopy(net.state_dict()), ep + 1)
        elif ep + 1 - best[2] >= patience:
            break
    net.load_state_dict(best[1])
    P = predict(net, make, np.arange(len(val)), "val")
    log(f"{name}: {'best' if shuffle else 'last'} epoch {best[2]}, validation BA {ba(yv, P.argmax(1)):.4f}")
    return P, best[2]

SPECS = {   # key: (name, inputs, in_dim, lr, batch, passes, shuffled)
    "mlp_std": ("MLP 256×2 (standardised inputs)", make_std, IN_STD, 1e-3, 8192, a.epochs, True),
    "mlp_ple": (f"MLP 256×2 + piecewise-linear encoding ({a.bins} bins)", make_ple, IN_PLE, 1e-3, 8192, a.epochs, True),
    "mlp_1pass": ("MLP 256×2, one pass in time order, AdamW 1e-4", make_std, IN_STD, 1e-4, 1024, 1, False),
    "mlp_1pass_lr1e-3": ("MLP 256×2, one pass in time order, AdamW 1e-3", make_std, IN_STD, 1e-3, 1024, 1, False),
    "mlp_1pass_shuffled": ("MLP 256×2, one pass shuffled, AdamW 1e-4", make_std, IN_STD, 1e-4, 1024, 1, True),
    "mlp_2pass": ("MLP 256×2, two passes in time order, AdamW 1e-4", make_std, IN_STD, 1e-4, 1024, 2, False),
}
learners = {}
for key in a.learners.split(","):
    name, make, in_dim, lr, batch, passes, shuffled = SPECS[key]
    f = a.out / f"proba_1h_{key}.npy"
    if f.exists():
        learners[name] = (np.load(f), "cached"); log(f"{key}: loaded {f.name}, validation BA {ba(yv, learners[name][0].argmax(1)):.4f}")
    else:
        P, ep = fit(key, make, in_dim, lr, batch, passes, shuffled); np.save(f, P)
        learners[name] = (P, f"{'best' if shuffled else 'last'} epoch {ep}")

# ── 5. the deep model's probabilities, alone and combined with LightGBM ───────────────────────
key = pd.DataFrame({"t_ns": tv, "market_idx": val.market_idx.values})
hour_par = (tv // HOUR_NS) % 2
deep = {}
for label in a.deep:
    p = ORIENT / f"val_predictions_1h_{label}.parquet"
    if not p.exists():
        log(f"no predictions for {label}"); continue
    df = pd.read_parquet(p, columns=["t", "market_idx", "y_pred", "p_down", "p_const", "p_up"]).rename(columns={"t": "t_ns"})
    m = key.merge(df, on=["t_ns", "market_idx"], how="left"); assert len(m) == len(val)
    deep[label] = (m.y_pred.fillna(1).astype(int).values, m[["p_down", "p_const", "p_up"]].fillna(1 / 3).values.astype(np.float32), float(m.y_pred.isna().mean()))
def combine(P1, P2, ws=np.round(np.arange(0.1, 1.0, 0.1), 1)):
    """argmax of w log P1 + (1-w) log P2; w fitted on the even hours for the odd ones and vice versa"""
    L1, L2 = np.log(P1 + 1e-9), np.log(P2 + 1e-9); pred = np.empty(len(P1), int); chosen = []
    for par in (0, 1):
        f, g = hour_par != par, hour_par == par
        best = max(ws, key=lambda w: ba(yv[f], (w * L1[f] + (1 - w) * L2[f]).argmax(1)))
        pred[g] = (best * L1[g] + (1 - best) * L2[g]).argmax(1); chosen.append(float(best))
    return pred, chosen

# ── report ────────────────────────────────────────────────────────────────────────────────────
rows = [("Mean Reversion", mr, ""), ("LightGBM G5, saved control predictions", saved, ""),
        ("LightGBM G5, retrained here", p_lgb, lgb_note)]
for name, (P, note) in learners.items():
    rows.append((name, P.argmax(1), note))
for label, (yp, P, miss) in deep.items():
    short = label.replace("lb_hm_sg_lb5_tf64_", "").replace("_ablation_pred_s0", "")
    rows.append((f"deep `{short}` (argmax)", yp, f"{miss:.2%} unmatched → CONST"))
    pred, ws = combine(P_lgb, P); rows.append((f"LightGBM ⊕ deep `{short}`, cross-fitted weight", pred, f"w(LightGBM) = {ws[0]}, {ws[1]}"))
    rows.append((f"LightGBM ⊕ deep `{short}`, w = 0.5", (np.log(P_lgb + 1e-9) + np.log(P + 1e-9)).argmax(1), ""))
for name, (P, _) in learners.items():
    if "piecewise" in name:
        pred, ws = combine(P_lgb, P); rows.append(("LightGBM ⊕ MLP + PLE, cross-fitted weight", pred, f"w(LightGBM) = {ws[0]}, {ws[1]}"))
lines = ["# Same inputs, different learners (1h)\n",
         f"The control's 61 features (G5) and samples: fit on 7–11 March ({fit_m.sum():,} rows), early stopping on 12 March ({es_m.sum():,}), "
         f"evaluation on the {len(val):,} validation samples of 13 March. Balanced accuracy; Δ in points against the retrained LightGBM with 95% hour-block bootstrap intervals. "
         "MLPs: 256 × 2 SiLU, dropout 0.1, class-balanced cross-entropy; inputs = robustly standardised numerics clipped at ±10 (NaN → 0 plus an indicator), hour as sin / cos, category one-hot. "
         "PLE = piecewise-linear encoding of each numeric feature over its training quantile bins (Gorishniy, Rubachev & Babenko 2022). "
         "⊕ = argmax of w · log p(LightGBM) + (1 − w) · log p(other), w chosen on the even validation hours for the odd ones and vice versa.\n",
         "| learner | all | Δ vs LightGBM | 95% CI | dedup | Δ | 95% CI | extreme | Δ | central | Δ | note |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, p, note in rows:
    cells = []
    for sn in ["all", "dedup", "extreme", "central"]:
        msk = subsets[sn]; d, lo_, hi_ = delta(tv[msk], yv[msk], p[msk], p_lgb[msk])
        cells.append(f"{ba(yv[msk], p[msk]):.4f} | {d:+.2f}" + (f" | [{lo_:+.2f}, {hi_:+.2f}]" if sn in ("all", "dedup") else ""))
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {note} |")
    log(f"{name}: " + " ".join(f"{sn} {ba(yv[subsets[sn]], p[subsets[sn]]):.4f}" for sn in subsets))
text = "\n".join(lines) + "\n"
(a.out / "learner_audit_1h.md").write_text(text); print(text)
log("done")
