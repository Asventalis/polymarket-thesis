"""tabular_net.py — pure neural tabular learners (no trees anywhere) on the control's features and samples.

Same protocol as learner_audit.py: the control's 61 features (G5) and samples, fit on 7–11 March, early
stopping on 12 March, evaluation on the 9,864,125 validation samples of 13 March, balanced accuracy with
hour-block bootstrap intervals against the saved LightGBM control (proba_1h_G5.npy / pred_G5.npy).

Learners (all shuffled mini-batches, AdamW, cosine decay, class-balanced cross-entropy, inputs = piecewise-linear
encoding of the numerics (Gorishniy, Rubachev & Babenko 2022) + NaN indicators + hour sin/cos + category one-hot):
  mlp   — MLP width × depth, SiLU, dropout; one model per seed, plus the probability average over seeds
  tabm  — TabM-mini (Gorishniy, Kotelnikov & Babenko 2025): k members sharing the backbone, a per-member
          elementwise adapter on the input and per-member heads; prediction = mean of the members

    ./run_dev.sh scripts/data_audit/tabular_net.py --device cuda:3 --variants mlp,tabm --seeds 0,1,2
"""
import argparse, copy, time, json
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn

AUDIT = Path("/data/v.pampinella/data_audit")
HOUR_NS = 3_600_000_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:3"); ap.add_argument("--threads", type=int, default=64)
ap.add_argument("--variants", default="mlp,tabm"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--bins", type=int, default=16); ap.add_argument("--width", type=int, default=512)
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--dropout", type=float, default=0.15)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=8192); ap.add_argument("--k", type=int, default=16, help="TabM members")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--tag", default="", help="suffix of the output files (hyper-parameter sets)")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--audit-dir", type=Path, default=AUDIT, help="dir of the control tables (feature_audit.py argv[5])")
ap.add_argument("--horizon", default="1h", help="control horizon label (1h: unsuffixed files; 1m/15m: *_<H> files, proba_<H>_G5.npy of control_proba.py)")
a = ap.parse_args()
AUDIT = a.audit_dir; HZ = a.horizon; SFX = "" if HZ == "1h" and (a.audit_dir / "val_features.parquet").exists() else f"_{HZ}"   # 1h: unsuffixed in the older audits
if a.out is None: a.out = AUDIT / "tabular_net"
a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
cfg = f"w{a.width}_d{a.depth}_do{a.dropout}_lr{a.lr}_b{a.batch}_bins{a.bins}" + (f"_{a.tag}" if a.tag else "")

GROUPS = {   # feature_audit.py, G5 = all 61
  "G1": ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "sgn_15m", "sgn_1h", "sgn_3h", "sgn_6h", "sgn_12h", "hour"],
  "G2": ["spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10", "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1"],
  "G3": ["ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev", "since_row", "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "since_move", "last_move_sign"],
  "G4": ["flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "log_since_trade"],
  "G5": ["category", "is_yesno", "log_hours_to_end", "log_duration_h", "log_hours_since_start"],
}
FEATS = [f for g in GROUPS.values() for f in g]
NUMERIC = [f for f in FEATS if f not in ("category", "hour")]
CAT_NAMES = ["esports", "sports", "crypto", "weather", "politics", "economy_tech", "culture", "other"]

# ── data (as learner_audit.py) ────────────────────────────────────────────────────────────────
tr = pd.read_parquet(AUDIT / f"train_features{SFX}.parquet", columns=["t", "y"] + FEATS)
val = pd.read_parquet(AUDIT / f"val_features{SFX}.parquet", columns=["t", "market_idx", "y", "dedup", "mid_model", "past_delta"] + FEATS)
log(f"train {tr.shape}, val {val.shape}")
tr_days = pd.to_datetime(tr.t).dt.day.values; fit_m = np.isin(tr_days, [7, 8, 9, 10, 11]); es_m = tr_days == 12
ytr = tr.y.values.astype(int); yv = val.y.values.astype(int)
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

saved = np.load(AUDIT / f"pred{SFX}_G5.npy"); P_lgb = np.load(AUDIT / f"proba_{HZ}_G5.npy")
log(f"LightGBM control: BA {ba(yv, saved):.4f} (argmax of the cached probabilities {ba(yv, P_lgb.argmax(1)):.4f})")

# ── encoding (as learner_audit.py, PLE computed on the device) ────────────────────────────────
def num(df):
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan) for c in NUMERIC]).astype(np.float32)
Rtr = num(tr); Rv = num(val)
sub = np.flatnonzero(fit_m)[::5]
S = Rtr[sub].astype(np.float64)
loc = np.nanmedian(S, 0); q75, q25 = np.nanpercentile(S, [75, 25], 0); scale = (q75 - q25) / 1.349
sd = np.nanstd(S, 0); scale = np.where(scale > 0, scale, sd); scale = np.where(scale > 0, scale, 1.0)
loc = loc.astype(np.float32); scale = scale.astype(np.float32)
has_nan = np.flatnonzero(np.isnan(Rtr[fit_m]).any(0) | np.isnan(Rv).any(0))
def extras(df, R):
    z = np.clip(np.nan_to_num((R - loc) / scale, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
    ind = np.isnan(R[:, has_nan]).astype(np.float32)
    h = df.hour.values.astype(np.float32) / 24.0
    hs = np.stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h)], 1).astype(np.float32)
    oh = np.eye(len(CAT_NAMES), dtype=np.float32)[df.category.cat.codes.values]
    return z, np.concatenate([ind, hs, oh], 1)
Ztr, Etr = extras(tr, Rtr); Zv, Ev = extras(val, Rv)
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
lo_t, wd_t, keep_t = torch.tensor(lo, device=dev), torch.tensor(wd, device=dev), torch.tensor(keep.ravel(), device=dev)
n_ple = int(keep.sum()); IN = n_ple + len(raw_cols) + Etr.shape[1]
log(f"inputs: PLE {len(ple_cols)} × ≤{Tmax} bins = {n_ple} + {len(raw_cols)} standardised + {Etr.shape[1]} indicators / hour / category = {IN}")
Ptr = np.ascontiguousarray(Rtr[:, ple_cols]); Pv = np.ascontiguousarray(Rv[:, ple_cols])
Ztr = np.ascontiguousarray(Ztr[:, raw_cols]); Zv = np.ascontiguousarray(Zv[:, raw_cols])

def make(idx, which):
    P, Z, E = (Ptr, Ztr, Etr) if which == "tr" else (Pv, Zv, Ev)
    x = torch.from_numpy(P[idx]).to(dev, non_blocking=True)
    e = torch.nan_to_num(((x.unsqueeze(-1) - lo_t) / wd_t).clamp(0, 1), nan=0.0).flatten(1)[:, keep_t]
    return torch.cat([e, torch.from_numpy(Z[idx]).to(dev), torch.from_numpy(E[idx]).to(dev)], 1)

# ── models ────────────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_in, width, depth, dropout):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]; d = width
        self.body = nn.Sequential(*layers); self.head = nn.Linear(d, 3)
    def forward(self, x):            # (B, d_in) → (B, 1, 3)
        return self.head(self.body(x)).unsqueeze(1)

class TabMMini(nn.Module):
    """k members: shared backbone, per-member ±1 / random adapter on the input, per-member heads."""
    def __init__(self, d_in, width, depth, dropout, k):
        super().__init__()
        self.k = k
        self.adapter = nn.Parameter(torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0))
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]; d = width
        self.body = nn.Sequential(*layers)
        self.head_w = nn.Parameter(torch.randn(k, d, 3) / d ** 0.5); self.head_b = nn.Parameter(torch.zeros(k, 3))
    def forward(self, x):            # (B, d_in) → (B, k, 3)
        h = self.body(x.unsqueeze(1) * self.adapter)
        return torch.einsum("bkw,kwc->bkc", h, self.head_w) + self.head_b

def predict(net, idx, which, batch=65536):
    net.eval(); out = []; batch = batch // getattr(net, "k", 1)   # TabM activations are k times larger
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            out.append(torch.softmax(net(make(idx[s:s + batch], which)), -1).mean(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def fit(name, build, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build().to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    fit_idx = np.flatnonzero(fit_m); es_idx = np.flatnonzero(es_m)
    steps = a.epochs * ((len(fit_idx) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos", div_factor=10, final_div_factor=100)
    cw = torch.tensor(1.0 / np.bincount(ytr[fit_m], minlength=3), dtype=torch.float32, device=dev); cw = cw / cw.mean()
    y_t = torch.tensor(ytr, dtype=torch.long)
    best = (-1.0, None, 0); hist = []
    log(f"{name} seed {seed}: {n_par:,} parameters, {steps} steps")
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(fit_idx); tot = 0.0; n = 0
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            out = net(make(idx, "tr"))                                    # (B, k, 3)
            y = y_t[idx].to(dev).unsqueeze(1).expand(-1, out.shape[1])
            loss = nn.functional.cross_entropy(out.reshape(-1, 3), y.reshape(-1), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx); n += len(idx)
        ba_es = ba(ytr[es_idx], predict(net, es_idx, "tr").argmax(1)); hist.append(ba_es)
        log(f"{name} seed {seed} epoch {ep + 1}: train loss {tot / n:.4f}, early-stopping BA (12 March) {ba_es:.4f}")
        if ba_es > best[0]:
            best = (ba_es, copy.deepcopy(net.state_dict()), ep + 1)
        elif ep + 1 - best[2] >= a.patience:
            break
    net.load_state_dict(best[1])
    P = predict(net, np.arange(len(val)), "val")
    log(f"{name} seed {seed}: best epoch {best[2]} (ES BA {best[0]:.4f}), validation BA {ba(yv, P.argmax(1)):.4f}")
    return P, {"best_epoch": best[2], "es_ba": best[0], "hist": hist, "params": n_par}

BUILD = {"mlp": lambda: MLP(IN, a.width, a.depth, a.dropout),
         "tabm": lambda: TabMMini(IN, a.width, a.depth, a.dropout, a.k)}
NAMES = {"mlp": f"MLP {a.width}×{a.depth} + PLE", "tabm": f"TabM-mini k={a.k}, {a.width}×{a.depth} + PLE"}
seeds = [int(s) for s in a.seeds.split(",")]
rows = [("Mean Reversion", mr, ""), ("LightGBM G5 (control)", saved, "")]
meta = {}
probs = {}
for v in a.variants.split(","):
    Ps = []
    for seed in seeds:
        f = a.out / f"proba_{HZ}_{v}_{cfg}_s{seed}.npy"; mf = f.with_suffix(".json")
        if f.exists():
            P = np.load(f); info = json.loads(mf.read_text()) if mf.exists() else {}; note = "cached"
        else:
            P, info = fit(NAMES[v], BUILD[v], seed); np.save(f, P); mf.write_text(json.dumps(info))
            note = f"best epoch {info['best_epoch']}"
        Ps.append(P); meta[f"{v}_s{seed}"] = info
        rows.append((f"{NAMES[v]}, seed {seed}", P.argmax(1), note))
    if len(Ps) > 1:
        Pm = np.mean(Ps, 0); rows.append((f"{NAMES[v]}, mean of {len(Ps)} seeds", Pm.argmax(1), ""))
    else:
        Pm = Ps[0]
    probs[v] = Pm; np.save(a.out / f"proba_{HZ}_{v}_{cfg}_mean.npy", Pm)
if len(probs) > 1:
    Pall = np.mean(list(probs.values()), 0); rows.append(("mean of all variants and seeds", Pall.argmax(1), ""))
    probs["all"] = Pall

# cross-fitted class offsets (DOWN, UP) on the log-probabilities, even hours → odd and vice versa
hour_par = (tv // HOUR_NS) % 2
def offsets(P, grid=np.round(np.arange(-0.6, 0.61, 0.1), 1)):
    L = np.log(P + 1e-9); pred = np.empty(len(P), int); chosen = []
    for par in (0, 1):
        f, g = hour_par != par, hour_par == par
        best = max(((d, u) for d in grid for u in grid), key=lambda o: ba(yv[f], (L[f] + np.array([o[0], 0, o[1]])).argmax(1)))
        pred[g] = (L[g] + np.array([best[0], 0, best[1]])).argmax(1); chosen.append((float(best[0]), float(best[1])))
    return pred, chosen
def combine(P1, P2, ws=np.round(np.arange(0.1, 1.0, 0.1), 1)):
    L1, L2 = np.log(P1 + 1e-9), np.log(P2 + 1e-9); pred = np.empty(len(P1), int); chosen = []
    for par in (0, 1):
        f, g = hour_par != par, hour_par == par
        best = max(ws, key=lambda w: ba(yv[f], (w * L1[f] + (1 - w) * L2[f]).argmax(1)))
        pred[g] = (best * L1[g] + (1 - best) * L2[g]).argmax(1); chosen.append(float(best))
    return pred, chosen
pred, ch = offsets(P_lgb); rows.insert(2, ("LightGBM G5, cross-fitted offsets", pred, f"(DOWN, UP) = {ch[0]}, {ch[1]}"))   # the same post-hoc calibration for the control
for v, P in probs.items():
    nm = NAMES.get(v, "all variants")
    pred, ch = offsets(P); rows.append((f"{nm} ensemble, cross-fitted offsets", pred, f"(DOWN, UP) = {ch[0]}, {ch[1]}"))
    pred, ws = combine(P_lgb, P); rows.append((f"LightGBM ⊕ {nm} ensemble, cross-fitted weight", pred, f"w(LightGBM) = {ws[0]}, {ws[1]}"))

lines = [f"# Pure neural tabular learners vs LightGBM (1h) — `{cfg}`\n",
         f"The control's 61 features (G5) and samples: fit on 7–11 March ({fit_m.sum():,} rows), early stopping on 12 March ({es_m.sum():,}, patience {a.patience}), "
         f"evaluation on the {len(val):,} validation samples of 13 March. Balanced accuracy; Δ in points against the saved LightGBM control with 95% hour-block bootstrap intervals. "
         f"Inputs: piecewise-linear encoding ({a.bins} quantile bins) of the numerics + NaN indicators + hour sin / cos + category one-hot ({IN} inputs). "
         f"Training: shuffled mini-batches of {a.batch}, AdamW lr {a.lr} (one-cycle cosine over {a.epochs} epochs), weight decay {a.wd}, dropout {a.dropout}, class-balanced cross-entropy. "
         "TabM-mini = k members sharing the backbone with per-member ±1 input adapters and per-member heads, trained jointly, prediction = mean of the members. "
         "Offsets / ⊕ weights chosen on the even validation hours for the odd ones and vice versa.\n",
         "| learner | all | Δ vs LightGBM | 95% CI | dedup | Δ | 95% CI | extreme | Δ | central | Δ | note |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, p, note in rows:
    cells = []
    for sn in ["all", "dedup", "extreme", "central"]:
        msk = subsets[sn]; d, lo_, hi_ = delta(tv[msk], yv[msk], p[msk], saved[msk])
        cells.append(f"{ba(yv[msk], p[msk]):.4f} | {d:+.2f}" + (f" | [{lo_:+.2f}, {hi_:+.2f}]" if sn in ("all", "dedup") else ""))
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {note} |")
    log(f"{name}: " + " ".join(f"{sn} {ba(yv[subsets[sn]], p[subsets[sn]]):.4f}" for sn in subsets))
text = "\n".join(lines) + "\n"
(a.out / f"tabular_net_{HZ}_{cfg}.md").write_text(text); print(text)
(a.out / f"meta_{cfg}.json").write_text(json.dumps(meta, indent=1))
log("ALL DONE")
