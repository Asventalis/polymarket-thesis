"""offset_fair.py — the same post-hoc calibration for everybody: two cross-fitted logit offsets (DOWN, UP) applied to
LightGBM, to the deep runs and to the tabular networks, then paired differences against the calibrated LightGBM.

Offsets are chosen on the even validation hours for the odd ones and vice versa (grid ±0.6, step 0.1); balanced
accuracy on the control's 9,864,125 validation samples; Δ in points with 95% hour-block bootstrap intervals.

    ./run_dev.sh scripts/data_audit/offset_fair.py --deep LABEL ... --tab /data/v.pampinella/data_audit/tabular_net/proba_1h_tabm_*_mean.npy
    ./run_dev.sh scripts/data_audit/offset_fair.py --horizon 1m --deep-horizon 60s --deep lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
""" 
import argparse, time
from pathlib import Path

import numpy as np, pandas as pd

AUDIT = Path("/data/v.pampinella/data_audit"); ORIENT = Path("/data/v.pampinella/agy_project_orient")
HOUR_NS = 3_600_000_000_000
ap = argparse.ArgumentParser()
ap.add_argument("--deep", nargs="*", default=["lb_hm_sg_lb5_tf64_hand_xp_ablation_pred_s0", "lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0",
                                             "lb_hm_sg_lb5_tf64_hand_xp_ema0.998_ablation_pred_s0", "lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ablation_pred_s0",
                                             "lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0"])
ap.add_argument("--tab", nargs="*", default=[])
ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--horizon", default="1h", help="control horizon label: 1h (val_features.parquet, proba_1h_G5.npy) or 1m / 15m (val_features_<H>.parquet, proba_<H>_G5.npy, control_proba.py)")
ap.add_argument("--deep-horizon", default=None, help="horizon label in the deep runs' file names (60s, 900s); default = --horizon")
ap.add_argument("--audit-dir", type=Path, default=AUDIT, help="dir of the control tables and proba_<H>_G5.npy")
ap.add_argument("--grid-max", type=float, default=0.6); ap.add_argument("--grid-step", type=float, default=0.1)
a = ap.parse_args(); AUDIT = a.audit_dir
H = a.horizon; DH = a.deep_horizon or H; sfx = "" if H == "1h" else f"_{H}"
if a.out is None: a.out = AUDIT / f"offset_fair_{H}.md"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

val = pd.read_parquet(AUDIT / f"val_features{sfx}.parquet", columns=["t", "market_idx", "y", "dedup", "mid_model"])
yv = val.y.values.astype(int); tv = pd.to_datetime(val.t).values.astype("datetime64[ns]").astype("int64"); mid_v = val.mid_model.values
subsets = {"all": np.ones(len(val), bool), "dedup": val.dedup.values.astype(bool), "extreme": (mid_v < 0.05) | (mid_v > 0.95), "central": (mid_v >= 0.05) & (mid_v <= 0.95)}
hour_par = (tv // HOUR_NS) % 2

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
GRID = np.round(np.arange(-a.grid_max, a.grid_max + a.grid_step / 2, a.grid_step), 2)
def offsets(P, grid=GRID):
    L = np.log(P + 1e-9); pred = np.empty(len(P), int); chosen = []
    for par in (0, 1):
        f, g = hour_par != par, hour_par == par
        best = max(((d, u) for d in grid for u in grid), key=lambda o: ba(yv[f], (L[f] + np.array([o[0], 0, o[1]])).argmax(1)))
        pred[g] = (L[g] + np.array([best[0], 0, best[1]])).argmax(1); chosen.append((float(best[0]), float(best[1])))
    return pred, chosen

models = {"LightGBM G5": np.load(AUDIT / f"proba_{H}_G5.npy")}
key = pd.DataFrame({"t_ns": tv, "market_idx": val.market_idx.values})
for label in a.deep:
    p = ORIENT / f"val_predictions_{DH}_{label}.parquet"
    if not p.exists(): log(f"no predictions for {label}"); continue
    df = pd.read_parquet(p, columns=["t", "market_idx", "p_down", "p_const", "p_up"]).rename(columns={"t": "t_ns"})
    m = key.merge(df, on=["t_ns", "market_idx"], how="left"); assert len(m) == len(val)
    models["deep `" + label.replace("lb_hm_sg_lb5_tf64_", "").replace("_ablation_pred_s0", "") + "`"] = m[["p_down", "p_const", "p_up"]].fillna(1 / 3).values.astype(np.float32)
for f in a.tab:
    models[f"tabular `{Path(f).stem.replace(f'proba_{H}_', '')}`"] = np.load(f)
log(f"{len(models)} models")
preds = {}
for name, P in models.items():
    pred, ch = offsets(P); preds[name] = (pred, ch); log(f"{name}: offsets {ch}, BA all {ba(yv, pred):.4f}")
ref_raw = models["LightGBM G5"].argmax(1); ref_off = preds["LightGBM G5"][0]
lines = [f"# The same calibration for everybody ({H})\n",
         f"Two cross-fitted logit offsets (DOWN, UP) per model (grid ±{a.grid_max}, step {a.grid_step}), chosen on the even validation hours for the odd ones and vice versa; "
         f"the control's {len(val):,} validation samples; balanced accuracy; Δ in points against the calibrated LightGBM, 95% hour-block bootstrap intervals.\n",
         "| model | offsets | all (raw → calibrated) | Δ vs LightGBM calibrated | 95% CI | dedup (raw → cal.) | Δ | 95% CI | extreme (cal.) | Δ | central (cal.) | Δ |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, P in models.items():
    pred, ch = preds[name]; raw = P.argmax(1); cells = []
    for sn in ["all", "dedup", "extreme", "central"]:
        msk = subsets[sn]; d, lo, hi = delta(tv[msk], yv[msk], pred[msk], ref_off[msk])
        v = f"{ba(yv[msk], raw[msk]):.4f} → {ba(yv[msk], pred[msk]):.4f}" if sn in ("all", "dedup") else f"{ba(yv[msk], pred[msk]):.4f}"
        cells.append(f"{v} | {d:+.2f}" + (f" | [{lo:+.2f}, {hi:+.2f}]" if sn in ("all", "dedup") else ""))
    lines.append(f"| {name} | {ch[0]}, {ch[1]} | " + " | ".join(cells) + " |")
text = "\n".join(lines) + "\n"; a.out.write_text(text); print(text); log("ALL DONE")
