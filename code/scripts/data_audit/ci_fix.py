import numpy as np, pandas as pd
from pathlib import Path
OUT = Path("/data/v.pampinella/data_audit")
v = pd.read_parquet(OUT / "val_features.parquet", columns=["t", "y", "dedup", "mid_model", "past_delta", "pred_tf64", "pred_gru10"])
tv = pd.to_datetime(v.t).values.astype("datetime64[ns]").astype("int64")
yv = v.y.values.astype(int); mid = v.mid_model.values
mr = np.where(np.isnan(v.past_delta) | (v.past_delta == 0), 1, np.where(v.past_delta > 0, 0, 2))
def hourly_cms(t, y, p):
    h = pd.factorize(t // 3_600_000_000_000, sort=True)[0]
    return np.bincount(h * 9 + y * 3 + p.astype(int), minlength=(h.max() + 1) * 9).reshape(-1, 3, 3).astype(float)
def ba_cm(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(-1), axis=-1)
rng = np.random.default_rng(0)
subsets = {"all": np.ones(len(v), bool), "dedup": v.dedup.values.astype(bool), "extreme": (mid < 0.05) | (mid > 0.95), "central": (mid >= 0.05) & (mid <= 0.95)}
preds = {"MeanRev": mr, "GRU K=10 model": v.pred_gru10.values, "time-first K=64 model": v.pred_tf64.values}
for g in ["G1", "G2", "G3", "G4", "G5"]:
    preds[g] = np.load(OUT / f"pred_{g}.npy")
print("hour blocks:", len(np.unique(tv // 3_600_000_000_000)))
lines = ["| subset | model | bal. acc. | Δ MR (pt) | 95% CI | Δ time-first (pt) | 95% CI |", "|---|---|---|---|---|---|---|"]
for sn, m in subsets.items():
    cms = {k: hourly_cms(tv[m], yv[m], p[m]) for k, p in preds.items()}
    nh = len(cms["MeanRev"]); boot = rng.integers(0, nh, size=(2000, nh))
    bs = {k: ba_cm(c[boot].sum(1)) for k, c in cms.items()}
    pt = {k: ba_cm(c.sum(0)) for k, c in cms.items()}
    for k in preds:
        d1 = bs[k] - bs["MeanRev"]; d2 = bs[k] - bs["time-first K=64 model"]
        lines.append(f"| {sn} | {k} | {pt[k]:.4f} | {100*(pt[k]-pt['MeanRev']):+.2f} | [{100*np.percentile(d1,2.5):+.2f}, {100*np.percentile(d1,97.5):+.2f}] | {100*(pt[k]-pt['time-first K=64 model']):+.2f} | [{100*np.percentile(d2,2.5):+.2f}, {100*np.percentile(d2,97.5):+.2f}] |")
# per-hour consistency of G5 vs time-first
cm_g = hourly_cms(tv, yv, preds["G5"]); cm_t = hourly_cms(tv, yv, preds["time-first K=64 model"])
wins = int((ba_cm(cm_g) > ba_cm(cm_t)).sum())
lines.append(f"\nG5 beats the time-first model in {wins} of {len(cm_g)} validation hours; hourly balanced accuracies G5 {np.round(ba_cm(cm_g),3).tolist()} vs time-first {np.round(ba_cm(cm_t),3).tolist()}")
# prediction mix
for k in ["MeanRev", "time-first K=64 model", "G5"]:
    lines.append(f"{k}: predicted class shares {np.round(np.bincount(preds[k].astype(int), minlength=3)/len(v), 3).tolist()} (labels {np.round(np.bincount(yv, minlength=3)/len(v),3).tolist()})")
(OUT / "ci_fixed_1h.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
