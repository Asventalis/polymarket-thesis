"""
hand_vs_lgb_report.py — the deep model with hand-made features against the no-state LightGBM
control, on the exact validation samples of the control (feature_audit.py, 1h).

  ./run_dev.sh hand_vs_lgb_report.py LABEL [LABEL ...] [--lgb G5] [--prior a,b,c] [--out DIR]

For every run label (val_predictions_1h_<LABEL>.parquet in --dirs) the report gives the balanced
accuracy on the all / dedup / extreme / central subsets of the control's samples, the difference
from the LightGBM prediction and from Mean Reversion with 95% hour-block bootstrap intervals, and,
when the run saved probabilities, the same for the prior-corrected decision
argmax_k p_k / prior_k (the balanced-accuracy-optimal rule for calibrated probabilities; the prior
is the training class share, never the validation one).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

AUDIT = Path("/data/v.pampinella/data_audit")
ap = argparse.ArgumentParser()
ap.add_argument("labels", nargs="+")
ap.add_argument("--dirs", nargs="+", default=["."])
ap.add_argument("--horizon", default="1h")
ap.add_argument("--lgb", default="G5")
ap.add_argument("--prior", default="", help="training class shares DOWN,CONST,UP for the prior correction")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--crossfit", action="store_true", help="also evaluate cross-fitted logit offsets (even/odd validation hours)")
ap.add_argument("--audit-dir", type=Path, default=AUDIT, help="dir of the control tables (val_features_<H>.parquet, pred_<H>_G5.npy)")
a = ap.parse_args(); AUDIT = a.audit_dir

# the first (1h) control run wrote its files without the horizon suffix
vf = AUDIT / f"val_features_{a.horizon}.parquet"; pf = AUDIT / f"pred_{a.horizon}_{a.lgb}.npy"
if not vf.exists(): vf = AUDIT / "val_features.parquet"; pf = AUDIT / f"pred_{a.lgb}.npy"
val = pd.read_parquet(vf, columns=["t", "market_idx", "y", "dedup", "mid_model", "past_delta", "pred_tf64"])
val["t_ns"] = pd.to_datetime(val.t).values.astype("datetime64[ns]").astype("int64")
val["lgb"] = np.load(pf)
val["mr"] = np.where(np.isnan(val.past_delta) | (val.past_delta == 0), 1, np.where(val.past_delta > 0, 0, 2))
key = val[["t_ns", "market_idx"]]

def find(label):
    for d in a.dirs:
        p = Path(d) / f"val_predictions_{a.horizon}_{label}.parquet"
        if p.exists(): return p
    raise SystemExit(f"no predictions for {label}")

def hourly_cms(t, y, p):
    h = pd.factorize(t // 3_600_000_000_000, sort=True)[0]
    return np.bincount(h * 9 + y.astype(int) * 3 + p.astype(int), minlength=(h.max() + 1) * 9).reshape(-1, 3, 3).astype(float)
def ba_cm(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.diagonal(cm, axis1=-2, axis2=-1) / cm.sum(-1), axis=-1)
rng = np.random.default_rng(0)
def delta(t, y, p, ref):
    cp, cr = hourly_cms(t, y, p), hourly_cms(t, y, ref)
    boot = rng.integers(0, len(cp), size=(a.n_boot, len(cp)))
    d = ba_cm(cp[boot].sum(1)) - ba_cm(cr[boot].sum(1)); lo, hi = np.percentile(d, [2.5, 97.5])
    return 100 * (ba_cm(cp.sum(0)) - ba_cm(cr.sum(0))), 100 * lo, 100 * hi

lines = [f"# Deep model with hand-made features vs the no-state LightGBM ({a.lgb}), horizon {a.horizon}\n",
         f"Samples: the {len(val):,} validation samples of the control. Balanced accuracy; Δ in points with 95% hour-block bootstrap intervals.\n"]
prior = np.array([float(x) for x in a.prior.split(",")]) if a.prior else None
for label in a.labels:
    df = pd.read_parquet(find(label))
    m = key.merge(df.rename(columns={"t": "t_ns"}), on=["t_ns", "market_idx"], how="left")
    assert len(m) == len(val), "duplicate keys"
    miss = m.y_pred.isna().mean()
    preds = {"argmax": m.y_pred.fillna(1).astype(int).values}
    if prior is not None and "p_up" in m and m.p_up.notna().any():
        P = m[["p_down", "p_const", "p_up"]].fillna(1 / 3).values
        preds["prior-corrected"] = np.argmax(P / prior, axis=1)
    if a.crossfit and "p_up" in m and m.p_up.notna().any():
        # cross-fitted logit offsets: the two offsets (DOWN, UP relative to CONST) that maximise the balanced
        # accuracy on the even validation hours are applied to the odd hours and vice versa, so every sample is
        # scored with offsets fitted on other hours.  Mimics what a calibration day would give.
        L = np.log(m[["p_down", "p_const", "p_up"]].fillna(1 / 3).values + 1e-9)
        hour = (val.t_ns.values // 3_600_000_000_000) % 2
        grid = np.arange(-2.0, 2.01, 0.1)
        y = val.y.values; pc = np.empty(len(val), dtype=int); fitted = {}
        for half in (0, 1):
            fit = hour == half; best = (-1, 0.0, 0.0)
            for od in grid:
                for ou in grid:
                    q = np.argmax(L[fit] + np.array([od, 0.0, ou]), axis=1)
                    cm = np.bincount(y[fit].astype(int) * 3 + q, minlength=9).reshape(3, 3).astype(float)
                    b = np.nanmean(np.diag(cm) / cm.sum(1))
                    if b > best[0]: best = (b, od, ou)
            fitted[half] = best
            pc[~fit] = np.argmax(L[~fit] + np.array([best[1], 0.0, best[2]]), axis=1)
        preds["cross-fitted offsets"] = pc
        lines.append(f"\nOffsets fitted on even hours (DOWN, UP): {fitted[0][1]:+.1f}, {fitted[0][2]:+.1f} (BA there {fitted[0][0]:.4f}); on odd hours: {fitted[1][1]:+.1f}, {fitted[1][2]:+.1f} ({fitted[1][0]:.4f}).")
    for name, p in preds.items():
        lines += [f"\n## {label} ({name}; {miss:.2%} of the samples without a prediction → CONST)\n",
                  "| subset | n | MR | LightGBM | run | Δ run − LightGBM | 95% CI | Δ run − MR | 95% CI | pred. class mix |", "|---|---|---|---|---|---|---|---|---|---|"]
        mid = val.mid_model.values
        for sn, msk in {"all": np.ones(len(val), bool), "dedup": val.dedup.values.astype(bool),
                        "extreme": (mid < 0.05) | (mid > 0.95), "central": (mid >= 0.05) & (mid <= 0.95)}.items():
            t, y = val.t_ns.values[msk], val.y.values[msk]
            dl = delta(t, y, p[msk], val.lgb.values[msk]); dm = delta(t, y, p[msk], val.mr.values[msk])
            mix = np.bincount(p[msk], minlength=3) / msk.sum()
            lines.append(f"| {sn} | {msk.sum():,} | {ba_cm(hourly_cms(t, y, val.mr.values[msk]).sum(0)):.4f} | {ba_cm(hourly_cms(t, y, val.lgb.values[msk]).sum(0)):.4f} | "
                         f"**{ba_cm(hourly_cms(t, y, p[msk]).sum(0)):.4f}** | {dl[0]:+.2f} | [{dl[1]:+.2f}, {dl[2]:+.2f}] | {dm[0]:+.2f} | [{dm[1]:+.2f}, {dm[2]:+.2f}] | "
                         f"[{mix[0]:.3f}, {mix[1]:.3f}, {mix[2]:.3f}] |")
        agree = np.mean(p == val.lgb.values)
        lines.append(f"\nAgreement with LightGBM: {agree:.1%}; label mix [{np.mean(val.y == 0):.3f}, {np.mean(val.y == 1):.3f}, {np.mean(val.y == 2):.3f}].")
text = "\n".join(lines) + "\n"
print(text)
if a.out:
    a.out.mkdir(parents=True, exist_ok=True); (a.out / f"hand_vs_lgb_{a.horizon}.md").write_text(text)
