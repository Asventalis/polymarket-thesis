"""test_report.py — the final numbers on a held-out period, with the calibration frozen on the validation day.

Every model is scored on the control's sample table of the evaluated period (val_features[_<H>].parquet of a
feature_audit.py run on those days, i.e. the samples of the deep run at that horizon): Mean Reversion from
past_delta, the LightGBM control from its probabilities (proba_<H>_G5.npy of control_proba.py; its hard
predictions pred[_<H>]_G5.npy when the probabilities are absent), the deep runs from their prediction files
merged on (block time, market), the tabular networks from probability files.  For every model with validation
probabilities two logit offsets (DOWN, UP) are chosen ONCE on the whole validation day (grid ±grid-max, step
grid-step, best balanced accuracy on all samples) and applied unchanged to the evaluated period; a model
without validation probabilities is reported raw.  Differences against the calibrated LightGBM (and against
Mean Reversion) in points with 95% hour-block bootstrap intervals on the evaluated period.

    ./run_dev.sh scripts/data_audit/test_report.py --horizon 1h --deep-horizon 1h \\
        --val-audit-dir /data/v.pampinella/data_audit --test-audit-dir /data/v.pampinella/data_audit/test_1h \\
        --val-pred-dirs /data/v.pampinella/agy_project_orient:'' recheck:_recheck --test-pred-dir test_runs --test-suffix _test \\
        --deep 1h_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0 ... \\
        --tab "TabM=/val/proba_1h_tabm_x_mean.npy:/test/proba_1h_tabm_x_mean.npy" --out test_report_1h.md
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--horizon", required=True, help="control horizon label: 1m, 15m, 1h, 6h, 12h")
ap.add_argument("--deep-horizon", default=None, help="horizon prefix of the deep labels' files if different (60s, 900s)")
ap.add_argument("--val-audit-dir", type=Path, required=True, help="control tables of the validation day")
ap.add_argument("--test-audit-dir", type=Path, required=True, help="control tables of the evaluated period")
ap.add_argument("--val-pred-dirs", nargs="*", default=[], metavar="DIR:SUFFIX",
                help="where the deep runs' validation predictions are searched, in order (the first file with probabilities wins)")
ap.add_argument("--test-pred-dir", type=Path, required=True)
ap.add_argument("--test-suffix", default="_test")
ap.add_argument("--deep", nargs="*", default=[], help="full labels of the deep runs (with the horizon prefix)")
ap.add_argument("--tab", nargs="*", default=[], metavar="NAME=VALFILE:TESTFILE", help="probability files of other models")
ap.add_argument("--grid-max", type=float, default=2.0)
ap.add_argument("--grid-step", type=float, default=0.1)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--title", default="Held-out period")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
H = a.horizon
sfx = "" if H == "1h" else f"_{H}"
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)


def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.diag(cm) / cm.sum(1)))


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
    d = ba_cm(cp[boot].sum(1)) - ba_cm(cr[boot].sum(1))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return 100 * (ba_cm(cp.sum(0)) - ba_cm(cr.sum(0))), 100 * lo, 100 * hi


class Period:
    def __init__(self, audit_dir: Path):
        cols = ["t", "market_idx", "y", "dedup", "mid_model", "past_delta"]
        # 1h tables: unsuffixed in the older audits, "_1h" in the newer ones
        self.sfx = sfx if (audit_dir / f"val_features{sfx}.parquet").exists() else f"_{H}"
        self.val = pd.read_parquet(audit_dir / f"val_features{self.sfx}.parquet", columns=cols)
        self.y = self.val.y.values.astype(int)
        self.t = pd.to_datetime(self.val.t).values.astype("datetime64[ns]").astype("int64")
        mid = self.val.mid_model.values
        self.subsets = {"all": np.ones(len(self.val), bool), "dedup": self.val.dedup.values.astype(bool),
                        "extreme": (mid < 0.05) | (mid > 0.95), "central": (mid >= 0.05) & (mid <= 0.95)}
        pdlt = self.val.past_delta.values
        self.mr = np.where(np.isnan(pdlt) | (pdlt == 0), 1, np.where(pdlt > 0, 0, 2))
        self.key = pd.DataFrame({"t_ns": self.t, "market_idx": self.val.market_idx.values})
        self.audit_dir = audit_dir
        self.hours = len(np.unique(self.t // HOUR_NS))

    def lightgbm(self):
        p = self.audit_dir / f"proba_{H}_G5.npy"
        if p.exists():
            return np.load(p), None
        hard = self.audit_dir / f"pred{self.sfx}_G5.npy"
        return None, np.load(hard)

    def deep(self, path: Path):
        df = pd.read_parquet(path).rename(columns={"t": "t_ns"})
        m = self.key.merge(df, on=["t_ns", "market_idx"], how="left")
        assert len(m) == len(self.val)
        missing = m.y_pred.isna().mean()
        hard = m.y_pred.fillna(1).values.astype(int)
        if "p_down" in m and m.p_down.notna().any():
            return m[["p_down", "p_const", "p_up"]].fillna(1 / 3).values.astype(np.float32), hard, missing
        return None, hard, missing


GRID = np.round(np.arange(-a.grid_max, a.grid_max + a.grid_step / 2, a.grid_step), 2)


def choose_offsets(P, y):
    L = np.log(P + 1e-9)
    best = max(((d, u) for d in GRID for u in GRID), key=lambda o: ba(y, (L + np.array([o[0], 0, o[1]])).argmax(1)))
    return float(best[0]), float(best[1])


def apply_offsets(P, off):
    return (np.log(P + 1e-9) + np.array([off[0], 0, off[1]])).argmax(1)


val, test = Period(a.val_audit_dir), Period(a.test_audit_dir)
log(f"validation: {len(val.val):,} samples, {val.hours} hours; evaluated period: {len(test.val):,} samples, {test.hours} hours")

# ── models: name → (validation probabilities or None, validation hard, test probabilities or None, test hard, note) ──
models = {}
Pv, hv = val.lightgbm()
Pt, ht = test.lightgbm()
models["LightGBM G5"] = (Pv, hv if Pv is None else Pv.argmax(1), Pt, ht if Pt is None else Pt.argmax(1), "")
DH = a.deep_horizon or H
for label in a.deep:
    tp = a.test_pred_dir / f"val_predictions_{label}{a.test_suffix}.parquet"
    if not tp.exists():
        log(f"no evaluated-period predictions for {label}: skipped")
        continue
    Pt, ht, miss_t = test.deep(tp)
    Pv, hv, note = None, None, ""
    for spec in a.val_pred_dirs:
        d, _, s = spec.partition(":")
        vp = Path(d) / f"val_predictions_{label}{s}.parquet"
        if vp.exists():
            P, h, miss_v = val.deep(vp)
            if Pv is None or (P is not None):
                Pv, hv, note = P, h, (f"validation from {Path(d).name}{s}" if s else "")
            if P is not None:
                break
    if miss_t > 0:
        note += f" {miss_t:.2%} of the samples without a prediction"
    models[label] = (Pv, hv, Pt, ht, note.strip())
for spec in a.tab:
    name, _, files = spec.partition("=")
    vf, _, tf = files.partition(":")
    Pv, Pt = np.load(vf), np.load(tf)
    models[name] = (Pv, Pv.argmax(1), Pt, Pt.argmax(1), "")
log(f"{len(models)} models")

# ── calibration on the validation day, applied to the evaluated period ────────────────────────
rows = {}
for name, (Pv, hv, Pt, ht, note) in models.items():
    if Pv is not None and Pt is not None:
        off = choose_offsets(Pv, val.y)
        rows[name] = dict(off=off, val_raw=hv, val_cal=apply_offsets(Pv, off), test_raw=ht, test_cal=apply_offsets(Pt, off), note=note)
    else:
        rows[name] = dict(off=None, val_raw=hv, val_cal=hv, test_raw=ht, test_cal=ht, note=(note + " raw (no probabilities)").strip())
    r = rows[name]
    log(f"{name}: offsets {r['off']}; validation {ba(val.y, r['val_raw']) if r['val_raw'] is not None else float('nan'):.4f} → "
        f"{ba(val.y, r['val_cal']) if r['val_cal'] is not None else float('nan'):.4f}; evaluated {ba(test.y, r['test_raw']):.4f} → {ba(test.y, r['test_cal']):.4f}")

ref = rows["LightGBM G5"]["test_cal"]
lines = [f"# {a.title} ({H})\n",
         f"Samples of the evaluated period: {len(test.val):,} ({test.hours} hours); validation day: {len(val.val):,} samples. "
         f"Two logit offsets (DOWN, UP) per model chosen once on the whole validation day (grid ±{a.grid_max}, step {a.grid_step}) "
         f"and applied unchanged; balanced accuracy; Δ in points against the calibrated LightGBM control on the evaluated period, "
         f"95% hour-block bootstrap intervals ({a.n_boot} resamples). Mean Reversion: {ba(test.y, test.mr):.4f} all, "
         f"{ba(test.y[test.subsets['dedup']], test.mr[test.subsets['dedup']]):.4f} dedup (validation {ba(val.y, val.mr):.4f} / "
         f"{ba(val.y[val.subsets['dedup']], val.mr[val.subsets['dedup']]):.4f}).\n",
         "| model | offsets | validation all (raw → cal.) | evaluated all (raw → cal.) | Δ vs LightGBM | 95% CI | Δ vs MR | 95% CI | dedup (cal.) | Δ | 95% CI | extreme (cal.) | central (cal.) | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, r in rows.items():
    off = f"{r['off'][0]:+.1f}, {r['off'][1]:+.1f}" if r["off"] else "—"
    vraw = ba(val.y, r["val_raw"]) if r["val_raw"] is not None else float("nan")
    vcal = ba(val.y, r["val_cal"]) if r["val_cal"] is not None else float("nan")
    cells = [name, off, f"{vraw:.4f} → {vcal:.4f}", f"{ba(test.y, r['test_raw']):.4f} → {ba(test.y, r['test_cal']):.4f}"]
    d, lo, hi = delta(test.t, test.y, r["test_cal"], ref)
    cells += [f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"]
    d, lo, hi = delta(test.t, test.y, r["test_cal"], test.mr)
    cells += [f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"]
    m = test.subsets["dedup"]
    d, lo, hi = delta(test.t[m], test.y[m], r["test_cal"][m], ref[m])
    cells += [f"{ba(test.y[m], r['test_cal'][m]):.4f}", f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"]
    for sn in ("extreme", "central"):
        m = test.subsets[sn]
        cells.append(f"{ba(test.y[m], r['test_cal'][m]):.4f}")
    cells.append(r["note"])
    lines.append("| " + " | ".join(cells) + " |")
# per-day view: does the ranking hold on every evaluated day?
days = pd.to_datetime(test.t).day
lines += ["\n## By day (calibrated, all samples)\n", "| model | " + " | ".join(f"{d} March" for d in sorted(set(days))) + " |",
          "|---|" + "---|" * len(set(days))]
lines.append("| Mean Reversion | " + " | ".join(f"{ba(test.y[days == d], test.mr[days == d]):.4f}" for d in sorted(set(days))) + " |")
for name, r in rows.items():
    lines.append(f"| {name} | " + " | ".join(f"{ba(test.y[days == d], r['test_cal'][days == d]):.4f}" for d in sorted(set(days))) + " |")
text = "\n".join(lines) + "\n"
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(text)
print(text)
log("ALL DONE")
