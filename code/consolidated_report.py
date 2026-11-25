"""
Consolidated comparison of full-length runs on the same validation samples (hour-block bootstrap).

    ./run_dev.sh consolidated_report.py --dirs DIR1 DIR2 --out OUTDIR LABEL [LABEL ...]

LABEL is the run label without the horizon prefix (e.g. ``lb_ablation_pred_s0``); the file
``val_predictions_<H>_<LABEL>.parquet`` is looked up in the given directories for H in
``--horizons``.  Runs are aligned on (t, market_idx); samples missing from any run are dropped.
For every subset (all, deduplicated, extreme and central prices) it prints the balanced accuracy
of every run and of Mean Reversion (from ``past_delta``), the difference of every run from Mean
Reversion and from the previous run in the list, each with a 95% interval from resampling the
validation hours (2000 draws).  A Markdown table per horizon is written to OUTDIR.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ba_from_cm(cm):
    r = cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / r
    return np.nanmean(rec, axis=-1)


def hourly_cms(df, col):
    h_idx = pd.factorize(df["t"].values.astype("int64") // 3_600_000_000_000, sort=True)[0]
    flat = np.bincount(h_idx * 9 + df["y_true"].values.astype(int) * 3 + df[col].values.astype(int),
                       minlength=(h_idx.max() + 1) * 9)
    return flat.reshape(-1, 3, 3).astype(float)


def find(dirs, name):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--dirs", nargs="+", default=["."])
    ap.add_argument("--horizons", nargs="+", default=["1h", "12h"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for H in args.horizons:
        base = None
        for k in args.labels:
            try:
                df = pd.read_parquet(find(args.dirs, f"val_predictions_{H}_{k}.parquet"),
                                     columns=["t", "market_idx", "y_true", "y_pred", "dedup", "mid", "past_delta"])
            except FileNotFoundError as e:
                print(H, "missing:", e); continue
            df = df.rename(columns={"y_pred": k})
            if base is None:
                base = df
            else:
                n0 = len(base)
                base = base.merge(df[["t", "market_idx", k]], on=["t", "market_idx"], how="inner")
                if len(base) != n0 or len(base) != len(df):
                    print(f"  [{H}] {k}: aligned on {len(base)} samples (had {n0} / {len(df)})")
        if base is None:
            continue
        keys = [k for k in args.labels if k in base.columns]
        pdl = base["past_delta"].values
        base["MeanRev"] = np.where(np.isnan(pdl) | (pdl == 0), 1, np.where(pdl > 0, 0, 2))
        mid = base["mid"].values
        subsets = {"all": base, "dedup": base[base["dedup"]],
                   "extreme (mid<0.05 or >0.95)": base[(mid < 0.05) | (mid > 0.95)],
                   "central (mid in [0.05,0.95])": base[(mid >= 0.05) & (mid <= 0.95)]}
        cols = keys + ["MeanRev"]
        lines = [f"# Horizon {H}: n = {len(base):,} validation samples, {len(keys)} runs\n",
                 "Balanced accuracy; `Δ MR` = run − Mean Reversion, `Δ prev` = run − previous row; "
                 "95% intervals from resampling the 24 validation hours.\n"]
        print(f"\n===== horizon {H}: n={len(base)} =====")
        for name, df in subsets.items():
            cms = {k: hourly_cms(df, k) for k in cols}
            pt = {k: ba_from_cm(cms[k].sum(0)) for k in cols}
            nh = len(cms[cols[0]])
            boot = rng.integers(0, nh, size=(args.n_boot, nh))
            bs = {k: ba_from_cm(cms[k][boot].sum(1)) for k in cols}
            print(f"-- {name}  n={len(df)}  MeanRev {pt['MeanRev']:.4f}")
            lines += [f"\n## {name} (n = {len(df):,}; Mean Reversion {pt['MeanRev']:.4f})\n",
                      "| run | bal. acc. | Δ MR (pt) | 95% CI | Δ prev (pt) | 95% CI |", "|---|---|---|---|---|---|"]
            prev = None
            for k in keys:
                dm = bs[k] - bs["MeanRev"]; lo, hi = np.percentile(dm, [2.5, 97.5])
                s = f"{k:<40} {pt[k]:.4f}  ΔMR {100 * (pt[k] - pt['MeanRev']):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
                row = f"| {k} | {pt[k]:.4f} | {100 * (pt[k] - pt['MeanRev']):+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
                if prev is not None:
                    dp = bs[k] - bs[prev]; lo2, hi2 = np.percentile(dp, [2.5, 97.5])
                    s += f"  Δprev {100 * (pt[k] - pt[prev]):+.2f} [{100 * lo2:+.2f}, {100 * hi2:+.2f}]"
                    row += f" {100 * (pt[k] - pt[prev]):+.2f} | [{100 * lo2:+.2f}, {100 * hi2:+.2f}] |"
                else:
                    row += " | |"
                print(s); lines.append(row); prev = k
        if args.out:
            (args.out / f"consolidated_{H}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
