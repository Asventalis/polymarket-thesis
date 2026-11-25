"""
Paired comparison of short test runs on the same validation samples (hour-block bootstrap).

    ./run_dev.sh paired_report.py REF KEY [KEY ...]      # e.g.  lob lob_skip lob_skip_glob

reads val_predictions_t_<H>_<KEY>.parquet for H in 1h, 12h and prints the balanced accuracy of
every run and of Mean Reversion (from the past_delta column), and the differences from REF.
"""
import sys

import numpy as np
import pandas as pd

KEYS = sys.argv[1:]
REF = KEYS[0]


def ba_from_cm(cm):
    r = cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / r
    return np.nanmean(rec, axis=-1)


def hourly_cms(df, col):
    h_idx = pd.factorize(df["t"].values.astype("int64") // 3_600_000_000_000, sort=True)[0]
    flat = np.bincount(h_idx * 9 + df["y_true"].values.astype(int) * 3 + df[col].values.astype(int), minlength=(h_idx.max() + 1) * 9)
    return flat.reshape(-1, 3, 3).astype(float)


rng = np.random.default_rng(0)
for H in ("1h", "12h"):
    try:
        d = {k: pd.read_parquet(f"val_predictions_t_{H}_{k}.parquet") for k in KEYS}
    except FileNotFoundError as e:
        print(H, "missing:", e); continue
    base = d[REF][["t", "market_idx", "y_true", "dedup", "mid", "past_delta"]].copy()
    for k in KEYS:
        assert len(d[k]) == len(base) and (d[k]["t"].values == base["t"].values).all() \
            and (d[k]["market_idx"].values == base["market_idx"].values).all()
        base[k] = d[k]["y_pred"].values
    pdl = base["past_delta"].values
    base["MeanRev"] = np.where(np.isnan(pdl) | (pdl == 0), 1, np.where(pdl > 0, 0, 2))
    mid = base["mid"].values
    print(f"\n===== horizon {H}: n={len(base)} =====")
    subsets = {"all samples": base, "dedup": base[base["dedup"]],
               "mid <0.05 or >0.95": base[(mid < 0.05) | (mid > 0.95)], "mid in [0.05, 0.95]": base[(mid >= 0.05) & (mid <= 0.95)]}
    cols = list(KEYS) + ["MeanRev"]
    for name, df in subsets.items():
        cms = {k: hourly_cms(df, k) for k in cols}
        pt = {k: ba_from_cm(cms[k].sum(0)) for k in cols}
        nh = len(cms[REF])
        boot = rng.integers(0, nh, size=(2000, nh))
        bs = {k: ba_from_cm(cms[k][boot].sum(1)) for k in cols}
        print(f"{name:<22} n={len(df):>8}  " + " | ".join(f"{k} {pt[k]:.4f}" for k in cols))
        for k in KEYS[1:]:
            dlt = bs[k] - bs[REF]; lo, hi = np.percentile(dlt, [2.5, 97.5])
            dm = bs[k] - bs["MeanRev"]; lo2, hi2 = np.percentile(dm, [2.5, 97.5])
            print(f"{'':<22} {k} − {REF}: {100 * (pt[k] - pt[REF]):+.2f} pt [{100 * lo:+.2f}, {100 * hi:+.2f}]"
                  f"    {k} − MeanRev: {100 * (pt[k] - pt['MeanRev']):+.2f} pt [{100 * lo2:+.2f}, {100 * hi2:+.2f}]")
