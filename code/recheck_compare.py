"""recheck_compare.py — does evaluating a checkpoint (train_full.py --init-from, warm-up days, then the validation
day) reproduce the validation predictions that the original run saved at the end of its training pass?

The checkpoint holds the weights but not the node states, the hierarchical memory or the LOB buffers; the
re-evaluation rebuilds them by streaming the warm-up days without learning.  The comparison, on the samples
common to the two prediction files (same block time and market), reports the agreement of the hard predictions,
the balanced accuracy of each file on its own samples and on the common ones, and, where both files carry class
probabilities, the mean absolute difference of the probabilities.

    ./run_dev.sh recheck_compare.py --label L [--orig-dir agy_project_orient] [--recheck-dir recheck] [--suffix _recheck]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--label", nargs="+", required=True)
ap.add_argument("--orig-dir", type=Path, default=Path("/data/v.pampinella/agy_project_orient"))
ap.add_argument("--recheck-dir", type=Path, default=Path("recheck"))
ap.add_argument("--suffix", default="_recheck")
ap.add_argument("--horizon-prefix", default="", help="unused; labels already carry the horizon")
args = ap.parse_args()


def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.diag(cm) / cm.sum(1)))


for label in args.label:
    a = pd.read_parquet(args.orig_dir / f"val_predictions_{label}.parquet")
    b = pd.read_parquet(args.recheck_dir / f"val_predictions_{label}{args.suffix}.parquet")
    m = a.merge(b, on=["t", "market_idx"], suffixes=("_o", "_r"))
    print(f"\n== {label}")
    print(f"samples: original {len(a):,}, re-evaluation {len(b):,}, common {len(m):,} "
          f"({len(m) / max(1, len(a)):.2%} of the original)")
    same_label = float((m.y_true_o.values == m.y_true_r.values).mean())
    print(f"same label on the common samples: {same_label:.4%}")
    agree = float((m.y_pred_o.values == m.y_pred_r.values).mean())
    print(f"hard prediction agreement: {agree:.4%}")
    print(f"balanced accuracy  original: all {ba(a.y_true.values, a.y_pred.values):.4f} / dedup {ba(a.y_true[a.dedup].values, a.y_pred[a.dedup].values):.4f}"
          f"  | re-evaluation: all {ba(b.y_true.values, b.y_pred.values):.4f} / dedup {ba(b.y_true[b.dedup].values, b.y_pred[b.dedup].values):.4f}")
    print(f"balanced accuracy on the common samples: original {ba(m.y_true_o.values, m.y_pred_o.values):.4f}, "
          f"re-evaluation {ba(m.y_true_o.values, m.y_pred_r.values):.4f}")
    if "p_down_o" in m and "p_down_r" in m and m.p_down_o.notna().any():
        d = np.abs(m[["p_down_o", "p_const_o", "p_up_o"]].values - m[["p_down_r", "p_const_r", "p_up_r"]].values)
        print(f"probabilities: mean |Δp| {d.mean():.4f}, 95th percentile {np.percentile(d.max(1), 95):.4f}, max {d.max():.4f}")
    # by hour of the day: the first hours after the warm-up are the ones where a short warm-up would show
    h = (pd.to_datetime(m.t).dt.hour.values)
    rows = []
    for hh in range(0, 24, 6):
        s = (h >= hh) & (h < hh + 6)
        rows.append(f"hours {hh:02d}-{hh + 5:02d}: agreement {float((m.y_pred_o.values[s] == m.y_pred_r.values[s]).mean()):.4%}, "
                    f"BA orig {ba(m.y_true_o.values[s], m.y_pred_o.values[s]):.4f} re-eval {ba(m.y_true_o.values[s], m.y_pred_r.values[s]):.4f}")
    print("\n".join(rows))
