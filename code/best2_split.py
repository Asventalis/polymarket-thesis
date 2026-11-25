"""Balanced accuracy by price level for the full-length runs (validation only)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score as ba

RUNS = sys.argv[1:] or ["lb_hm_sg", "lb5_sg_mr", "lb5_hm_sg_mr"]
for H in ("1h", "12h"):
    d = {k: pd.read_parquet(f"val_predictions_{H}_{k}_ablation_pred_s0.parquet") for k in RUNS}
    base = d[RUNS[0]]
    cols = [c for c in base.columns if c.startswith("y_pred") or c == "pred"]
    pc = cols[0]
    mid = base["mid"].values
    y = base["y_true"].values
    pd_ = base["past_delta"].values
    mr = np.where(pd_ > 0, 0, np.where(pd_ < 0, 2, 1))
    masks = {"all": np.ones(len(y), bool), "extreme": (mid < 0.05) | (mid > 0.95), "central": (mid >= 0.05) & (mid <= 0.95)}
    print(H, "pred col", pc, "| MR class counts", np.bincount(mr))
    for name, m in masks.items():
        row = f"  {name:8s} n={m.sum():8d}  MR {ba(y[m], mr[m]):.4f}"
        for k in RUNS:
            assert (d[k]["y_true"].values == y).all()
            row += f"  {k} {ba(y[m], d[k][pc].values[m]):.4f}"
        print(row)
