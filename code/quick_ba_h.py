"""quick_ba_h.py H LABEL ... — balanced accuracy of saved validation predictions at horizon label H (1h, 60s, 900s, 6h ...)."""
import sys, numpy as np, pandas as pd
H = sys.argv[1]
for lab in sys.argv[2:]:
    df = pd.read_parquet(f"val_predictions_{H}_{lab}.parquet", columns=["y_true", "y_pred", "dedup", "p_down", "p_const", "p_up"])
    yt = df.y_true.values; yp = df.y_pred.values; dd = df.dedup.values.astype(bool)
    def ba(m):
        return np.mean([(yp[m][yt[m] == k] == k).mean() for k in range(3)])
    mix = np.bincount(yp, minlength=3) / len(yp)
    print(f"{lab}: n={len(df):,} BA all={ba(np.ones(len(df), bool)):.4f} dedup={ba(dd):.4f} pred mix={np.round(mix, 3).tolist()} "
          f"label mix={np.round(np.bincount(yt, minlength=3) / len(yt), 3).tolist()} mean p={np.round(df[['p_down','p_const','p_up']].mean().values, 3).tolist()}")
