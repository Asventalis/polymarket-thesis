"""lag_bootstrap.py (25 Nov 2026, v2: rows aligned by (t, market_idx)) - the cost of the gap between validation and
test: the stateless control fitted on 7-11 March (thesis) against the same control fitted on 6-20 (all) and on
14-20 (recent), on the same test samples; balanced accuracy on all and deduplicated samples; paired differences
with 95% hour-block bootstrap intervals (2000 draws, seed 0)."""
import numpy as np, pandas as pd, sys
A = "/data/v.pampinella/data_audit"
def ba_cm(C):
    d = np.diag(C); s = C.sum(1)
    return np.nanmean(np.where(s > 0, d / np.maximum(s, 1), np.nan))
rng = np.random.default_rng(0)
print("| horizon | subset | n | thesis (7-11) | all days (6-20) | diff | 95% CI | recent (14-20) | diff | 95% CI |")
print("|---|---|---|---|---|---|---|---|---|---|")
for H in ["1h", "15m", "6h", "12h"]:
    frames = {}
    for k, d in [("thesis", "test_" + H), ("all", "lag_" + H + "_all"), ("recent", "lag_" + H + "_recent")]:
        f = pd.read_parquet(A + "/" + d + "/val_features_" + H + ".parquet", columns=["t", "market_idx", "y", "dedup"])
        p = np.load(A + "/" + d + "/pred_" + H + "_G5.npy").astype(np.int8)
        assert len(p) == len(f), (d, len(p), len(f))
        f["p_" + k] = p
        f = f.drop_duplicates(["t", "market_idx"])
        frames[k] = f.set_index(["t", "market_idx"])
    m = frames["thesis"].join(frames["all"][["p_all"]], how="inner").join(frames["recent"][["p_recent"]], how="inner")
    y = m["y"].to_numpy().astype(int); dd = m["dedup"].to_numpy().astype(bool)
    t = m.index.get_level_values(0).to_numpy().astype("datetime64[ns]").astype("int64")
    hours, h = np.unique(t // 3_600_000_000_000, return_inverse=True); nh = len(hours)
    preds = {k: m["p_" + k].to_numpy().astype(int) for k in ["thesis", "all", "recent"]}
    for name, mask in [("all", np.ones(len(y), bool)), ("dedup", dd)]:
        C = {}
        for k, v in preds.items():
            c = np.zeros((nh, 3, 3)); np.add.at(c, (h[mask], y[mask], v[mask]), 1); C[k] = c
        row = [H, name, str(int(mask.sum())), "%.4f" % ba_cm(C["thesis"].sum(0))]
        draws = rng.integers(0, nh, size=(2000, nh))
        for k in ["all", "recent"]:
            base = ba_cm(C[k].sum(0)) - ba_cm(C["thesis"].sum(0))
            diffs = np.array([ba_cm(C[k][d].sum(0)) - ba_cm(C["thesis"][d].sum(0)) for d in draws])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row += ["%.4f" % ba_cm(C[k].sum(0)), "%+.2f" % (100 * base), "[%+.2f, %+.2f]" % (100 * lo, 100 * hi)]
        print("| " + " | ".join(row) + " |"); sys.stdout.flush()
