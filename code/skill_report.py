"""Paired comparison of the short wallet-skill test runs (same samples, hour-block bootstrap).
References: t_<H>_lob (LOB-only) and t_<H>_full_ob (graph, book-oriented edges) of the orientation test."""
import sys, numpy as np, pandas as pd
from polymarket_gnn.data.graph_builder import load_market_orientation

BASE = "/data/v.pampinella/agy_project/data"
mp = pd.read_parquet(f"{BASE}/derived/id_mappings/market_to_idx.parquet")
ori = load_market_orientation(f"{BASE}/universe_markets.parquet")
grp_of_idx = dict(zip(mp["market_idx"], [ori.get(m, 1) for m in mp["market_id"]]))

def ba_from_cm(cm):
    r = cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / r
    return np.nanmean(rec, axis=-1)

def hourly_cms(df, pred_col):
    hours = (df["t"].values // 3_600_000_000_000)
    h_idx = pd.factorize(hours, sort=True)[0]
    flat = np.bincount(h_idx * 9 + df["y_true"].values.astype(int) * 3 + df[pred_col].values.astype(int), minlength=(h_idx.max() + 1) * 9)
    return flat.reshape(-1, 3, 3).astype(float)

KEYS = ("lob", "full_ob", "full_ob_sk", "full_ob_skp")
rng = np.random.default_rng(0)
for H in ("1h", "12h"):
    try:
        d = {k: pd.read_parquet(f"val_predictions_t_{H}_{k}.parquet") for k in KEYS}
    except FileNotFoundError as e:
        print(H, "missing:", e); continue
    base = d["lob"][["t", "market_idx", "y_true", "dedup"]].copy()
    for k in d:
        assert len(d[k]) == len(base) and (d[k]["t"].values == base["t"].values).all() and (d[k]["market_idx"].values == base["market_idx"].values).all()
        base[k] = d[k]["y_pred"].values
    base["grp"] = base["market_idx"].map(grp_of_idx).fillna(1).astype(int)
    print(f"\n===== horizon {H}: train 11–12 March, validation 13 March, seed 0, n={len(base)} =====")
    subsets = {"all samples": base, "dedup": base[base["dedup"]],
               "Yes/No markets (book = token 1)": base[base.grp == 1], "other markets (book = token 2)": base[base.grp == -1]}
    for name, df in subsets.items():
        cms = {k: hourly_cms(df, k) for k in KEYS}
        pt = {k: ba_from_cm(cms[k].sum(0)) for k in cms}
        nh = len(cms["lob"])
        boot = rng.integers(0, nh, size=(2000, nh))
        bs = {k: ba_from_cm(cms[k][boot].sum(1)) for k in cms}
        def ci(a, b):
            dlt = bs[a] - bs[b]; lo, hi = np.percentile(dlt, [2.5, 97.5])
            return f"{100 * (pt[a] - pt[b]):+.2f} pt [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        print(f"{name:<34} n={len(df):>8}  LOB-only {pt['lob']:.4f} | graph {pt['full_ob']:.4f} | graph + skill {pt['full_ob_sk']:.4f} | graph + placebo skill {pt['full_ob_skp']:.4f}")
        print(f"{'':<34} graph effect: no skill {ci('full_ob', 'lob')}   with skill {ci('full_ob_sk', 'lob')}   with placebo {ci('full_ob_skp', 'lob')}")
        print(f"{'':<34} skill − no skill: {ci('full_ob_sk', 'full_ob')}   skill − placebo: {ci('full_ob_sk', 'full_ob_skp')}")
