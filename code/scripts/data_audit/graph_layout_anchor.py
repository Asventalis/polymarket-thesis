"""Fallback large-graph layout for the wallet-market bipartite graph (bipartite anchoring).

Markets are mapped with UMAP from their 64-d SVD coordinates (svd_joint.npy of
graph_animation.py, stage L); every wallet is placed at the fill-weighted centroid of the
markets it traded, pushed slightly outwards from the local market cloud with a deterministic
jitter so that single-market wallets form a halo around their market instead of a point.
Writes layout_anchor.npz (same format as layout.npz).  usage: ./run_dev.sh graph_layout_anchor.py [--jobs 48]"""
import argparse, os, sys, time
sys.path.insert(0, "/data/v.pampinella/dm_pkgs")
import numpy as np, scipy.sparse as sp

OUT = "/data/v.pampinella/data_audit/graphviz"
p = argparse.ArgumentParser(); p.add_argument("--jobs", type=int, default=48); p.add_argument("--neighbors", type=int, default=15)
p.add_argument("--cat-weight", type=float, default=0.0, help="weight of the one-hot category block appended to the unit-norm SVD vector (0 = similarity only)")
p.add_argument("--out", default="layout_anchor.npz"); a = p.parse_args()


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


e = np.load(f"{OUT}/edges.npz"); ew, em, mult = e["ew"], e["em"], e["mult"].astype(np.float64)
nW, nM = int(ew.max()) + 1, int(em.max()) + 1
Z = np.load(f"{OUT}/svd_joint.npy"); ZM = Z[nW:].astype(np.float64)
metric = "cosine"
if a.cat_weight > 0:
    import pandas as pd
    CAT_ORDER = ["crypto", "sports", "esports", "politics", "weather", "economy_tech", "culture", "other"]
    meta = pd.read_parquet(f"{OUT}/market_meta.parquet").sort_values("mi")
    ci = np.array([CAT_ORDER.index(c) if c in CAT_ORDER else CAT_ORDER.index("other") for c in meta["category"].fillna("other")])
    assert len(ci) == nM
    nrm = np.linalg.norm(ZM, axis=1, keepdims=True); nrm[nrm == 0] = 1
    onehot = np.zeros((nM, len(CAT_ORDER))); onehot[np.arange(nM), ci] = a.cat_weight
    ZM = np.hstack([ZM / nrm, onehot]); metric = "euclidean"
    log("category block appended, weight", a.cat_weight, "-> extra distance between categories", round(a.cat_weight * np.sqrt(2), 3), "(within-similarity distance ranges 0-2)")
log("markets", nM, "wallets", nW, "svd", Z.shape)
import umap
t = time.time()
red = umap.UMAP(n_components=2, n_neighbors=a.neighbors, min_dist=0.05, metric=metric, n_jobs=a.jobs, random_state=None, verbose=True)
XM = red.fit_transform(ZM).astype(np.float64)
log("umap markets in", round(time.time() - t), "s")
# wallets: weighted centroid of their markets (weights log1p(fills on the edge))
w = np.log1p(mult)
A = sp.csr_matrix((w, (ew, em)), shape=(nW, nM))
rs = np.asarray(A.sum(1)).ravel(); rs[rs == 0] = 1
XW = (A @ XM) / rs[:, None]
# spread of each wallet's markets (how far apart they are): wallets of one market get a halo
d2 = (A @ (XM ** 2)) / rs[:, None] - XW ** 2
spread = np.sqrt(np.clip(d2.sum(1), 0, None))
degW = np.asarray((A > 0).sum(1)).ravel()
rng = np.random.default_rng(0)
scale = np.median(np.linalg.norm(XM - XM.mean(0), axis=1))
r = np.where(degW == 1, 0.06 * scale * np.sqrt(rng.random(nW)), 0.02 * scale) + 0.15 * spread
ang = rng.random(nW) * 2 * np.pi
XW = XW + np.column_stack([np.cos(ang), np.sin(ang)]) * r[:, None]
XY = np.vstack([XW, XM]).astype(np.float32)
np.savez(f"{OUT}/{a.out}", xy=XY, nW=nW, nM=nM)
log("layout saved", a.out, XY.shape, "spread quantiles", np.quantile(spread, [0.5, 0.9, 0.99]).round(3), "scale", round(scale, 3))
