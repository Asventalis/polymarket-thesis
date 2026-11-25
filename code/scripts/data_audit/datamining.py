"""Data-mining study of the 61 hand features (validation day, 13 March).

Stages, per horizon (1m, 15m, 1h, 6h, 12h), all on CPU:
  A  structure of the feature space: PCA, Spearman correlation + hierarchical
     clustering of the features, Gaussian random projection, metric MDS, UMAP, t-SNE,
     neighbourhood purity of class / category / market in the embeddings;
  B  separability: LDA, multinomial logistic regression, kNN (balanced vote, several k),
     depth-3 decision tree, mutual information feature-label; LightGBM (G5) and Mean
     Reversion on the same samples for reference;
  C  (1h only, applied to all horizons) clustering of the markets on per-market aggregates:
     Ward + k-means + HDBSCAN, cross-tab with the Gamma category, per-cluster accuracy.

Inputs are the feature parquets built by feature_audit.py (data_audit/).  Outputs go to
data_audit/datamining/<H>/ as CSV/JSON/PNG and a report.md per horizon.  Nothing touches
the test set.  Usage:  datamining.py --horizon 15m [--stage A,B,C] [--force]
"""
import sys, os, json, time, argparse, warnings
sys.path.insert(0, "/data/v.pampinella/dm_pkgs")
os.environ.setdefault("OMP_NUM_THREADS", "32"); os.environ.setdefault("MKL_NUM_THREADS", "32")
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path("/data/v.pampinella/data_audit")
FILES = {"1m": ("deep_1m/val_features_1m.parquet", "deep_1m/train_features_1m.parquet", "deep_1m/pred_1m_G5.npy"),
         "15m": ("deep_15m/val_features_15m.parquet", "deep_15m/train_features_15m.parquet", "deep_15m/pred_15m_G5.npy"),
         "1h": ("val_features.parquet", "train_features.parquet", "pred_G5.npy"),
         "6h": ("val_features_6h.parquet", "train_features_6h.parquet", "pred_6h_G5.npy"),
         "12h": ("val_features_12h.parquet", "train_features_12h.parquet", "pred_12h_G5.npy")}
GROUPS = {
  "G1 price": ["mid", "ret_15m", "ret_1h", "ret_3h", "ret_6h", "ret_12h", "sgn_15m", "sgn_1h", "sgn_3h", "sgn_6h", "sgn_12h", "hour"],
  "G2 book": ["spread_ticks", "imb1", "imb5", "imb10", "micro_minus_mid_ticks", "log_bid_sz0", "log_ask_sz0", "log_bid_sz10", "log_ask_sz10", "bid_levels", "ask_levels", "depth_skew", "bid_rel1", "ask_rel1"],
  "G3 flow/clock": ["ofi", "ofi_5", "ofi_20", "ofi_15m", "ofi_20_z", "ofi_15m_z", "log_vol_15m", "buy_ratio", "cancel_ratio", "log_dt_prev", "since_row", "n_1m", "n_5m", "n_15m", "n_60m", "moves_60m", "rv_60m", "range_pos_60m", "mid_minus_mean60", "since_move", "last_move_sign"],
  "G4 trades": ["flow_5m", "flow_15m", "flow_60m", "usd_15m", "usd_60m", "ntr_15m", "flow_ratio_15m", "last_trade_minus_mid_ticks", "log_since_trade"],
  "G5 static": ["is_yesno", "log_hours_to_end", "log_duration_h", "log_hours_since_start"],
}
FEATS = [f for g in GROUPS.values() for f in g]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
META = ["market_id", "t", "y", "dedup", "mid_model", "past_delta", "pred_tf64", "category"]
CLASS = ["DOWN", "CONST", "UP"]
# thesis palette
C_A, C_B, C_C, C_D, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#d9d8d3"
CLASS_COL = {0: C_B, 1: "#b5b3ae", 2: C_A}
CAT_COL = {"esports": C_A, "sports": C_C, "crypto": C_D, "weather": "#8e5bd6", "politics": C_B, "economy_tech": "#d64f8e", "culture": "#6b8e23", "other": "#8a8884"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 220, "savefig.dpi": 220, "savefig.bbox": "tight"})

ap = argparse.ArgumentParser(); ap.add_argument("--horizon", required=True); ap.add_argument("--stage", default="A,B,C")
ap.add_argument("--force", action="store_true"); ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args(); H = A.horizon; STAGES = A.stage.split(","); OUT = D / "datamining" / H; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(s):
    m = f"[{H} {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:5.1f}m] {s}"; print(m, flush=True)
    with open(OUT / "log.txt", "a") as f: f.write(m + "\n")
def done(name): return (OUT / name).exists() and not A.force
def ba(y, p):
    cm = np.bincount(y.astype(int) * 3 + p.astype(int), minlength=9).reshape(3, 3).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"): return float(np.nanmean(np.diag(cm) / cm.sum(1)))
def boot_delta(t, y, p, r, n=2000, seed=0):
    """hourly block bootstrap of 100*(BA(p)-BA(r)); returns (delta, lo, hi)"""
    h = pd.factorize(t.values.astype("datetime64[h]"), sort=True)[0]
    def cms(pp): return np.bincount(h * 9 + y.astype(int) * 3 + pp.astype(int), minlength=(h.max() + 1) * 9).reshape(-1, 3, 3).astype(float)
    cp, cr = cms(p), cms(r); rng = np.random.default_rng(seed); nh = len(cp); idx = rng.integers(0, nh, size=(n, nh))
    def bam(c):
        with np.errstate(invalid="ignore", divide="ignore"): return np.nanmean(np.diagonal(c, axis1=-2, axis2=-1) / c.sum(-1), axis=-1)
    d = 100 * (bam(cp[idx].sum(1)) - bam(cr[idx].sum(1)))
    return float(100 * (bam(cp.sum(0)) - bam(cr.sum(0)))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
def mr_rule(past_delta):
    pdl = np.asarray(past_delta, float); return np.where(np.isnan(pdl) | (pdl == 0), 1, np.where(pdl > 0, 0, 2))

# ── load ─────────────────────────────────────────────────────────────────────────────
vf, tf, pf = FILES[H]
log(f"loading {vf}")
val = pq.read_table(D / vf, columns=META + FEATS).to_pandas(); val["t"] = pd.to_datetime(val.t)
val["pred_lgb"] = np.load(D / pf); val["pred_mr"] = mr_rule(val.past_delta.values)
yv = val.y.values.astype(int); log(f"val {val.shape}; class shares {np.bincount(yv, minlength=3)/len(yv)}")
log(f"loading {tf}")
tr = pq.read_table(D / tf, columns=["y", "market_id"] + FEATS).to_pandas(); ytr = tr.y.values.astype(int); log(f"train {tr.shape}")
DROPPED = [f for f in FEATS if tr[f].nunique(dropna=True) <= 1]; FEATS = [f for f in FEATS if f not in DROPPED]; log(f"dropped (constant in train): {DROPPED}")
rng = np.random.default_rng(0)
# preprocessing: median imputation (train) + rank-gauss (QuantileTransformer fitted on 1M train rows)
from sklearn.preprocessing import QuantileTransformer
med = tr[FEATS].median()
def prep_fit():
    sub = rng.choice(len(tr), size=min(1_000_000, len(tr)), replace=False)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=1_000_000, random_state=0)
    qt.fit(tr[FEATS].iloc[sub].fillna(med).values.astype(np.float32)); return qt
QT = prep_fit()
def prep(df): return QT.transform(df[FEATS].fillna(med).values.astype(np.float32)).astype(np.float32)
log("preprocessing fitted (median impute + rank-gauss)")
iv = rng.choice(len(val), size=200_000, replace=False); Xv200 = prep(val.iloc[iv]); yv200 = yv[iv]
report = [f"# Data-mining study, horizon {H} (validation day 13 March)\n",
          f"Validation samples {len(val):,}, class shares {np.round(np.bincount(yv, minlength=3)/len(yv), 4).tolist()}; training rows {len(tr):,}. "
          f"Features: {len(FEATS)} numeric hand features ({', '.join(f'{len(v)} {k}' for k, v in GROUPS.items())}); `category` kept as a label, not a feature. "
          "Preprocessing: median imputation (train) and rank-gauss transform (QuantileTransformer to a normal, fitted on 1M train rows). "
          f"Features constant in the training rows and therefore dropped: {DROPPED or 'none'}.\n"]
def flush(): open(OUT / "report.md", "w").write("\n".join(report) + "\n")

# ═════ Stage A: structure of the feature space ═══════════════════════════════════════
if "A" in STAGES:
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.manifold import MDS, TSNE
    from sklearn.neighbors import NearestNeighbors
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
    from scipy.spatial.distance import squareform, pdist
    # A1 PCA
    pca = PCA(n_components=len(FEATS), random_state=0).fit(Xv200)
    evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
    n50, n80, n95 = [int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.8, 0.95)]
    pr = float(evr.sum() ** 2 / (evr ** 2).sum())  # participation ratio of the variance spectrum
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "evr": evr, "cum": cum}).to_csv(OUT / "pca_spectrum.csv", index=False)
    L = pd.DataFrame(pca.components_[:6].T, index=FEATS, columns=[f"PC{i}" for i in range(1, 7)]); L["group"] = [GROUP_OF[f] for f in FEATS]
    L.to_csv(OUT / "pca_loadings.csv")
    gl = (L[[f"PC{i}" for i in range(1, 7)]] ** 2).groupby(L.group).sum(); gl.to_csv(OUT / "pca_group_loadings.csv")
    top = {f"PC{i}": [f"{f} ({L[f'PC{i}'][f]:+.2f})" for f in L[f"PC{i}"].abs().sort_values(ascending=False).index[:4]] for i in range(1, 5)}
    log(f"PCA: n50={n50} n80={n80} n95={n95} PR={pr:.1f}")
    report += ["\n## A1. PCA (200k validation rows, rank-gauss features)\n",
               f"Components for 50/80/95% of the variance: {n50}/{n80}/{n95} of {len(FEATS)}; participation ratio of the spectrum {pr:.1f}. First four components: "
               + "; ".join(f"{k} ({evr[int(k[2:])-1]:.1%}): {', '.join(v)}" for k, v in top.items()) + ".\n",
               "Share of the squared loadings by feature group:\n", gl.round(3).to_markdown()]
    # A2 feature correlation + hierarchical clustering of the features
    rho = pd.DataFrame(Xv200, columns=FEATS).rank().corr().values  # Spearman via ranks
    np.fill_diagonal(rho, 1.0); dist = 1 - np.abs(rho); dist = np.clip((dist + dist.T) / 2, 0, None); np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average"); order = leaves_list(Z)
    pd.DataFrame(rho, index=FEATS, columns=FEATS).to_csv(OUT / "spearman.csv")
    blocks = {}
    for thr in (0.5, 0.7, 0.9):
        cl = fcluster(Z, t=1 - thr, criterion="distance"); groups = pd.Series(FEATS).groupby(cl).apply(list)
        blocks[thr] = [g for g in groups if len(g) > 1]
    lam = np.linalg.eigvalsh(rho); lam = np.clip(lam, 0, None); eff = float(lam.sum() ** 2 / (lam ** 2).sum())
    json.dump({"effective_n_features": eff, "blocks": {str(k): v for k, v in blocks.items()}}, open(OUT / "corr_blocks.json", "w"), indent=1)
    log(f"corr: effective n features {eff:.1f}; blocks at |rho|>=0.7: {len(blocks[0.7])}")
    report += ["\n## A2. Correlation structure of the features\n",
               f"Effective number of features (participation ratio of the correlation spectrum): {eff:.1f}. Blocks of features with pairwise |Spearman| ≥ 0.7 (average linkage): "
               + "; ".join("{" + ", ".join(b) + "}" for b in blocks[0.7]) + f". At ≥ 0.9: " + "; ".join("{" + ", ".join(b) + "}" for b in blocks[0.9]) + ".\n"]
    fig, ax = plt.subplots(figsize=(6.3, 5.6)); im = ax.imshow(rho[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS))); ax.set_yticks(range(len(FEATS))); lab = [FEATS[i] for i in order]
    ax.set_xticklabels(lab, rotation=90, fontsize=4.2); ax.set_yticklabels(lab, fontsize=4.2)
    gcol = {g: c for g, c in zip(GROUPS, [C_A, C_B, C_C, C_D, "#8e5bd6"])}
    for tl in ax.get_xticklabels() + ax.get_yticklabels(): tl.set_color(gcol[GROUP_OF[tl.get_text()]])
    ax.spines[:].set_visible(False); cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
    fig.savefig(OUT / "corr_heatmap.png"); plt.close(fig)
    # A3 random projection: distance distortion on 3k points
    i3 = rng.choice(len(Xv200), 3000, replace=False); d0 = pdist(Xv200[i3])
    rp_rows = []
    for k in (10, 20, 40):
        Xr = GaussianRandomProjection(n_components=k, random_state=0).fit_transform(Xv200[i3]); r = pdist(Xr) / d0
        rp_rows.append(dict(k=k, ratio_mean=r.mean(), ratio_sd=r.std(), p05=np.percentile(r, 5), p95=np.percentile(r, 95)))
    RP = pd.DataFrame(rp_rows); RP.to_csv(OUT / "random_projection.csv", index=False); log(f"RP done {RP.ratio_sd.round(3).tolist()}")
    report += ["\n## A3. Gaussian random projection (3,000 rows): ratio projected/original pairwise distance\n", RP.round(3).to_markdown(index=False)]
    # A4 metric MDS on 2k points vs PCA-2D
    i2 = rng.choice(len(Xv200), 2000, replace=False); Dm = squareform(pdist(Xv200[i2]))
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=2, max_iter=300, normalized_stress="auto"); Em = mds.fit_transform(Dm)
    Ep = PCA(2, random_state=0).fit_transform(Xv200[i2])
    def stress1(E): d = pdist(E); return float(np.sqrt(((d - pdist(Xv200[i2])) ** 2).sum() / (pdist(Xv200[i2]) ** 2).sum()))
    s_mds, s_pca = stress1(Em), stress1(Ep); log(f"MDS stress-1 {s_mds:.3f} vs PCA-2D {s_pca:.3f}")
    pd.DataFrame({"mds1": Em[:, 0], "mds2": Em[:, 1], "pc1": Ep[:, 0], "pc2": Ep[:, 1], "y": yv200[i2], "category": val.category.values[iv][i2]}).to_csv(OUT / "mds_2k.csv", index=False)
    report += [f"\n## A4. Metric MDS (2,000 rows)\nKruskal stress-1 of the 2-D metric MDS: {s_mds:.3f}; of the PCA 2-D projection on the same rows: {s_pca:.3f}.\n"]
    # A5 UMAP (50k) and t-SNE (20k) with neighbourhood purity
    import umap
    i5 = rng.choice(len(Xv200), 50_000, replace=False); X5 = Xv200[i5]; meta5 = val.iloc[iv[i5]].copy()
    meta5["mid_bucket"] = pd.cut(np.abs(meta5.mid_model - 0.5), [-0.01, 0.25, 0.45, 0.5], labels=["central (|mid-.5|<.25)", "mid (.25-.45)", "extreme (>.45)"])
    meta5["act_bucket"] = pd.qcut(np.log1p(meta5.n_60m), 4, labels=["q1 quiet", "q2", "q3", "q4 busy"], duplicates="drop")
    if not done("umap_50k.csv"):
        t1 = time.time(); U = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=0).fit_transform(X5); log(f"UMAP 50k in {(time.time()-t1)/60:.1f} min")
        i6 = rng.choice(len(X5), 20_000, replace=False)
        t1 = time.time(); Ts = TSNE(n_components=2, perplexity=50, init="pca", random_state=0, n_jobs=A.jobs).fit_transform(X5[i6]); log(f"t-SNE 20k in {(time.time()-t1)/60:.1f} min")
        E = meta5[["market_id", "y", "category", "mid_model", "mid_bucket", "act_bucket", "dedup", "pred_lgb", "pred_mr"]].copy(); E["u1"], E["u2"] = U[:, 0], U[:, 1]
        E["ts1"] = np.nan; E["ts2"] = np.nan; E.iloc[i6, E.columns.get_loc("ts1")] = Ts[:, 0]; E.iloc[i6, E.columns.get_loc("ts2")] = Ts[:, 1]
        E.to_csv(OUT / "umap_50k.csv", index=False)
    E = pd.read_csv(OUT / "umap_50k.csv")
    def purity(Xe, labels, k=15):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=A.jobs).fit(Xe); idx = nn.kneighbors(Xe, return_distance=False)[:, 1:]
        lab = pd.factorize(labels)[0]; same = (lab[idx] == lab[:, None]).mean(); p = np.bincount(lab) / len(lab); chance = float((p ** 2).sum())
        return float(same), chance
    pur = []
    for name, Xe in [("umap", E[["u1", "u2"]].values), ("tsne", E.dropna(subset=["ts1"])[["ts1", "ts2"]].values), ("original60", X5)]:
        Es = E if name != "tsne" else E.dropna(subset=["ts1"])
        for lab in ["y", "category", "mid_bucket", "act_bucket", "market_id"]:
            s, c = purity(Xe, Es[lab].astype(str).values); pur.append(dict(space=name, label=lab, same_neighbour=s, chance=c, lift=s / c))
    PUR = pd.DataFrame(pur); PUR.to_csv(OUT / "purity.csv", index=False); log("purity done")
    report += ["\n## A5. UMAP (50k rows, n_neighbors 30) and t-SNE (20k rows, perplexity 50): 15-NN neighbourhood purity\n",
               "`same_neighbour` = share of the 15 nearest neighbours carrying the same label; `chance` = Σ p²; `lift` = ratio.\n", PUR.round(3).to_markdown(index=False)]
    # figures: UMAP 2x2 panels, t-SNE 1x2
    def scat(ax, x, y, labels, cmap, order=None, title="", s=1.2):
        order = order or list(cmap)
        for lab in order:
            m = labels == lab
            if m.any(): ax.scatter(x[m], y[m], s=s, c=cmap[lab], alpha=0.55, linewidths=0, label=f"{lab} ({m.mean():.0%})", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False); ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.legend(fontsize=5.5, markerscale=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, handletextpad=0.2, columnspacing=0.8)
    seq = ["#bcd4f1", "#5a93dc", "#154a8a"]
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.6))
    ycls = E.y.map(dict(enumerate(CLASS))).values
    scat(axs[0, 0], E.u1.values, E.u2.values, ycls, {"CONST": CLASS_COL[1], "DOWN": CLASS_COL[0], "UP": CLASS_COL[2]}, order=["CONST", "DOWN", "UP"], title=f"a) label at {H}")
    scat(axs[0, 1], E.u1.values, E.u2.values, E.category.values, CAT_COL, title="b) market category")
    mb = ["central (|mid-.5|<.25)", "mid (.25-.45)", "extreme (>.45)"]; scat(axs[1, 0], E.u1.values, E.u2.values, E.mid_bucket.values, dict(zip(mb, seq)), order=mb, title="c) distance of the mid from 0.5")
    ab = ["q1 quiet", "q2", "q3", "q4 busy"]; scat(axs[1, 1], E.u1.values, E.u2.values, E.act_bucket.values, dict(zip(ab, ["#f9d9c9", "#f3a98a", "#eb6834", "#9a3a12"])), order=ab, title="d) rows in the last 60 min (quartile)")
    fig.tight_layout(); fig.savefig(OUT / "umap_panels.png"); plt.close(fig)
    Et = E.dropna(subset=["ts1"]); fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.5))
    scat(axs[0], Et.ts1.values, Et.ts2.values, Et.y.map(dict(enumerate(CLASS))).values, {"CONST": CLASS_COL[1], "DOWN": CLASS_COL[0], "UP": CLASS_COL[2]}, order=["CONST", "DOWN", "UP"], title=f"a) t-SNE, label at {H}")
    scat(axs[1], Et.ts1.values, Et.ts2.values, Et.category.values, CAT_COL, title="b) t-SNE, market category")
    fig.tight_layout(); fig.savefig(OUT / "tsne_panels.png"); plt.close(fig); log("stage A figures done"); flush()

# ═════ Stage B: separability ═════════════════════════════════════════════════════════
if "B" in STAGES:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.random_projection import GaussianRandomProjection
    itr = rng.choice(len(tr), size=min(1_000_000, len(tr)), replace=False); Xtr = prep(tr.iloc[itr]); ytr1 = ytr[itr]
    Xv_all = prep(val); log("val transformed")
    rows = []; preds = {}
    def add(name, p, note=""):
        preds[name] = p; rows.append(dict(model=name, ba_all=ba(yv, p), ba_dedup=ba(yv[val.dedup.values], p[val.dedup.values]), note=note)); log(f"{name}: {rows[-1]['ba_all']:.4f} / {rows[-1]['ba_dedup']:.4f}")
    add("Mean Reversion", val.pred_mr.values); add("LightGBM G5 (control)", val.pred_lgb.values, "raw, uncalibrated")
    if H in ("1h", "6h", "12h"): add("stateful LOB-only (October)", val.pred_tf64.values, "pred_tf64 in the audit parquet")
    # B1 LDA
    lda = LinearDiscriminantAnalysis(solver="svd", priors=[1/3, 1/3, 1/3]).fit(Xtr, ytr1); add("LDA (uniform priors)", lda.predict(Xv_all).astype(int))
    S = pd.DataFrame(lda.scalings_[:, :2], index=FEATS, columns=["LD1", "LD2"]); S["group"] = [GROUP_OF[f] for f in FEATS]; S.to_csv(OUT / "lda_scalings.csv")
    ldtop = {c: [f"{f} ({S[c][f]:+.2f})" for f in S[c].abs().sort_values(ascending=False).index[:5]] for c in ("LD1", "LD2")}
    i20 = rng.choice(len(val), 20_000, replace=False); Z2 = lda.transform(Xv_all[i20])[:, :2]
    pd.DataFrame({"ld1": Z2[:, 0], "ld2": Z2[:, 1], "y": yv[i20], "category": val.category.values[i20], "mid_model": val.mid_model.values[i20]}).to_csv(OUT / "lda_2d_20k.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for c in (1, 0, 2):
        m = yv[i20] == c; ax.scatter(Z2[m, 0], Z2[m, 1], s=1.5, c=CLASS_COL[c], alpha=0.5, linewidths=0, label=f"{CLASS[c]} ({m.mean():.0%})", rasterized=True)
    ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.legend(frameon=False, markerscale=5, fontsize=7); fig.savefig(OUT / "lda_2d.png"); plt.close(fig)
    # B2 logistic regression
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, n_jobs=A.jobs).fit(Xtr, ytr1); add("Logistic regression (balanced)", lr.predict(Xv_all).astype(int))
    # B3 kNN with balanced vote on a 300k validation subsample
    iq = rng.choice(len(val), 300_000, replace=False); Xq = Xv_all[iq]; yq = yv[iq]
    t1 = time.time(); nn = NearestNeighbors(n_neighbors=1000, algorithm="brute", n_jobs=A.jobs).fit(Xtr); nid = nn.kneighbors(Xq, return_distance=False); log(f"kNN search 300k x 1M in {(time.time()-t1)/60:.1f} min")
    Ln = ytr1[nid]; w = 1.0 / (np.bincount(ytr1, minlength=3) / len(ytr1))
    cums = np.stack([np.cumsum(Ln == c, axis=1) for c in range(3)], -1).astype(np.float32)  # (300k, 1000, 3)
    knn_rows = []
    for k in (5, 20, 50, 200, 1000):
        pk = np.argmax(cums[:, k - 1, :] * w, axis=1); pk_plain = np.argmax(cums[:, k - 1, :], axis=1)
        d, lo, hi = boot_delta(val.t.iloc[iq], yq, pk, val.pred_lgb.values[iq])
        knn_rows.append(dict(k=k, ba_all=ba(yq, pk), ba_dedup=ba(yq[val.dedup.values[iq]], pk[val.dedup.values[iq]]), ba_plain_vote=ba(yq, pk_plain), delta_vs_lgb=d, lo=lo, hi=hi))
        log(f"kNN k={k}: {knn_rows[-1]['ba_all']:.4f} (plain vote {knn_rows[-1]['ba_plain_vote']:.4f}) Δ vs LGB {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    del cums
    # kNN on a 20-D random projection, k=50
    rp = GaussianRandomProjection(n_components=20, random_state=0).fit(Xtr); nn2 = NearestNeighbors(n_neighbors=50, algorithm="brute", n_jobs=A.jobs).fit(rp.transform(Xtr))
    L2 = ytr1[nn2.kneighbors(rp.transform(Xq), return_distance=False)]; p2 = np.argmax(np.stack([(L2 == c).sum(1) for c in range(3)], -1) * w, axis=1)
    knn_rows.append(dict(k=50, ba_all=ba(yq, p2), ba_dedup=ba(yq[val.dedup.values[iq]], p2[val.dedup.values[iq]]), ba_plain_vote=np.nan, delta_vs_lgb=np.nan, lo=np.nan, hi=np.nan, note="on 20-D Gaussian random projection"))
    KNN = pd.DataFrame(knn_rows); KNN.to_csv(OUT / "knn.csv", index=False)
    ref_rows = {n: ba(yq, preds[n][iq]) for n in preds}; ref_rows_d = {n: ba(yq[val.dedup.values[iq]], preds[n][iq][val.dedup.values[iq]]) for n in preds}
    # B4 depth-3 tree
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", min_samples_leaf=5000, random_state=0).fit(tr[FEATS].iloc[itr].fillna(med).values, ytr1)
    pdt = dt.predict(val[FEATS].fillna(med).values).astype(int); add("Decision tree, depth 3 (balanced)", pdt, "raw features")
    tree_txt = export_text(dt, feature_names=FEATS, class_names=CLASS, decimals=3, show_weights=False); open(OUT / "tree_depth3.txt", "w").write(tree_txt)
    dt5 = DecisionTreeClassifier(max_depth=6, class_weight="balanced", min_samples_leaf=5000, random_state=0).fit(tr[FEATS].iloc[itr].fillna(med).values, ytr1)
    add("Decision tree, depth 6 (balanced)", dt5.predict(val[FEATS].fillna(med).values).astype(int), "raw features")
    # B5 mutual information on 300k val rows
    t1 = time.time(); mi = mutual_info_classif(Xq, yq, n_neighbors=3, random_state=0, n_jobs=A.jobs); log(f"MI in {(time.time()-t1)/60:.1f} min")
    MI = pd.DataFrame({"feature": FEATS, "group": [GROUP_OF[f] for f in FEATS], "mi_nats": mi}).sort_values("mi_nats", ascending=False); MI.to_csv(OUT / "mutual_information.csv", index=False)
    Hy = float(-(np.bincount(yq, minlength=3) / len(yq) * np.log(np.bincount(yq, minlength=3) / len(yq))).sum())
    SEP = pd.DataFrame(rows); SEP.to_csv(OUT / "separability.csv", index=False)
    report += ["\n## B. Separability (training subsample 1M rows; evaluation on the full validation day unless stated)\n", SEP.round(4).to_markdown(index=False),
               f"\nLDA leading discriminant directions (standardised scalings): LD1: {', '.join(ldtop['LD1'])}; LD2: {', '.join(ldtop['LD2'])}.\n",
               "\n### kNN, balanced vote (300k validation rows; Δ in balanced-accuracy points vs LightGBM G5 on the same rows, hourly block bootstrap 95% CI)\n",
               f"Reference on the same 300k rows: " + ", ".join(f"{n} {v:.4f} (dedup {ref_rows_d[n]:.4f})" for n, v in ref_rows.items()) + ".\n", KNN.round(4).to_markdown(index=False),
               "\n### Depth-3 decision tree (balanced class weights, raw features)\n```\n" + tree_txt + "```\n",
               f"\n### Mutual information feature–label (300k rows, k=3 estimator); label entropy {Hy:.3f} nats\n", MI.head(15).round(4).to_markdown(index=False),
               "\nBy group (sum of MI):\n", MI.groupby("group").mi_nats.sum().round(4).to_markdown()]
    flush()

# ═════ Stage C: market-level clustering ═════════════════════════════════════════════
if "C" in STAGES:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
    from sklearn.preprocessing import StandardScaler
    import umap
    g = val.groupby("market_id", observed=True)
    M = pd.DataFrame({"n": g.size(), "spread_ticks": g.spread_ticks.median(), "log_n60": g.n_60m.apply(lambda s: np.log1p(s).mean()), "rv_60m": g.rv_60m.apply(lambda s: np.log1p(100 * s).mean()),
                      "mid_dist": g.mid_model.apply(lambda s: np.abs(s - 0.5).mean()), "p_move": g.y.apply(lambda s: (s != 1).mean()), "depth0": (g.log_bid_sz0.mean() + g.log_ask_sz0.mean()) / 2,
                      "log_duration_h": g.log_duration_h.median(), "log_hours_to_end": g.log_hours_to_end.median(), "usd_60m": g.usd_60m.mean(), "cancel_ratio": g.cancel_ratio.mean(),
                      "log_moves_60m": g.moves_60m.apply(lambda s: np.log1p(s).mean()), "is_yesno": g.is_yesno.first(), "category": g.category.first()})
    M = M[M.n >= 100].copy(); M["log_n"] = np.log(M.n); M["is_yesno"] = M.is_yesno.fillna(0); CF = ["log_n", "spread_ticks", "log_n60", "rv_60m", "mid_dist", "p_move", "depth0", "log_duration_h", "log_hours_to_end", "usd_60m", "cancel_ratio", "log_moves_60m", "is_yesno"]
    M[CF] = M[CF].fillna(M[CF].median()); Xm = StandardScaler().fit_transform(M[CF].clip(M[CF].quantile(0.005), M[CF].quantile(0.995), axis=1).values); log(f"markets with ≥100 samples: {len(M)}")
    Zm = linkage(Xm, method="ward"); sil = []
    for k in range(2, 11):
        lw = fcluster(Zm, k, criterion="maxclust"); km = KMeans(k, n_init=10, random_state=0).fit_predict(Xm)
        sil.append(dict(k=k, sil_ward=silhouette_score(Xm, lw), sil_kmeans=silhouette_score(Xm, km)))
    SIL = pd.DataFrame(sil); SIL.to_csv(OUT / "market_silhouette.csv", index=False); kbest = int(SIL.loc[SIL.sil_ward.idxmax(), "k"]); log(f"silhouette: best k (Ward) {kbest}")
    k_use = max(kbest, 4)  # at least four clusters for a readable description
    M["ward"] = fcluster(Zm, k_use, criterion="maxclust"); M["kmeans"] = KMeans(k_use, n_init=10, random_state=0).fit_predict(Xm) + 1
    Um = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0).fit_transform(Xm); M["u1"], M["u2"] = Um[:, 0], Um[:, 1]
    hdb = HDBSCAN(min_cluster_size=max(15, len(M) // 60)).fit(umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5, random_state=0).fit_transform(Xm)); M["hdbscan"] = hdb.labels_
    ami = {m: adjusted_mutual_info_score(M.category.astype(str), M[m]) for m in ("ward", "kmeans", "hdbscan")}
    # per-cluster accuracy at this horizon
    cl_of = M.ward.to_dict(); vc = val.market_id.map(cl_of)
    desc = M.groupby("ward")[CF].mean().round(2); desc["n_markets"] = M.groupby("ward").size()
    ct = pd.crosstab(M.ward, M.category); ct.to_csv(OUT / "market_cluster_x_category.csv")
    acc = []
    for c in sorted(M.ward.unique()):
        m = (vc == c).values
        if m.sum() < 1000: continue
        r = dict(cluster=c, samples=int(m.sum()), markets=int((M.ward == c).sum()), p_move=float((yv[m] != 1).mean()), ba_mr=ba(yv[m], val.pred_mr.values[m]), ba_lgb=ba(yv[m], val.pred_lgb.values[m]))
        if H in ("1h", "6h", "12h"): r["ba_stateful_oct"] = ba(yv[m], val.pred_tf64.values[m]); r["delta_stateful_vs_lgb"] = 100 * (r["ba_stateful_oct"] - r["ba_lgb"])
        r["top_categories"] = ", ".join(f"{k} {v}" for k, v in ct.loc[c].sort_values(ascending=False).head(3).items())
        acc.append(r)
    ACC = pd.DataFrame(acc); ACC.to_csv(OUT / "market_cluster_accuracy.csv", index=False); M.to_csv(OUT / "markets.csv")
    log("clusters: " + "; ".join(f"{r['cluster']}: {r['markets']} mk, BA lgb {r['ba_lgb']:.3f}" for r in acc))
    # figures: dendrogram + UMAP of markets
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    dendrogram(Zm, ax=axs[0], truncate_mode="lastp", p=30, no_labels=True, color_threshold=Zm[-(k_use - 1), 2], above_threshold_color=INK, leaf_rotation=90)
    axs[0].set_title("a) Ward dendrogram of the markets (30 leaves)", loc="left", fontsize=8, color=INK); axs[0].set_ylabel("Ward distance"); axs[0].set_xticks([])
    cmap_cl = dict(zip(sorted(M.ward.unique()), [C_A, C_B, C_C, C_D, "#8e5bd6", "#d64f8e", "#6b8e23", "#8a8884", "#154a8a", "#9a3a12"]))
    for c in sorted(M.ward.unique()):
        m = M.ward == c; axs[1].scatter(M.u1[m], M.u2[m], s=4, c=cmap_cl[c], alpha=0.7, linewidths=0, label=f"cluster {c} ({m.sum()})", rasterized=True)
    axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].spines[:].set_visible(False); axs[1].set_title("b) UMAP of the markets, Ward clusters", loc="left", fontsize=8, color=INK)
    axs[1].legend(fontsize=5.5, markerscale=3, frameon=False, loc="best", handletextpad=0.2); fig.tight_layout(); fig.savefig(OUT / "markets_clusters.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for c in CAT_COL:
        m = M.category.astype(str) == c
        if m.any(): ax.scatter(M.u1[m], M.u2[m], s=4, c=CAT_COL[c], alpha=0.7, linewidths=0, label=f"{c} ({m.sum()})", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False); ax.set_title("UMAP of the markets, category", loc="left", fontsize=8, color=INK); ax.legend(fontsize=5.5, markerscale=3, frameon=False, loc="best", handletextpad=0.2)
    fig.savefig(OUT / "markets_category.png"); plt.close(fig)
    report += [f"\n## C. Clustering of the markets ({len(M)} markets with ≥ 100 validation samples; 13 standardised per-market aggregates)\n",
               "Silhouette by k:\n", SIL.round(3).to_markdown(index=False), f"\nUsed k = {k_use} (Ward). Adjusted mutual information with the Gamma category: " + ", ".join(f"{k} {v:.3f}" for k, v in ami.items()) + f"; HDBSCAN found {int((M.hdbscan.max() + 1))} clusters and {int((M.hdbscan == -1).mean() * 100)}% noise.\n",
               "\nCluster profiles (means of the aggregates):\n", desc.to_markdown(), "\n\nCluster × category (markets):\n", ct.to_markdown(), "\n\nAccuracy inside each cluster at this horizon:\n", ACC.round(4).to_markdown(index=False)]

flush(); log("ALL DONE")
