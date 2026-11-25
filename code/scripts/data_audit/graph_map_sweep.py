"""Sweep of market-map layouts: how much the category (or the contract template) should weigh
against the trading similarity in the 2-D map of the wallet--market graph.

Every variant re-embeds the 157,776 markets with UMAP from their 64-d SVD coordinates
(svd_joint.npy of graph_animation.py), anchors the wallets as graph_layout_anchor.py does, finds
the islands with HDBSCAN as graph_islands.py does, renders the light static maps and writes a few
numbers that say what the map preserves (trading neighbours) and what it imposes (category).

usage (server):  ./run_dev.sh graph_map_sweep.py --variant cat025 [--jobs 36]
                 ./run_dev.sh graph_map_sweep.py --variant knn64        # once, 64-d neighbours for the metrics
                 ./run_dev.sh graph_map_sweep.py --variant render --tags cat000,cat050   # light maps of the old layouts
                 ./run_dev.sh graph_map_sweep.py --variant metrics --tags cat000,cat050,cat025,...
variants:  catW   one-hot category block of weight W/100 appended to the unit-norm SVD vector
                  (cat050 is the layout of 27 Oct; cat000 the similarity-only layout)
           supW   supervised UMAP: category as target, target_weight W/100 (cross-category
                  edges of the neighbour graph are attenuated, the neighbours themselves unchanged)
           nnK    similarity only, K neighbours instead of 15 (more global structure)
           tplW   one-hot block of the contract template (Up/Down length, sports line type,
                  temperature, tweets, price level...) of weight W/100, no category
           dayW   numeric block: first trading day of the market, weight W/100 (date as an axis)
Outputs in /data/v.pampinella/data_audit/graphviz:  layout_anchor_<tag>.npz, market_clusters_<tag>.csv,
market_cluster_labels_<tag>.npy, islands_<tag>.json, static_map_light_<tag>.png,
static_map_lightcat_<tag>.png, map_metrics.csv (one row per tag)."""
import argparse, json, os, re, sys, time, collections
sys.path.insert(0, "/data/v.pampinella/dm_pkgs")
import numpy as np, pandas as pd, scipy.sparse as sp

OUT = "/data/v.pampinella/data_audit/graphviz"
CAT_ORDER = ["crypto", "sports", "esports", "politics", "weather", "economy_tech", "culture", "other"]
# category colours for the light map (darker than the video palette so that they read on white)
CAT_LIGHT = {"crypto": "#d9a400", "sports": "#e8641a", "esports": "#d0343f", "politics": "#b847a8", "weather": "#e07a7a",
             "economy_tech": "#9c7a4a", "culture": "#8a2545", "other": "#8c8c8c"}
WALLET_RGB = np.array([0.165, 0.47, 0.84]); MARKET_RGB = np.array([0.92, 0.41, 0.2])
KNN = 15


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


# ------------------------------------------------------------------ data
def load_base():
    e = np.load(f"{OUT}/edges.npz")
    ew, em, mult = e["ew"], e["em"], e["mult"].astype(np.float64)
    nW, nM = int(ew.max()) + 1, int(em.max()) + 1
    Z = np.load(f"{OUT}/svd_joint.npy"); ZM = Z[nW:].astype(np.float64)
    nrm = np.linalg.norm(ZM, axis=1, keepdims=True); nrm[nrm == 0] = 1
    ZM = ZM / nrm
    meta = pd.read_parquet(f"{OUT}/market_meta.parquet").sort_values("mi").reset_index(drop=True)
    assert len(meta) == nM
    cat = meta["category"].fillna("other").map(lambda c: c if c in CAT_ORDER else "other")
    ci = np.array([CAT_ORDER.index(c) for c in cat])
    return e, ew, em, mult, nW, nM, ZM, meta, ci


def interval(q):
    mm = re.search(r"(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)", str(q))
    if not mm: return None
    h1, m1, p1, h2, m2, p2 = mm.groups()
    t1 = (int(h1) % 12 + (12 if p1 == "PM" else 0)) * 60 + int(m1); t2 = (int(h2) % 12 + (12 if p2 == "PM" else 0)) * 60 + int(m2)
    return (t2 - t1) % 1440


def template(q, smt, cat):
    """Contract template from the question text and the sports line type: ~20 classes."""
    ql = str(q).lower()
    iv = interval(q)
    if iv is not None:
        return f"updown_{iv}m" if iv in (5, 15, 60, 240) else "updown_other"
    if "up or down" in ql: return "updown_long"
    if smt is not None and str(smt) != "None": return f"sports_{smt}"
    if cat == "weather" or "temperature" in ql or "°" in ql: return "temperature"
    if "tweet" in ql: return "tweets"
    if cat == "crypto" and any(k in ql for k in ("above", "below", "reach", "dip", "between", "hit", "price")): return "price_level"
    if cat in ("sports", "esports"):
        for k in ("map", "kill", "winner", "win", "spread", "o/u", "mvp", "champion"):
            if k in ql: return f"{cat}_{k}"
    return f"{cat}_other"


def market_day(q):
    mm = re.search(r"March (\d+)", str(q)); return int(mm.group(1)) if mm else -1


# ------------------------------------------------------------------ embedding variants
def embed(variant, ZM, ci, meta, e, em, nM, jobs):
    import umap
    kind, val = re.match(r"([a-z]+)(\d+)", variant).groups(); val = int(val)
    nn = 15; X = ZM; y = None; kw = {}
    if kind == "cat":
        w = val / 100
        if w > 0:
            oh = np.zeros((nM, len(CAT_ORDER))); oh[np.arange(nM), ci] = w; X = np.hstack([ZM, oh])
        log(f"category block, weight {w}: extra distance between categories {w * np.sqrt(2):.3f} (similarity distance ranges 0-2)")
    elif kind == "sup":
        y = ci; kw = dict(target_metric="categorical", target_weight=val / 100)
        log("supervised umap, target_weight", val / 100)
    elif kind == "nn":
        nn = val; log("similarity only,", nn, "neighbours")
    elif kind == "tpl":
        w = val / 100
        tpl = np.array([template(q, s, c) for q, s, c in zip(meta.question, meta.sports_market_type, meta.category.fillna("other"))])
        names, ti = np.unique(tpl, return_inverse=True)
        oh = np.zeros((nM, len(names))); oh[np.arange(nM), ti] = w; X = np.hstack([ZM, oh])
        log("template block, weight", w, "classes", len(names), dict(collections.Counter(tpl).most_common(12)))
    elif kind == "day":
        w = val / 100
        first = np.full(nM, np.inf); np.minimum.at(first, em, e["first_ts"].astype(np.float64))
        d = (first - first.min()) / 86400; d = (d - d.mean()) / d.std() * w
        X = np.hstack([ZM, d[:, None]]); log("day block, weight", w, "day range", round(d.min(), 2), round(d.max(), 2))
    else:
        raise SystemExit("unknown variant " + variant)
    t = time.time()
    red = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.05, metric="euclidean", n_jobs=jobs, random_state=None, verbose=True, **kw)
    XM = red.fit_transform(X, y=y).astype(np.float64)
    log("umap in", round(time.time() - t), "s")
    return XM


def anchor_wallets(XM, ew, em, mult, nW, nM):
    w = np.log1p(mult)
    A = sp.csr_matrix((w, (ew, em)), shape=(nW, nM))
    rs = np.asarray(A.sum(1)).ravel(); rs[rs == 0] = 1
    XW = (A @ XM) / rs[:, None]
    d2 = (A @ (XM ** 2)) / rs[:, None] - XW ** 2
    spread = np.sqrt(np.clip(d2.sum(1), 0, None))
    degW = np.asarray((A > 0).sum(1)).ravel()
    rng = np.random.default_rng(0)
    scale = np.median(np.linalg.norm(XM - XM.mean(0), axis=1))
    r = np.where(degW == 1, 0.06 * scale * np.sqrt(rng.random(nW)), 0.02 * scale) + 0.15 * spread
    ang = rng.random(nW) * 2 * np.pi
    XW = XW + np.column_stack([np.cos(ang), np.sin(ang)]) * r[:, None]
    return np.vstack([XW, XM]).astype(np.float32)


# ------------------------------------------------------------------ islands
def islands(XM, meta, fills, deg, min_size=150):
    from sklearn.cluster import HDBSCAN
    lab = HDBSCAN(min_cluster_size=min_size, min_samples=10).fit(XM).labels_
    log("islands", lab.max() + 1, "noise share %.3f" % (lab < 0).mean(), "fills in islands %.3f" % (fills[lab >= 0].sum() / fills.sum()))
    stop = set("will the be in on of to a an and or by at for vs is win at than more less over under up down yes no price this that with".split())
    tokens = lambda q: [t for t in re.sub(r"[0-9\$,\.\%\:\(\)\?\-]+", " ", str(q)).split() if t.lower() not in stop and len(t) > 1]
    rows = []
    for c in range(lab.max() + 1):
        m = np.flatnonzero(lab == c); qs = meta.question.values[m]; cats = meta.category.fillna("other").values[m]; cc = collections.Counter(cats)
        tok = collections.Counter(t for q in qs[:4000] for t in tokens(q))
        days = [int(x) for q in qs for x in re.findall(r"March (\d+)", str(q))]
        iv = collections.Counter(x for q in qs[:3000] if (x := interval(q)) is not None)
        rows.append(dict(cluster=c, n=len(m), fills=int(fills[m].sum()), wallets_med=int(np.median(deg[m])), cx=float(XM[m, 0].mean()), cy=float(XM[m, 1].mean()),
                         top_cat=cc.most_common(1)[0][0], purity=cc.most_common(1)[0][1] / len(m), cats="; ".join(f"{k} {v / len(m):.2f}" for k, v in cc.most_common(3)),
                         updown_share=float(np.mean(["Up or Down" in str(q) for q in qs])), day_p10=np.percentile(days, 10) if days else None, day_p90=np.percentile(days, 90) if days else None,
                         interval_min=iv.most_common(1)[0][0] if iv else None, tokens=", ".join(k for k, v in tok.most_common(8)),
                         examples=" | ".join(str(meta.question.values[i])[:60] for i in m[::max(1, len(m) // 3)][:3])))
    return lab, pd.DataFrame(rows).sort_values("n", ascending=False)


# ------------------------------------------------------------------ rendering (same recipe as graph_animation.render_background, light only)
def to_pixels(xy, W, H, margin=0.03):
    lo, hi = np.quantile(xy, 0.003, axis=0), np.quantile(xy, 0.997, axis=0)
    span = (hi - lo).max(); c = (lo + hi) / 2; s = min(W, H) * (1 - 2 * margin) / span
    px = (xy[:, 0] - c[0]) * s + W / 2; py = (c[1] - xy[:, 1]) * s + H / 2
    return np.clip(px, 0, W - 1).astype(np.float32), np.clip(py, 0, H - 1).astype(np.float32)


def raster_lines(x0, y0, x1, y1, w, W, H, max_samples=1000):
    if len(x0) == 0: return np.zeros((H, W), np.float32)
    length = np.hypot(x1 - x0, y1 - y0)
    n = np.clip((length / 1.5).astype(np.int64) + 2, 2, max_samples)
    tot = int(n.sum()); rep = np.repeat(np.arange(len(n)), n); start = np.cumsum(n) - n
    tpos = (np.arange(tot) - start[rep]) / (n[rep] - 1)
    xs = x0[rep] + (x1 - x0)[rep] * tpos; ys = y0[rep] + (y1 - y0)[rep] * tpos
    idx = np.clip(ys, 0, H - 1).astype(np.int64) * W + np.clip(xs, 0, W - 1).astype(np.int64)
    return np.bincount(idx, weights=(w / n)[rep], minlength=W * H).astype(np.float32).reshape(H, W)


def splat(px, py, w, W, H, sigma):
    from scipy.ndimage import gaussian_filter
    img = np.bincount(np.clip(py, 0, H - 1).astype(np.int64) * W + np.clip(px, 0, W - 1).astype(np.int64), weights=w, minlength=W * H).astype(np.float32).reshape(H, W)
    return gaussian_filter(img, sigma) if sigma > 0 else img


def tone(img, gain, gamma=0.5):
    return np.clip((img * gain) ** gamma, 0, 1)


def render_light(xy, nW, nM, e, ew, em, mult, ci, tag, W=1440, H=1080, bg_edges=6_000_000):
    from PIL import Image
    px, py = to_pixels(xy, W, H)
    rng = np.random.default_rng(0)
    keep = rng.random(len(ew)) < min(1.0, bg_edges / len(ew))
    L = raster_lines(px[ew[keep]], py[ew[keep]], px[em[keep] + nW], py[em[keep] + nW], np.log1p(mult[keep]).astype(np.float32), W, H)
    fw = np.log1p(e["fillsW"]).astype(np.float32); fm = np.log1p(e["fillsM"]).astype(np.float32)
    Wn = splat(px[:nW], py[:nW], fw, W, H, 0.8); Mn = splat(px[nW:], py[nW:], fm, W, H, 1.0)
    gL, gW, gM = 1.0 / np.quantile(L[L > 0], 0.995), 1.0 / np.quantile(Wn[Wn > 0], 0.995), 1.0 / np.quantile(Mn[Mn > 0], 0.995)
    base = np.ones((H, W, 3), np.float32)
    base -= tone(L, gL, 0.6)[..., None] * np.array([0.35, 0.35, 0.35]) * 0.5
    base -= tone(Wn, gW, 0.6)[..., None] * (1 - WALLET_RGB) * 0.9
    light = base - tone(Mn, gM, 0.6)[..., None] * (1 - MARKET_RGB) * 0.9
    Image.fromarray((np.clip(light, 0, 1) * 255).astype(np.uint8)).save(f"{OUT}/static_map_light_{tag}.png")
    lightcat = base.copy()
    for k, c in enumerate(CAT_ORDER):
        m = ci == k
        if not m.any(): continue
        lightcat -= tone(splat(px[nW:][m], py[nW:][m], fm[m], W, H, 1.0), gM, 0.6)[..., None] * (1 - hex_rgb(CAT_LIGHT[c])) * 0.9
    Image.fromarray((np.clip(lightcat, 0, 1) * 255).astype(np.uint8)).save(f"{OUT}/static_map_lightcat_{tag}.png")
    log("maps rendered", tag)


def centroids(xy, nW, lab, fills, tag):
    XM = xy[nW:]
    lo, hi = np.quantile(xy, 0.003, axis=0), np.quantile(xy, 0.997, axis=0)
    out = {}
    for l in np.unique(lab):
        if l < 0: continue
        m = lab == l
        out[int(l)] = dict(x=float(np.median(XM[m, 0])), y=float(np.median(XM[m, 1])), n=int(m.sum()), fills=float(fills[m].sum()))
    json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), islands=out), open(f"{OUT}/islands_{tag}.json", "w"))


# ------------------------------------------------------------------ metrics
def knn2d(XM, k=KNN):
    from sklearn.neighbors import NearestNeighbors
    nnb = NearestNeighbors(n_neighbors=k + 1, n_jobs=16).fit(XM)
    return nnb.kneighbors(XM, return_distance=False)[:, 1:]


def metrics(tag, xy, nW, meta, ci, fills, nn64):
    XM = xy[nW:].astype(np.float64); N = knn2d(XM)
    tpl = np.array([template(q, s, c) for q, s, c in zip(meta.question, meta.sports_market_type, meta.category.fillna("other"))])
    names, ti = np.unique(tpl, return_inverse=True)
    day = np.array([market_day(q) for q in meta.question])
    same_cat = (ci[N] == ci[:, None]).mean()
    same_tpl = (ti[N] == ti[:, None]).mean()
    short = np.isin(tpl, ["updown_5m", "updown_15m"]) & (day > 0)
    same_day = (day[N[short]] == day[short, None]).mean()
    # neighbour fidelity: share of the 15 nearest 2-D neighbours that are among the 15 nearest in the 64-d trading space
    fid = np.array([len(np.intersect1d(N[i], nn64[i], assume_unique=True)) for i in range(len(N))]) / KNN
    lab = np.load(f"{OUT}/market_cluster_labels_{tag}.npy") if os.path.exists(f"{OUT}/market_cluster_labels_{tag}.npy") else np.full(len(XM), -1)
    inl = lab >= 0
    pur = []
    for c in range(lab.max() + 1):
        m = lab == c; pur.append((m.sum(), collections.Counter(ci[m]).most_common(1)[0][1] / m.sum()))
    pur = np.array(pur) if pur else np.zeros((0, 2))
    row = dict(tag=tag, same_category=round(float(same_cat), 4), same_template=round(float(same_tpl), 4), same_day_short_updown=round(float(same_day), 4),
               neighbour_fidelity=round(float(fid.mean()), 4), islands=int(lab.max() + 1), noise_share=round(float((~inl).mean()), 4),
               fills_in_islands=round(float(fills[inl].sum() / fills.sum()), 4), island_purity=round(float((pur[:, 0] * pur[:, 1]).sum() / pur[:, 0].sum()), 4) if len(pur) else None)
    log(row)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--variant", required=True); p.add_argument("--jobs", type=int, default=36)
    p.add_argument("--tags", default="cat000,cat050"); p.add_argument("--min-size", type=int, default=150); a = p.parse_args()
    e, ew, em, mult, nW, nM, ZM, meta, ci = load_base()
    fills, deg = e["fillsM"], e["degM"]
    log("markets", nM, "wallets", nW, "variant", a.variant)
    if a.variant == "knn64":
        from pynndescent import NNDescent
        t = time.time()
        idx = NNDescent(ZM, metric="euclidean", n_neighbors=KNN + 1, n_jobs=a.jobs, random_state=0).neighbor_graph[0][:, 1:]
        np.save(f"{OUT}/knn64.npy", idx); log("64-d neighbours in", round(time.time() - t), "s")
    elif a.variant == "render":   # light maps of layouts made before this script (cat000, cat050)
        for tag in a.tags.split(","):
            f = {"cat000": "layout_anchor.npz", "cat050": "layout_anchor_cat.npz"}.get(tag, f"layout_anchor_{tag}.npz")
            lay = np.load(f"{OUT}/{f}"); render_light(lay["xy"], int(lay["nW"]), nM, e, ew, em, mult, ci, tag)
    elif a.variant == "metrics":
        nn64 = np.load(f"{OUT}/knn64.npy")
        rows = []
        for tag in a.tags.split(","):
            f = {"cat000": "layout_anchor.npz", "cat050": "layout_anchor_cat.npz"}.get(tag, f"layout_anchor_{tag}.npz")
            if tag in ("cat000", "cat050"):
                src = {"cat000": "market_cluster_labels.npy", "cat050": "market_cluster_labels_cat.npy"}[tag]
                if not os.path.exists(f"{OUT}/market_cluster_labels_{tag}.npy"): os.symlink(src, f"{OUT}/market_cluster_labels_{tag}.npy")
            lay = np.load(f"{OUT}/{f}"); rows.append(metrics(tag, lay["xy"], int(lay["nW"]), meta, ci, fills, nn64))
        df = pd.DataFrame(rows); df.to_csv(f"{OUT}/map_metrics.csv", index=False); print(df.to_string(index=False))
    else:
        tag = a.variant
        XM = embed(tag, ZM, ci, meta, e, em, nM, a.jobs)
        xy = anchor_wallets(XM, ew, em, mult, nW, nM)
        np.savez(f"{OUT}/layout_anchor_{tag}.npz", xy=xy, nW=nW, nM=nM); log("layout saved", tag)
        lab, df = islands(XM, meta, fills, deg, a.min_size)
        df.to_csv(f"{OUT}/market_clusters_{tag}.csv", index=False); np.save(f"{OUT}/market_cluster_labels_{tag}.npy", lab)
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
        print(df[["cluster", "n", "fills", "top_cat", "purity", "cats", "updown_share", "day_p10", "day_p90", "interval_min"]].head(30).to_string(index=False))
        print(); print(df[["cluster", "n", "tokens", "examples"]].head(30).to_string(index=False))
        centroids(xy, nW, lab, fills, tag)
        render_light(xy, nW, nM, e, ew, em, mult, ci, tag)
        log("DONE", tag)
