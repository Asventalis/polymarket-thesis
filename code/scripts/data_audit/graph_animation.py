"""Animated map of the wallet-market transaction graph over the whole collected window.

Every on-chain fill leg (wallet, market, timestamp) is an edge event of the bipartite graph
wallets x markets.  The script

  stage E  extracts a compact, time-sorted edge-event table from the partitioned transactions
           (wallet index, market index, unix seconds, usdc, role) and the two id tables;
  stage S  computes exact graph statistics: degrees, edge multiplicities, connected components
           of the whole graph and of every hourly slice, growth of the graph over time;
  stage L  lays the 800k nodes out with a large-graph technique: truncated SVD of the
           (log-weighted, row-normalised) biadjacency matrix gives wallets and markets a common
           64-d latent space; UMAP maps the two together to the plane;
  stage F  renders one frame per time bin (default 10 min): every fill lights its edge and its
           two end nodes, both fade exponentially (edge half-life 20 min, node half-life 60 min)
           over a faint static drawing of the complete graph; a side panel shows the clock, the
           counters of the active sub-graph and its connected components, and the activity curve;
  stage V  encodes the frames to an H.264 mp4 with the ffmpeg binary of imageio-ffmpeg.

usage (server, CPU only):
  ./run_dev.sh scripts/data_audit/graph_animation.py [--stage E,S,L,F,V] [--bin-min 10]
      [--jobs 32] [--width 1920] [--height 1080] [--fps 30] [--force]
outputs in /data/v.pampinella/data_audit/graphviz/
"""
import argparse, json, os, sys, time, math
sys.path.insert(0, "/data/v.pampinella/dm_pkgs")
import numpy as np

OUT = "/data/v.pampinella/data_audit/graphviz"
TX = "/data/v.pampinella/agy_project/data/derived/partitioned/transactions/**/*.parquet"
MARKETS = "/data/v.pampinella/agy_project/data/universe_markets.parquet"
os.makedirs(OUT, exist_ok=True)

# thesis palette
VIZ = dict(A="#2a78d6", B="#eb6834", C="#1baf7a", D="#eda100", INK="#52514e")
BG_DARK = np.array([0.055, 0.06, 0.075])
WALLET_RGB = np.array([0.37, 0.62, 0.95])      # blue
MARKET_RGB = np.array([0.98, 0.55, 0.25])      # orange
EDGE_RGB = np.array([0.95, 0.92, 0.72])        # warm white
CAT_ORDER = ["crypto", "sports", "esports", "politics", "weather", "economy_tech", "culture", "other"]
CAT_HEX = {"crypto": "#f2c230", "sports": "#f06a1e", "esports": "#e03a4e", "politics": "#d957c8", "weather": "#ff9e9e",
           "economy_tech": "#c8a070", "culture": "#9c2b4a", "other": "#8c8c8c"}
CAT_RGB = {c: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]) for c, h in CAT_HEX.items()}


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f:
        f.write(msg + "\n")


# ────────────────────────────────────────────────────────────────────────────── stage E
def stage_E(args):
    import duckdb
    if os.path.exists(f"{OUT}/events.npz") and not args.force:
        log("E: events.npz exists, skip"); return
    con = duckdb.connect(); con.execute(f"set threads={args.jobs}")
    con.execute(f"create table w as select wallet, row_number() over (order by wallet) - 1 as wi from (select distinct wallet from read_parquet('{TX}', hive_partitioning=1))")
    con.execute(f"create table m as select market_id, row_number() over (order by market_id) - 1 as mi from (select distinct market_id from read_parquet('{TX}', hive_partitioning=1))")
    con.execute(f"copy w to '{OUT}/wallets.parquet'"); con.execute(f"copy m to '{OUT}/markets.parquet'")
    log("E: ids written", con.execute("select count(*) from w").fetchone(), con.execute("select count(*) from m").fetchone())
    df = con.execute(f"""
        select w.wi::int as wi, m.mi::int as mi, epoch(t.timestamp)::int as ts,
               t.total_usdc::float as usdc, (t.role = 'TAKER')::tinyint as taker
        from read_parquet('{TX}', hive_partitioning=1) t join w using (wallet) join m using (market_id)
        order by ts""").to_arrow_table()
    ev = {k: df.column(k).to_numpy() for k in ["wi", "mi", "ts", "usdc", "taker"]}
    np.savez(f"{OUT}/events.npz", **ev)
    log("E: events", len(ev["ts"]), "from", ev["ts"][0], "to", ev["ts"][-1])
    # market metadata for the colouring of the static map
    try:
        meta = con.execute(f"""select m.mi, u.category, u.question from m left join read_parquet('{MARKETS}') u
                               on lower(u.market_id) = lower(m.market_id) order by mi""").df()
        meta.to_parquet(f"{OUT}/market_meta.parquet")
        log("E: market meta columns", list(meta.columns), "categories", meta["category"].nunique())
    except Exception as e:
        log("E: market meta not available:", repr(e))


def load_events():
    z = np.load(f"{OUT}/events.npz")
    return {k: z[k] for k in z.files}


# ────────────────────────────────────────────────────────────────────────────── stage S
def components(rows, cols, nW, nM):
    """Connected components of the bipartite graph with edges rows (wallets) - cols (markets)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = nW + nM
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols + nW)), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    return k, lab


def stage_S(args):
    import pandas as pd
    if os.path.exists(f"{OUT}/stats.json") and not args.force:
        log("S: stats.json exists, skip"); return
    ev = load_events(); wi, mi, ts = ev["wi"], ev["mi"], ev["ts"]
    nW, nM = int(wi.max()) + 1, int(mi.max()) + 1
    key = wi.astype(np.int64) * nM + mi
    uniq, first_idx, mult = np.unique(key, return_index=True, return_counts=True)
    E = len(uniq)
    ew, em = (uniq // nM).astype(np.int32), (uniq % nM).astype(np.int32)
    degW = np.bincount(ew, minlength=nW); degM = np.bincount(em, minlength=nM)
    fillsW = np.bincount(wi, minlength=nW); fillsM = np.bincount(mi, minlength=nM)
    usdcW = np.bincount(wi, weights=ev["usdc"], minlength=nW); usdcM = np.bincount(mi, weights=ev["usdc"], minlength=nM)
    k, lab = components(ew, em, nW, nM)
    sizes = np.bincount(lab); order = np.argsort(-sizes)
    giant = order[0]
    st = dict(
        n_events=int(len(ts)), n_wallets=nW, n_markets=nM, n_edges=E,
        n_taker_fills=int(ev["taker"].sum()),
        density_bipartite=E / (nW * nM),
        mean_degree_wallet=float(degW.mean()), mean_degree_market=float(degM.mean()),
        median_degree_wallet=float(np.median(degW)), median_degree_market=float(np.median(degM)),
        wallets_degree_1=int((degW == 1).sum()), markets_degree_1=int((degM == 1).sum()),
        max_degree_wallet=int(degW.max()), max_degree_market=int(degM.max()),
        edge_multiplicity_quantiles={q: float(np.quantile(mult, q)) for q in [0.5, 0.9, 0.99, 0.999]},
        edges_single_fill=int((mult == 1).sum()), max_multiplicity=int(mult.max()),
        n_components=int(k), giant_nodes=int(sizes[giant]),
        giant_wallets=int((lab[:nW] == giant).sum()), giant_markets=int((lab[nW:] == giant).sum()),
        giant_share_nodes=float(sizes[giant] / (nW + nM)),
        giant_share_events=float(((lab[wi] == giant)).mean()),
        component_sizes_top20=[int(s) for s in sizes[order[:20]]],
        components_size_2=int((sizes == 2).sum()), components_size_le5=int((sizes <= 5).sum()),
        gini_fills_wallet=gini(fillsW), gini_usdc_wallet=gini(usdcW), gini_fills_market=gini(fillsM),
        top1pct_wallets_share_fills=float(np.sort(fillsW)[-max(1, nW // 100):].sum() / fillsW.sum()),
        top1pct_markets_share_fills=float(np.sort(fillsM)[-max(1, nM // 100):].sum() / fillsM.sum()),
    )
    # k-core of the bipartite graph (igraph)
    try:
        import igraph as ig
        g = ig.Graph(n=nW + nM, edges=np.column_stack([ew, em + nW]).tolist(), directed=False)
        core = np.array(g.coreness())
        st["max_core"] = int(core.max()); st["nodes_in_max_core"] = int((core == core.max()).sum())
        st["nodes_core_ge_10"] = int((core >= 10).sum())
        np.save(f"{OUT}/coreness.npy", core.astype(np.int32))
        log("S: k-core done", st["max_core"], st["nodes_in_max_core"])
    except Exception as e:
        log("S: igraph coreness failed:", repr(e))
    json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1)
    np.savez(f"{OUT}/edges.npz", ew=ew, em=em, mult=mult.astype(np.int32), first_ts=ts[first_idx], label=lab.astype(np.int32),
             degW=degW.astype(np.int32), degM=degM.astype(np.int32), fillsW=fillsW.astype(np.int32), fillsM=fillsM.astype(np.int32),
             usdcW=usdcW.astype(np.float32), usdcM=usdcM.astype(np.float32))
    pd.DataFrame({"size": sizes[order]}).to_csv(f"{OUT}/component_sizes.csv", index=False)
    log("S: global stats", {k_: v for k_, v in st.items() if not isinstance(v, (list, dict))})

    # degree distributions (log-binned)
    rows = []
    for name, d in [("wallet_degree", degW), ("market_degree", degM), ("wallet_fills", fillsW), ("market_fills", fillsM), ("edge_multiplicity", mult)]:
        vals, cnt = np.unique(d, return_counts=True)
        for v, c in zip(vals, cnt):
            rows.append((name, int(v), int(c)))
    pd.DataFrame(rows, columns=["quantity", "value", "count"]).to_csv(f"{OUT}/degree_distributions.csv", index=False)

    # hourly slices: active nodes/edges, new edges, components of the hourly graph
    t0 = int(ts[0]) // 3600 * 3600
    hour = (ts - t0) // 3600
    nH = int(hour.max()) + 1
    bounds = np.searchsorted(hour, np.arange(nH + 1))
    seenW = np.zeros(nW, bool); seenM = np.zeros(nM, bool)
    edge_id = np.searchsorted(uniq, key)
    seenE = np.zeros(E, bool)
    hrows = []
    for h in range(nH):
        a, b = bounds[h], bounds[h + 1]
        if b <= a:
            hrows.append((t0 + h * 3600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)); continue
        w_, m_, e_ = wi[a:b], mi[a:b], edge_id[a:b]
        ue = np.unique(e_); uw = np.unique(w_); um = np.unique(m_)
        newE = int((~seenE[ue]).sum()); seenE[ue] = True
        newW = int((~seenW[uw]).sum()); seenW[uw] = True
        newM = int((~seenM[um]).sum()); seenM[um] = True
        kk, ll = components(ew[ue], em[ue], nW, nM)
        # components among touched nodes only (untouched nodes are isolated singletons)
        touched = np.concatenate([uw, um + nW]); lt = ll[touched]
        cs = np.bincount(lt); cs = cs[cs > 0]
        hrows.append((t0 + h * 3600, b - a, int(ev["taker"][a:b].sum()), len(uw), len(um), len(ue), newW, newM, newE,
                      len(cs), float(cs.max() / len(touched)), float(ev["usdc"][a:b].sum())))
    pd.DataFrame(hrows, columns=["t", "events", "taker_fills", "wallets", "markets", "edges", "new_wallets", "new_markets",
                                 "new_edges", "components", "giant_share", "usdc"]).to_csv(f"{OUT}/hourly.csv", index=False)
    log("S: hourly slices done", nH)

    # daily cumulative growth of the graph
    day = (ts - t0) // 86400; nD = int(day.max()) + 1
    db = np.searchsorted(day, np.arange(nD + 1))
    seenW[:] = False; seenM[:] = False; seenE[:] = False; drows = []
    for d in range(nD):
        a, b = db[d], db[d + 1]
        seenW[wi[a:b]] = True; seenM[mi[a:b]] = True; seenE[edge_id[a:b]] = True
        kk, ll = components(ew[seenE], em[seenE], nW, nM)
        touched = np.concatenate([np.flatnonzero(seenW), np.flatnonzero(seenM) + nW]); lt = ll[touched]
        cs = np.bincount(lt); cs = cs[cs > 0]
        drows.append((t0 + d * 86400, int(seenW.sum()), int(seenM.sum()), int(seenE.sum()), len(cs), float(cs.max() / len(touched))))
    pd.DataFrame(drows, columns=["t", "cum_wallets", "cum_markets", "cum_edges", "components", "giant_share"]).to_csv(f"{OUT}/daily_growth.csv", index=False)
    log("S: daily growth done")


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ────────────────────────────────────────────────────────────────────────────── stage L
def stage_L(args):
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    if os.path.exists(f"{OUT}/layout.npz") and not args.force:
        log("L: layout.npz exists, skip"); return
    e = np.load(f"{OUT}/edges.npz"); ew, em, mult = e["ew"], e["em"], e["mult"]
    nW, nM = int(ew.max()) + 1, int(em.max()) + 1
    w = np.log1p(mult.astype(np.float64))
    A = sp.csr_matrix((w, (ew, em)), shape=(nW, nM))
    # idf-like column scaling (very popular markets would otherwise dominate), then row L2 norm
    df_ = np.bincount(em, minlength=nM).astype(np.float64)
    A = A @ sp.diags(np.log((1 + nW) / (1 + df_)))
    rn = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()); rn[rn == 0] = 1
    A = sp.diags(1 / rn) @ A
    k = args.svd_dim
    t = time.time(); U, S, Vt = svds(A.astype(np.float32), k=k); log("L: svds", k, "in", round(time.time() - t), "s; top sv", np.round(S[::-1][:5], 3))
    ZW = U * np.sqrt(S); ZM = Vt.T * np.sqrt(S)
    Z = np.vstack([ZW, ZM]).astype(np.float32)
    np.save(f"{OUT}/svd_joint.npy", Z)
    import umap
    t = time.time()
    red = umap.UMAP(n_components=2, n_neighbors=args.umap_neighbors, min_dist=0.05, metric="cosine",
                    low_memory=True, n_jobs=args.jobs, random_state=None, verbose=True)
    XY = red.fit_transform(Z).astype(np.float32)
    log("L: umap in", round(time.time() - t), "s")
    np.savez(f"{OUT}/layout.npz", xy=XY, nW=nW, nM=nM)
    log("L: layout saved", XY.shape, XY.min(0), XY.max(0))


# ────────────────────────────────────────────────────────────────────────────── rendering
def to_pixels(xy, W, H, margin=0.03):
    lo, hi = np.quantile(xy, 0.003, axis=0), np.quantile(xy, 0.997, axis=0)
    span = (hi - lo).max()
    c = (lo + hi) / 2
    s = min(W, H) * (1 - 2 * margin) / span
    px = (xy[:, 0] - c[0]) * s + W / 2
    py = (c[1] - xy[:, 1]) * s + H / 2
    return np.clip(px, 0, W - 1).astype(np.float32), np.clip(py, 0, H - 1).astype(np.float32)


def raster_lines(x0, y0, x1, y1, w, W, H, max_samples=1000):
    """Accumulate weighted line segments into an H x W float image (additive)."""
    img = np.zeros(W * H, np.float32)
    if len(x0) == 0: return img.reshape(H, W)
    length = np.hypot(x1 - x0, y1 - y0)
    n = np.clip((length / 1.5).astype(np.int64) + 2, 2, max_samples)
    tot = int(n.sum())
    rep = np.repeat(np.arange(len(n)), n)
    # position along each segment: 0..1
    start = np.cumsum(n) - n
    tpos = (np.arange(tot) - start[rep]) / (n[rep] - 1)
    xs = x0[rep] + (x1 - x0)[rep] * tpos; ys = y0[rep] + (y1 - y0)[rep] * tpos
    idx = np.clip(ys, 0, H - 1).astype(np.int64) * W + np.clip(xs, 0, W - 1).astype(np.int64)
    img = np.bincount(idx, weights=(w / n)[rep], minlength=W * H).astype(np.float32)
    return img.reshape(H, W)


def splat(px, py, w, W, H, sigma):
    from scipy.ndimage import gaussian_filter
    img = np.bincount(np.clip(py, 0, H - 1).astype(np.int64) * W + np.clip(px, 0, W - 1).astype(np.int64), weights=w, minlength=W * H).astype(np.float32).reshape(H, W)
    return gaussian_filter(img, sigma) if sigma > 0 else img


def market_layer(px, py, w, W, H, sigma, catM, gain, gamma, strength):
    """Sum over categories of the splat of the markets of that category, in the category colour."""
    out = np.zeros((H, W, 3), np.float32)
    for ci, c in enumerate(CAT_ORDER):
        m = catM == ci
        if not m.any() or w[m].sum() == 0: continue
        out += tone(splat(px[m], py[m], w[m], W, H, sigma), gain, gamma)[..., None] * CAT_RGB[c] * strength
    return out


def tone(img, gain, gamma=0.5):
    return np.clip((img * gain) ** gamma, 0, 1)


G = {}  # globals shared by fork with the frame workers


def load_catM(nM):
    import pandas as pd
    meta = pd.read_parquet(f"{OUT}/market_meta.parquet")
    cat = np.full(nM, CAT_ORDER.index("other"), np.int8)
    idx = {c: i for i, c in enumerate(CAT_ORDER)}
    cat[meta["mi"].to_numpy()] = [idx.get(c, idx["other"]) for c in meta["category"].fillna("other")]
    return cat


def render_background(args):
    """Static map: all edges faint, nodes as splats; saved as PNG and reused under every frame."""
    W, H = args.map_w, args.height
    lay = np.load(f"{OUT}/{args.layout}"); xy = lay["xy"]; nW, nM = int(lay["nW"]), int(lay["nM"])
    px, py = to_pixels(xy, W, H)
    e = np.load(f"{OUT}/edges.npz"); ew, em, mult = e["ew"], e["em"], e["mult"]
    t = time.time()
    # subsample edges for the background density (weighted), all nodes drawn
    rng = np.random.default_rng(0)
    keep = rng.random(len(ew)) < min(1.0, args.bg_edges / len(ew))
    L = raster_lines(px[ew[keep]], py[ew[keep]], px[em[keep] + nW], py[em[keep] + nW], np.log1p(mult[keep]).astype(np.float32), W, H)
    log("bg: edges rasterised", int(keep.sum()), "in", round(time.time() - t), "s")
    Wn = splat(px[:nW], py[:nW], np.log1p(e["fillsW"]).astype(np.float32), W, H, 0.8)
    Mn = splat(px[nW:], py[nW:], np.log1p(e["fillsM"]).astype(np.float32), W, H, 1.0)
    catM = load_catM(nM)
    rgb = np.ones((H, W, 3), np.float32) * BG_DARK
    rgb += tone(L, 1.0 / np.quantile(L[L > 0], 0.995), 0.6)[..., None] * EDGE_RGB * 0.16
    rgb += tone(Wn, 1.0 / np.quantile(Wn[Wn > 0], 0.995), 0.6)[..., None] * WALLET_RGB * 0.35
    rgb += market_layer(px[nW:], py[nW:], np.log1p(e["fillsM"]).astype(np.float32), W, H, 1.0, catM, 1.0 / np.quantile(Mn[Mn > 0], 0.995), 0.6, 0.45)
    rgb = np.clip(rgb, 0, 1)
    np.save(f"{OUT}/background.npy", rgb.astype(np.float32))
    from PIL import Image
    Image.fromarray((rgb * 255).astype(np.uint8)).save(f"{OUT}/static_map.png")
    # static map, light background, for the thesis
    light = np.ones((H, W, 3), np.float32)
    light -= tone(L, 1.0 / np.quantile(L[L > 0], 0.995), 0.6)[..., None] * np.array([0.35, 0.35, 0.35]) * 0.5
    light -= tone(Wn, 1.0 / np.quantile(Wn[Wn > 0], 0.995), 0.6)[..., None] * (1 - np.array([0.165, 0.47, 0.84])) * 0.9
    light -= tone(Mn, 1.0 / np.quantile(Mn[Mn > 0], 0.995), 0.6)[..., None] * (1 - np.array([0.92, 0.41, 0.2])) * 0.9
    Image.fromarray((np.clip(light, 0, 1) * 255).astype(np.uint8)).save(f"{OUT}/static_map_light.png")
    log("bg: done")


def frame_worker(fr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    a = G["args"]; W, H = a.map_w, a.height
    ev = G["ev"]; ts = ev["ts"]; px, py = G["px"], G["py"]; nW = G["nW"]; bg = G["bg"]
    binsec = a.bin_min * 60; t0 = G["t0"]
    out = []
    for f in fr:
        path = f"{OUT}/{a.frames_dir}/f{f:05d}.png"
        if os.path.exists(path) and not a.force: continue
        t_end = t0 + (f + 1) * binsec
        lookback = int(a.edge_half_life * 60 * 8)
        a0, a1 = np.searchsorted(ts, t_end - lookback), np.searchsorted(ts, t_end)
        wi, mi, tt = ev["wi"][a0:a1], ev["mi"][a0:a1], ts[a0:a1]
        age = (t_end - tt).astype(np.float32)
        we = 0.5 ** (age / (a.edge_half_life * 60)); wn = 0.5 ** (age / (a.node_half_life * 60))
        key = wi.astype(np.int64) * G["nM"] + mi
        uk, inv = np.unique(key, return_inverse=True)
        ewt = np.bincount(inv, weights=we).astype(np.float32)
        ew_, em_ = (uk // G["nM"]).astype(np.int64), (uk % G["nM"]).astype(np.int64)
        act = ewt > 0.02
        L = raster_lines(px[ew_[act]], py[ew_[act]], px[em_[act] + nW], py[em_[act] + nW], np.log1p(ewt[act]), W, H)
        wW = np.bincount(wi, weights=wn, minlength=nW).astype(np.float32); wM = np.bincount(mi, weights=wn, minlength=G["nM"]).astype(np.float32)
        Wn = splat(px[:nW], py[:nW], np.log1p(wW), W, H, 1.2)
        rgb = bg.copy()
        rgb += tone(L, G["gainL"] * 0.3, 0.65)[..., None] * EDGE_RGB * 0.8
        rgb += tone(Wn, G["gainW"] * 0.5, 0.65)[..., None] * WALLET_RGB * 0.9
        rgb += market_layer(px[nW:], py[nW:], np.log1p(wM), W, H, 1.6, G["catM"], G["gainM"] * 0.6, 0.65, 1.0)
        rgb = np.clip(rgb, 0, 1)
        # counters of the active sub-graph (events of the current bin and of the last hour)
        b0 = np.searchsorted(ts, t_end - binsec); h0 = np.searchsorted(ts, t_end - 3600)
        n_bin = a1 - b0
        wh, mh = ev["wi"][h0:a1], ev["mi"][h0:a1]
        keyh = np.unique(wh.astype(np.int64) * G["nM"] + mh)
        ewh, emh = (keyh // G["nM"]).astype(np.int32), (keyh % G["nM"]).astype(np.int32)
        uw, um = np.unique(wh), np.unique(mh)
        if len(keyh):
            k, lab = components(ewh, emh, nW, G["nM"])
            touched = np.concatenate([uw, um + nW]); cs = np.bincount(lab[touched]); cs = cs[cs > 0]
            ncomp, giant = len(cs), cs.max() / len(touched)
        else:
            ncomp, giant = 0, 0.0
        usdc_bin = float(ev["usdc"][b0:a1].sum())
        # compose the frame
        fig = plt.figure(figsize=(a.width / 100, a.height / 100), dpi=100, facecolor=tuple(BG_DARK))
        gs = GridSpec(6, 2, width_ratios=[W, a.width - W], figure=fig, left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax = fig.add_subplot(gs[:, 0]); ax.imshow(rgb, interpolation="bilinear"); ax.axis("off")
        import matplotlib.patheffects as pe
        for (lx, ly, name) in G["labels"]:
            ax.text(lx, ly, name, fontsize=7.6, color="#f4f1e8", ha="center", va="center", family="DejaVu Sans",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=tuple(BG_DARK))], zorder=5)
        side = fig.add_subplot(gs[:4, 1]); side.axis("off"); side.set_facecolor(tuple(BG_DARK))
        tstr = time.strftime("%a %d %b %Y  %H:%M", time.gmtime(t_end)) + " UTC"
        txt = [("Polymarket, wallets × markets", 15, "#ffffff", "bold"),
               ("every on-chain fill lights an edge", 10.5, "#b8b6b0", None),
               ("", 6, None, None),
               (tstr, 13.5, "#ffffff", "bold"),
               ("", 6, None, None),
               (f"last {a.bin_min} min", 10.5, "#b8b6b0", None),
               (f"{n_bin:,} fill legs   {usdc_bin / 1e3:,.0f} k USDC", 12.5, "#f2ecc0", None),
               ("", 6, None, None),
               ("active sub-graph, last 60 min", 10.5, "#b8b6b0", None),
               (f"{len(uw):,} wallets", 12.5, "#8fc1ff", None),
               (f"{len(um):,} markets", 12.5, "#ffb57a", None),
               (f"{len(keyh):,} edges", 12.5, "#f2ecc0", None),
               (f"{ncomp:,} connected components", 12.5, "#ffffff", None),
               (f"giant component {giant * 100:.0f}% of the active nodes", 12.5, "#ffffff", None),
               ("", 6, None, None),
               (f"whole window: {G['st']['n_wallets']:,} wallets, {G['st']['n_markets']:,} markets,", 9.5, "#8a8884", None),
               (f"{G['st']['n_edges']:,} edges, {G['st']['n_events']:,} fill legs,", 9.5, "#8a8884", None),
               (f"{G['st']['n_components']:,} connected component(s)", 9.5, "#8a8884", None),
               (f"edge half-life {a.edge_half_life:.0f} min, node half-life {a.node_half_life:.0f} min", 9.5, "#8a8884", None),
               (("layout: SVD (64-d) + UMAP of the markets," if "anchor" in a.layout else "layout: truncated SVD (64-d)"), 9.5, "#8a8884", None),
               (("wallets anchored to their markets" if "anchor" in a.layout else "+ UMAP of the bipartite graph"), 9.5, "#8a8884", None),
               ("", 6, None, None),
               ("\u25cf wallets", 10.5, "#8fc1ff", None)] + [("\u25cf markets: " + c.replace("_", " & "), 10.5, CAT_HEX[c], None) for c in CAT_ORDER]
        y = 0.96
        for s, fs, col, wt in txt:
            if s: side.text(0.06, y, s, transform=side.transAxes, fontsize=fs, color=col, fontweight=wt or "normal", va="top", family="DejaVu Sans")
            y -= fs * 0.0017 + 0.011
        # activity curve with cursor
        axc = fig.add_subplot(gs[4:, 1]); axc.set_facecolor(tuple(BG_DARK))
        hc = G["hourly"]
        axc.fill_between(hc[:, 0], 0, hc[:, 1] / 1e3, color="#f2ecc0", alpha=0.25, lw=0)
        axc.plot(hc[:, 0], hc[:, 1] / 1e3, color="#f2ecc0", lw=0.8)
        axc.axvline(t_end, color="#ffffff", lw=1.0)
        axc.set_xlim(hc[0, 0], hc[-1, 0] + 3600); axc.set_ylim(0, hc[:, 1].max() / 1e3 * 1.15)
        days = np.arange(G["t0"], hc[-1, 0] + 3600, 86400 * 3)
        axc.set_xticks(days); axc.set_xticklabels([time.strftime("%d %b", time.gmtime(d)) for d in days], color="#b8b6b0", fontsize=8)
        axc.tick_params(colors="#b8b6b0", labelsize=8, length=2); axc.set_ylabel("fill legs per hour (k)", color="#b8b6b0", fontsize=8.5)
        for sp_ in axc.spines.values(): sp_.set_color("#3a3a40")
        pos = axc.get_position(); axc.set_position([pos.x0 + 0.02, pos.y0 + 0.06, pos.width - 0.05, pos.height - 0.12])
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
        out.append(f)
    return out


def stage_F(args):
    os.makedirs(f"{OUT}/{args.frames_dir}", exist_ok=True)
    if not os.path.exists(f"{OUT}/background.npy") or args.force:
        render_background(args)
    import pandas as pd
    ev = load_events(); lay = np.load(f"{OUT}/{args.layout}"); nW, nM = int(lay["nW"]), int(lay["nM"])
    px, py = to_pixels(lay["xy"], args.map_w, args.height)
    hourly = pd.read_csv(f"{OUT}/hourly.csv")[["t", "events"]].to_numpy(dtype=np.float64)
    st = json.load(open(f"{OUT}/stats.json"))
    t0 = int(ev["ts"][0]) // 86400 * 86400
    nF = int(math.ceil((int(ev["ts"][-1]) - t0) / (args.bin_min * 60)))
    G.update(args=args, ev=ev, px=px, py=py, nW=nW, nM=nM, bg=np.load(f"{OUT}/background.npy"), t0=t0, hourly=hourly, st=st, catM=load_catM(nM), labels=[])
    if args.labels:
        names = json.load(open(args.labels)); cl = np.load(f"{OUT}/{args.clusters}")
        cand = []
        for k, name in names.items():
            m = np.flatnonzero(cl == int(k))
            if len(m): cand.append((len(m), float(np.median(px[nW + m])), float(np.median(py[nW + m])), name))
        boxes = []
        for n_, lx, ly, name in sorted(cand, reverse=True):
            w_, h_ = 5.6 * len(name) + 10, 18.0
            box = (lx - w_ / 2, ly - h_ / 2, lx + w_ / 2, ly + h_ / 2)
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes): continue
            boxes.append(box); G["labels"].append((lx, ly, name))
        log("F: labels", len(G["labels"]))
    # gains calibrated on a busy reference frame so that brightness is comparable across frames
    ref = nF // 2; t_end = t0 + (ref + 1) * args.bin_min * 60; ts = ev["ts"]
    a0, a1 = np.searchsorted(ts, t_end - args.edge_half_life * 60 * 8), np.searchsorted(ts, t_end)
    age = (t_end - ts[a0:a1]).astype(np.float32); we = 0.5 ** (age / (args.edge_half_life * 60)); wn = 0.5 ** (age / (args.node_half_life * 60))
    key = ev["wi"][a0:a1].astype(np.int64) * nM + ev["mi"][a0:a1]; uk, inv = np.unique(key, return_inverse=True)
    ewt = np.bincount(inv, weights=we).astype(np.float32); ew_, em_ = uk // nM, uk % nM
    L = raster_lines(px[ew_], py[ew_], px[em_ + nW], py[em_ + nW], np.log1p(ewt), args.map_w, args.height)
    Wn = splat(px[:nW], py[:nW], np.log1p(np.bincount(ev["wi"][a0:a1], weights=wn, minlength=nW)).astype(np.float32), args.map_w, args.height, 1.2)
    Mn = splat(px[nW:], py[nW:], np.log1p(np.bincount(ev["mi"][a0:a1], weights=wn, minlength=nM)).astype(np.float32), args.map_w, args.height, 1.6)
    G["gainL"] = 1.0 / np.quantile(L[L > 0], 0.99); G["gainW"] = 1.0 / np.quantile(Wn[Wn > 0], 0.995); G["gainM"] = 1.0 / np.quantile(Mn[Mn > 0], 0.995)
    log("F: frames", nF, "bin", args.bin_min, "min; gains", G["gainL"], G["gainW"], G["gainM"])
    frames = list(range(nF)) if args.frames is None else [int(x) for x in args.frames.split(",")]
    chunks = [frames[i::args.jobs] for i in range(args.jobs)]
    from multiprocessing import get_context
    t = time.time()
    with get_context("fork").Pool(args.jobs) as pool:
        done = sum(len(r) for r in pool.map(frame_worker, chunks))
    log("F: rendered", done, "frames in", round(time.time() - t), "s")


def stage_V(args):
    import subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = f"{OUT}/polymarket_graph_{args.bin_min}min{args.tag}.mp4"
    cmd = [ff, "-y", "-framerate", str(args.fps), "-i", f"{OUT}/{args.frames_dir}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "medium", "-movflags", "+faststart", out]
    log("V:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log("V: ffmpeg exit", r.returncode, r.stderr[-400:])
    log("ALL DONE", out, os.path.getsize(out) if os.path.exists(out) else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="E,S,L,F,V"); p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=32); p.add_argument("--bin-min", type=int, default=10)
    p.add_argument("--edge-half-life", type=float, default=20.0); p.add_argument("--node-half-life", type=float, default=60.0)
    p.add_argument("--width", type=int, default=1920); p.add_argument("--height", type=int, default=1080); p.add_argument("--map-w", type=int, default=1440)
    p.add_argument("--fps", type=int, default=30); p.add_argument("--svd-dim", type=int, default=64); p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--bg-edges", type=int, default=6_000_000); p.add_argument("--layout", default="layout.npz"); p.add_argument("--labels", default=None); p.add_argument("--clusters", default="market_cluster_labels.npy"); p.add_argument("--frames-dir", default="frames"); p.add_argument("--tag", default=""); p.add_argument("--frames", default=None, help="comma list of frame ids (test)")
    args = p.parse_args()
    for s in args.stage.split(","):
        log(f"=== stage {s} ===")
        {"E": stage_E, "S": stage_S, "L": stage_L, "F": stage_F, "V": stage_V}[s](args)
