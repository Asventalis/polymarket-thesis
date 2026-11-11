"""Build the cross-horizon pgfplots figures and LaTeX tables of the data-mining chapter from
the per-horizon CSV outputs of scripts/data_audit/datamining.py (stdlib only).

usage: python3 make_cross_horizon.py <dir with subdirs 1m 15m 1h 6h 12h> <thesis figures/dm dir>
Writes fig_dm_spectrum.tex, fig_dm_knn.tex and tables_cross_horizon.tex (to be pasted / input)."""
import csv, json, sys, os, math
SRC, OUT = sys.argv[1], sys.argv[2]
HS = [("1m", "1 min"), ("15m", "15 min"), ("1h", "1 h"), ("6h", "6 h"), ("12h", "12 h")]
COL = {"1m": "vizA", "15m": "vizB", "1h": "vizC", "6h": "vizD", "12h": "vizInk"}
MARK = {"1m": "*", "15m": "square*", "1h": "triangle*", "6h": "diamond*", "12h": "pentagon*"}
def rows(h, name):
    p = os.path.join(SRC, h, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None
have = [h for h, _ in HS if rows(h, "separability.csv") is not None and rows(h, "knn.csv") is not None]
print("horizons with results:", have)

# ── figure: cumulative explained variance ──────────────────────────────────────────────
with open(os.path.join(OUT, "fig_dm_spectrum.tex"), "w") as f:
    f.write("\\begin{tikzpicture}\n\\begin{axis}[thesis, width=0.62\\textwidth, height=5.2cm, xlabel={number of principal components}, ylabel={cumulative share of variance}, xmin=0, xmax=40, ymin=0.15, ymax=1.02, legend pos=south east, legend columns=1]\n")
    for h, lab in HS:
        r = rows(h, "pca_spectrum.csv")
        if not r: continue
        pts = " ".join(f"({x['component']},{float(x['cum']):.4f})" for x in r if int(x["component"]) <= 40)
        f.write(f"\\addplot[{COL[h]}, thick, mark=none] coordinates {{{pts}}};\n\\addlegendentry{{{lab}}}\n")
    f.write("\\addplot[vizGrid, dashed, thin, mark=none] coordinates {(0,0.8) (40,0.8)};\n\\end{axis}\n\\end{tikzpicture}\n")

# ── figure: kNN balanced accuracy against k, with LightGBM and MR on the same rows ────────
with open(os.path.join(OUT, "fig_dm_knn.tex"), "w") as f:
    f.write("\\begin{tikzpicture}\n\\begin{axis}[thesis, width=0.62\\textwidth, height=5.4cm, xmode=log, log basis x=10, xlabel={$k$ (neighbours)}, ylabel={balanced accuracy, all samples}, xmin=3, xmax=1600, ymin=0.44, ymax=0.70, legend pos=outer north east, legend columns=1, legend cell align=left]\n")
    for h, lab in HS:
        r = rows(h, "knn.csv")
        if not r: continue
        r = [x for x in r if not x.get("note")]
        pts = " ".join(f"({x['k']},{float(x['ba_all']):.4f})" for x in r)
        f.write(f"\\addplot[{COL[h]}, thick, mark={MARK[h]}, mark size=1.6pt] coordinates {{{pts}}};\n\\addlegendentry{{kNN {lab}}}\n")
    f.write("\\end{axis}\n\\end{tikzpicture}\n")

# ── tables ─────────────────────────────────────────────────────────────────────────────
T = []
def fmt(x, d=3): return f"{float(x):.{d}f}"
# dimensionality
T.append("% dimensionality by horizon\n\\begin{tabular}{lccccc}\n\\toprule\n& " + " & ".join(l for h, l in HS if h in have) + " \\\\\n\\midrule\n")
dim = {}
for h in have:
    sp = rows(h, "pca_spectrum.csv"); cum = [float(x["cum"]) for x in sp]
    n = lambda q: next(i + 1 for i, c in enumerate(cum) if c >= q)
    cb = json.load(open(os.path.join(SRC, h, "corr_blocks.json")))
    rp = {x["k"]: x for x in rows(h, "random_projection.csv")}
    rep = open(os.path.join(SRC, h, "report.md")).read()
    import re
    m = re.search(r"stress-1 of the 2-D metric MDS: ([0-9]+\.[0-9]+); of the PCA 2-D projection on the same rows: ([0-9]+\.[0-9]+)", rep)
    pr = re.search(r"participation ratio of the spectrum ([0-9]+\.[0-9]+)", rep).group(1)
    dim[h] = dict(n50=n(0.5), n80=n(0.8), n95=n(0.95), pr=pr, eff=f"{cb['effective_n_features']:.1f}", b7=len(cb["blocks"]["0.7"]), b9=len(cb["blocks"]["0.9"]),
                  rp20=fmt(rp["20"]["ratio_sd"], 2), mds=m.group(1), pca2=m.group(2))
for key, lab in [("n50", "components for 50\\% of the variance"), ("n80", "components for 80\\%"), ("n95", "components for 95\\%"), ("pr", "participation ratio of the PCA spectrum"),
                 ("eff", "effective number of features (correlation)"), ("b7", "blocks at $|\\rho| \\ge 0.7$"), ("b9", "pairs at $|\\rho| \\ge 0.9$"),
                 ("rp20", "RP to 20-D: s.d.\\ of the distance ratio"), ("mds", "MDS 2-D stress-1"), ("pca2", "PCA 2-D stress-1")]:
    T.append(lab + " & " + " & ".join(str(dim[h][key]) for h in have) + " \\\\\n")
T.append("\\bottomrule\n\\end{tabular}\n\n")
# separability
T.append("% separability by horizon (all / dedup)\n\\begin{tabular}{l" + "cc" * len(have) + "}\n\\toprule\n& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for h, l in HS if h in have) + " \\\\\n")
T.append("".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(have))) + "\nModel & " + " & ".join("all & dedup" for _ in have) + " \\\\\n\\midrule\n")
sep = {h: {x["model"]: x for x in rows(h, "separability.csv")} for h in have}
knn = {h: [x for x in rows(h, "knn.csv") if not x.get("note")] for h in have}
def cell(h, name):
    x = sep[h].get(name); return f"{fmt(x['ba_all'])} & {fmt(x['ba_dedup'])}" if x else "--- & ---"
for name, lab in [("Mean Reversion", "Mean Reversion-$H$"), ("LDA (uniform priors)", "LDA"), ("Logistic regression (balanced)", "Logistic regression"),
                  ("Decision tree, depth 3 (balanced)", "Decision tree, depth 3"), ("Decision tree, depth 6 (balanced)", "Decision tree, depth 6")]:
    T.append(lab + " & " + " & ".join(cell(h, name) for h in have) + " \\\\\n")
best = {h: max(knn[h], key=lambda x: float(x["ba_all"])) for h in have}
T.append("kNN, best $k$ (300k rows) & " + " & ".join(f"{fmt(best[h]['ba_all'])} & {fmt(best[h]['ba_dedup'])}" for h in have) + " \\\\\n")
T.append("\\quad best $k$; $\\Delta$ vs control [CI] & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{best[h]['k']}; ${float(best[h]['delta_vs_lgb']):+.1f}$ [${float(best[h]['lo']):+.1f}$, ${float(best[h]['hi']):+.1f}$]}}" for h in have) + " \\\\\n")
T.append("LightGBM G5 (control) & " + " & ".join(cell(h, "LightGBM G5 (control)") for h in have) + " \\\\\n")
T.append("\\bottomrule\n\\end{tabular}\n\n")
# purity lifts
T.append("% purity lifts (original space / UMAP)\n\\begin{tabular}{l" + "c" * len(have) + "}\n\\toprule\nLift of & " + " & ".join(l for h, l in HS if h in have) + " \\\\\n\\midrule\n")
pur = {h: {(x["space"], x["label"]): x for x in rows(h, "purity.csv")} for h in have}
for lab, key in [("prediction label", "y"), ("Gamma category", "category"), ("price-level bucket", "mid_bucket"), ("activity quartile", "act_bucket"), ("market identity", "market_id")]:
    T.append(lab + " & " + " & ".join((f"{float(pur[h][('original60', key)]['lift']):.0f} / {float(pur[h][('umap', key)]['lift']):.0f}" if key == "market_id" else f"{float(pur[h][('original60', key)]['lift']):.2f} / {float(pur[h][('umap', key)]['lift']):.2f}") for h in have) + " \\\\\n")
T.append("\\bottomrule\n\\end{tabular}\n\n")
# mutual information top 6 per horizon
T.append("% mutual information: top six features per horizon (nats)\n\\begin{tabular}{" + "ll" * len(have) + "}\n\\toprule\n" + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for h, l in HS if h in have) + " \\\\\n\\midrule\n")
mi = {h: rows(h, "mutual_information.csv")[:6] for h in have}
for i in range(6):
    T.append(" & ".join(f"\\path{{{mi[h][i]['feature']}}} & {float(mi[h][i]['mi_nats']):.3f}" for h in have) + " \\\\\n")
T.append("\\midrule\n" + " & ".join("\\multicolumn{2}{l}{" + "; ".join(f"{g.split()[0]} {s:.2f}" for g, s in sorted({g: sum(float(x['mi_nats']) for x in rows(h, 'mutual_information.csv') if x['group'] == g) for g in set(x['group'] for x in rows(h, 'mutual_information.csv'))}.items())) + "}" for h in have) + " \\\\\n\\bottomrule\n\\end{tabular}\n\n")
# clusters per horizon
T.append("% market clusters per horizon\n")
for h in have:
    acc = rows(h, "market_cluster_accuracy.csv")
    if not acc: T.append(f"% {h}: no clustering output\n"); continue
    rep = open(os.path.join(SRC, h, "report.md")).read()
    ami = re.search(r"Adjusted mutual information with the Gamma category: ward ([0-9]+\.[0-9]+), kmeans ([0-9]+\.[0-9]+), hdbscan ([0-9]+\.[0-9]+)", rep)
    T.append(f"% {h}: AMI ward {ami.group(1)} kmeans {ami.group(2)} hdbscan {ami.group(3)}\n")
    for a in acc:
        T.append(f"%   cluster {a['cluster']}: {a['markets']} markets, {a['samples']} samples, p_move {float(a['p_move']):.2f}, MR {fmt(a['ba_mr'])}, LGB {fmt(a['ba_lgb'])}" + (f", stateful(Oct) {fmt(a['ba_stateful_oct'])}" if a.get("ba_stateful_oct") else "") + f"; {a['top_categories']}\n")
open(os.path.join(OUT, "tables_cross_horizon.tex"), "w").write("".join(T)); print("written")
