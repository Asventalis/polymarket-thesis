"""Build the pgfplots/TikZ figures of the exploratory-analysis section (stdlib only).

usage: python3 make_expl_figures.py <agy_project/reports/analysis dir> <thesis figures/expl dir>

Sources: data_audit_2026-10-15 (raw stream, univariate hit rates, text audit),
graphviz_2026-10-27 (degree distributions), datamining_2026-10-26/<H> (purity, separability,
kNN, mutual information), and two queries on the validation feature parquets
(probability of a move by time since the last move; histogram of the mid), whose
output is embedded below (vp_eda.py, 11 Nov 2026)."""
import csv, os, sys
SRC, OUT = sys.argv[1], sys.argv[2]
HS = [("1m", "1 min"), ("15m", "15 min"), ("1h", "1 h"), ("6h", "6 h"), ("12h", "12 h")]
HZ = {"1m": "hzA", "15m": "hzB", "1h": "hzC", "6h": "hzD", "12h": "hzE"}
GROUPS = [("G1 price", "price", "vizA"), ("G2 book", "book shape", "vizB"),
          ("G3 flow/clock", "activity and clock", "vizC"), ("G4 trades", "on-chain trades", "vizD"),
          ("G5 static", "market metadata", "vizE")]
H_ENT = {"1m": 0.42, "15m": 0.834, "1h": 1.00, "6h": 1.10, "12h": 1.09}

def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)
    print("wrote", name)

def xbar(rows, xlabel, xmin, xmax, width="0.47\\textwidth", height=None, color="vizA", fmt="%.0f",
         unit="", extra="", bar_width="9pt", ref=None, refname=None, multi=None, legend=None, labelw=None):
    """rows = [(label, value), ...] from top to bottom. multi = list of (series_name, color, [values])."""
    labels = [r[0] for r in rows]
    n = len(labels)
    height = height or f"{1.0 + 0.62 * n:.1f}cm"
    s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, xbar, width=%s, height=%s, xmin=%s, xmax=%s,\n" % (width, height, xmin, xmax)
    s += "  ymajorgrids=false, xmajorgrids, y dir=reverse, ytick=data, symbolic y coords={%s},\n" % ",".join("{%s}" % l for l in labels)
    s += "  xlabel={%s}, bar width=%s, enlarge y limits=%.3f, y axis line style={draw=none}, ytick style={draw=none},\n" % (xlabel, bar_width, 0.6 / max(n - 1, 1))
    s += "  yticklabel style={font=\\footnotesize%s},\n" % (", text width=%s, align=right" % labelw if labelw else "")
    s += "  nodes near coords, nodes near coords align={horizontal}, every node near coord/.append style={font=\\scriptsize, color=vizInk, /pgf/number format/fixed, /pgf/number format/precision=%d},\n" % (int(fmt.split(".")[1].rstrip("f")) if "." in fmt else 0)
    if legend: s += "  legend style={at={(0.5,1.02)}, anchor=south}, legend columns=2,\n"
    s += extra + "]\n"
    if multi:
        for name, col, vals in multi:
            s += "\\addplot[fill=%s, draw=none] coordinates {%s};\n" % (col, " ".join("(%s,{%s})" % (v, l) for l, v in zip(labels, vals)))
        if legend: s += "\\legend{%s}\n" % ", ".join(legend)
    else:
        s += "\\addplot[fill=%s, draw=none] coordinates {%s};\n" % (color, " ".join("(%s,{%s})" % (v, l) for l, v in rows))
    if ref is not None:
        s += "\\draw[vizInk, thin] (axis cs:%s,{%s}) -- (axis cs:%s,{%s}) node[pos=1, below, font=\\scriptsize, color=vizInk, xshift=2pt, yshift=1pt] {%s};\n" % (ref, labels[0], ref, labels[-1], refname or "")
    s += "\\end{axis}\n\\end{tikzpicture}\n"
    return s

# ── A. the event stream ───────────────────────────────────────────────────────────────
a = xbar([("add size to a price level", 55), ("remove size from a level", 44), ("touch the best quote", 8), ("change the mid", 4)],
         "events out of 100", 0, 70, bar_width="11pt")
b = xbar([("within 1 ms", 30), ("within 1 s", 79), ("within 1 min", 98)], "share of events (\\%)", 0, 118, bar_width="11pt")
w("fig_expl_stream_a.tex", a); w("fig_expl_stream_b.tex", b)

# ── B. histogram of the mid (1 h validation samples, bins of 0.02; vp_eda.py) ───────────
MID = [2375178,917441,638442,430659,322088,317841,207365,262417,176522,188105,209611,162104,285979,151506,168496,174913,106453,134881,146208,93843,117646,108333,109612,136745,125186,155133,75193,78532,108222,49988,83437,79242,54584,64836,58753,41110,49697,66946,54672,53279,62172,33152,36518,62853,47936,81408,173748,82057,62521,80562]
tot = sum(MID)
pts = " ".join("(%.2f,%.2f)" % (0.01 + 0.02 * i, 100.0 * c / tot) for i, c in enumerate(MID))
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, ybar, bar width=4.2pt, xmin=0, xmax=1, ymin=0, ymax=10.5, xlabel={mid price at the sample time}, ylabel={share of the samples (\\%)}, xtick={0,0.05,0.2,0.4,0.6,0.8,0.95,1}, xticklabels={0,0.05,0.2,0.4,0.6,0.8,0.95,1}, clip=true]\n"
s += "\\addplot[fill=vizA, draw=none] coordinates {%s};\n" % pts
s += "\\draw[vizInk, thin] (axis cs:0.05,0) -- (axis cs:0.05,10.5);\n\\draw[vizInk, thin] (axis cs:0.95,0) -- (axis cs:0.95,10.5);\n"
s += "\\node[font=\\scriptsize, color=vizInk, anchor=west, align=left] at (axis cs:0.06,9.6) {first bar cut: %.0f\\%% of the samples have a mid below $0.02$};\n" % (100.0 * MID[0] / tot)
s += "\\node[font=\\scriptsize, color=vizInk, anchor=north, align=center] at (axis cs:0.5,7.6) {central samples ($0.05 \\le \\mathrm{mid} \\le 0.95$): %.0f\\%% of the samples};\n" % (100.0 * sum(MID[3:47]) / tot)
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_expl_midhist.tex", s)

# ── C. the activity clock: P(move within H) by time since the last move (vp_eda.py) ───
CLOCK = {  # bucket -> p(const) per horizon
 "1m":  [0.2910,0.4752,0.6821,0.7850,0.8485,0.9245,0.9594,0.9792,0.9843,0.9907,0.9943,0.9969,0.9973,0.9996],
 "15m": [0.1691,0.1860,0.2647,0.3358,0.4231,0.5576,0.6743,0.7521,0.8160,0.8875,0.9257,0.9594,0.9729,0.9957],
 "1h":  [0.1397,0.1264,0.1675,0.1994,0.2516,0.3367,0.4184,0.4908,0.5831,0.6967,0.7744,0.8681,0.9029,0.9826],
 "6h":  [0.0844,0.0767,0.0863,0.0966,0.1144,0.1465,0.1776,0.2132,0.2664,0.3453,0.4464,0.5297,0.6445,0.9124],
 "12h": [0.0641,0.0572,0.0630,0.0693,0.0817,0.1042,0.1274,0.1472,0.1773,0.2293,0.2938,0.3693,0.4966,0.8574]}
MIDPT = [5, 17, 42, 85, 190, 424, 849, 1470, 2546, 5091, 10182, 20364, 49883, 160000]
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, xmode=log, log basis x=10, xmin=3, xmax=400000, ymin=0, ymax=1.0, xlabel={time since the mid last moved}, ylabel={share of samples whose mid moves within $H$},\n"
s += "  xtick={10,60,600,3600,21600,86400}, xticklabels={10 s,1 min,10 min,1 h,6 h,1 day}, legend style={at={(0.5,-0.28)}, anchor=north, font=\\footnotesize}, legend columns=5]\n"
for h, lab in HS:
    pts = " ".join("(%d,%.4f)" % (x, 1 - p) for x, p in zip(MIDPT, CLOCK[h]))
    s += "\\addplot[%s, thick, mark=*, mark size=1.3pt] coordinates {%s};\n\\addlegendentry{$H$ = %s}\n" % (HZ[h], pts, lab)
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_expl_clock.tex", s)

# ── D. direction given a move: hit rate of single features at 1 h (univariate_1h.csv) ──
NAMES = {"mid_minus_mean60": "mid minus its 60-min mean", "last_move_sign": "sign of the last move", "ret_3h": "3-hour return",
         "imb1": "first-level imbalance", "ret_1h": "1-hour return", "range_pos_60m": "position in the 60-min range",
         "micro_minus_mid_ticks": "microprice minus mid", "ret_15m": "15-min return", "flow_15m": "taker order flow, 15 min",
         "imb5": "imbalance over five levels", "ret_12h": "12-hour return", "ret_6h": "6-hour return"}
uni = {r["feature"]: float(r["hit_rate_on_moves"]) for r in csv.DictReader(open(os.path.join(SRC, "data_audit_2026-10-15", "univariate_1h.csv")))}
order = ["mid_minus_mean60", "last_move_sign", "ret_3h", "imb1", "ret_1h", "range_pos_60m", "micro_minus_mid_ticks", "ret_15m", "flow_15m", "imb5"]
rows = [(NAMES[f], round(100 * uni[f], 1)) for f in order]
w("fig_expl_direction.tex", xbar(rows, "direction guessed right (\\% of the samples that moved within 1 h)", 48, 60, width="0.76\\textwidth", fmt="%.1f", ref=50, refname="coin toss", labelw="4.2cm"))

# ── E. the kind of contract: share of samples whose mid moves within 1 h (text_audit.md) ─
cat = [("weather", 69.9), ("esports", 67.3), ("culture", 62.7), ("sports", 45.9), ("economy and tech", 45.2), ("crypto", 43.6), ("politics", 35.8)]
tpl = [("game over/under line", 81.2), ("Oscars award", 69.7), ("game winner", 62.1), ("tournament winner", 47.8),
       ("show season winner", 39.3), ("party nominee", 31.3), ("primary election", 26.1), ("club wins on a date", 24.2)]
w("fig_expl_contract_a.tex", xbar(cat, "samples whose mid moves within 1 h (\\%)", 0, 85, bar_width="9pt"))
w("fig_expl_contract_b.tex", xbar(tpl, "samples whose mid moves within 1 h (\\%)", 0, 95, bar_width="9pt"))

# ── F. graph degrees: how many markets a wallet trades, how many wallets a market sees ──
deg = {"wallet_degree": {}, "market_degree": {}}
for r in csv.DictReader(open(os.path.join(SRC, "graphviz_2026-10-27", "degree_distributions.csv"))):
    if r["quantity"] in deg: deg[r["quantity"]][int(r["value"])] = int(r["count"])
def share(d, lo, hi):
    tot = sum(d.values()); return 100.0 * sum(c for v, c in d.items() if lo <= v <= hi) / tot
wd, md = deg["wallet_degree"], deg["market_degree"]
rows_w = [("one market", share(wd, 1, 1)), ("two", share(wd, 2, 2)), ("three to five", share(wd, 3, 5)), ("six to twenty", share(wd, 6, 20)),
          ("21 to 100", share(wd, 21, 100)), ("more than 100", share(wd, 101, 10**9))]
rows_m = [("2 to 10 wallets", share(md, 1, 10)), ("11 to 30", share(md, 11, 30)), ("31 to 100", share(md, 31, 100)),
          ("101 to 1{,}000", share(md, 101, 1000)), ("more than 1{,}000", share(md, 1001, 10**9))]
w("fig_graph_degrees_a.tex", xbar([(l, round(v, 1)) for l, v in rows_w], "share of the wallets (\\%)", 0, 40, bar_width="9pt"))
w("fig_graph_degrees_b.tex", xbar([(l, round(v, 1)) for l, v in rows_m], "share of the markets (\\%)", 0, 45, bar_width="9pt"))
print("wallet degree shares:", [(l, round(v, 1)) for l, v in rows_w]); print("market degree shares:", [(l, round(v, 1)) for l, v in rows_m])

# ── G. what organises the feature space: neighbourhood lift at 15 min (purity.csv) ─────
pur = {r["label"]: float(r["lift"]) for r in csv.DictReader(open(os.path.join(SRC, "datamining_2026-10-26", "15m", "purity.csv"))) if r["space"] == "original60"}
rows = [("the label", pur["y"]), ("the price level", pur["mid_bucket"]), ("the category", pur["category"]), ("the activity", pur["act_bucket"]), ("the market itself", pur["market_id"])]
labels = [l for l, _ in rows]
s_ = "\\begin{tikzpicture}\n\\begin{axis}[thesis, xbar, width=0.78\\textwidth, height=4.6cm, xmin=1, xmax=3.1, ymajorgrids=false, xmajorgrids, y dir=reverse, ytick=data,\n"
s_ += "  symbolic y coords={%s}, xlabel={how much more often than chance the fifteen nearest neighbours of a sample share its value (1 = chance)}, bar width=9pt, enlarge y limits=0.15,\n" % ",".join("{%s}" % l for l in labels)
s_ += "  y axis line style={draw=none}, ytick style={draw=none}, yticklabel style={font=\\footnotesize}, xtick={1,1.5,2,2.5}, clip=false,\n"
s_ += "  nodes near coords, nodes near coords align={horizontal}, point meta=explicit symbolic, every node near coord/.append style={font=\\scriptsize, color=vizInk}]\n"
pts = " ".join("(%.2f,{%s}) [%.2f]" % (v, l, v) for l, v in rows[:-1]) + " (2.45,{%s}) [%.0f, bar cut]" % (rows[-1][0], rows[-1][1])
s_ += "\\addplot[fill=vizA, draw=none] coordinates {%s};\n\\end{axis}\n\\end{tikzpicture}\n" % pts
w("fig_structure_lift.tex", s_)

# ── H. separability at 15 min: simple learners, all / deduplicated (separability.csv, knn.csv)
sep = {r["model"]: (float(r["ba_all"]), float(r["ba_dedup"])) for r in csv.DictReader(open(os.path.join(SRC, "datamining_2026-10-26", "15m", "separability.csv")))}
knn = [r for r in csv.DictReader(open(os.path.join(SRC, "datamining_2026-10-26", "15m", "knn.csv"))) if r["k"] == "200" and not r["note"]][0]
rows = [("Mean Reversion (the rule)", sep["Mean Reversion"]), ("linear discriminant analysis", sep["LDA (uniform priors)"]),
        ("logistic regression", sep["Logistic regression (balanced)"]), ("decision tree, depth 3", sep["Decision tree, depth 3 (balanced)"]),
        ("decision tree, depth 6", sep["Decision tree, depth 6 (balanced)"]), ("nearest neighbours (200)", (float(knn["ba_all"]), float(knn["ba_dedup"]))),
        ("LightGBM (the control)", sep["LightGBM G5 (control)"])]
s = xbar([(l, 0) for l, _ in rows], "balanced accuracy at 15 min", 0.33, 0.66, width="0.76\\textwidth", fmt="%.3f", bar_width="6pt",
         multi=[("all samples", "vizA", [round(v[0], 3) for _, v in rows]), ("deduplicated", "vizB", [round(v[1], 3) for _, v in rows])],
         legend=["all samples", "deduplicated"], extra="  xtick={0.333,0.4,0.5,0.6}, xticklabels={chance,0.40,0.50,0.60},\n", labelw="4.2cm")
w("fig_structure_sep.tex", s)

# ── J. mutual information at 15 min as a share of the label's uncertainty (top 12) ─────
MI_NAMES = {"rv_60m": "volatility, last 60 min", "moves_60m": "moves, last 60 min", "since_move": "time since the last move",
            "mid_minus_mean60": "mid minus its 60-min mean", "range_pos_60m": "position in the 60-min range", "micro_minus_mid_ticks": "microprice minus mid",
            "ret_15m": "15-min return", "ret_1h": "1-hour return", "log_duration_h": "duration of the market", "imb1": "first-level imbalance",
            "sgn_15m": "sign of the 15-min return", "ret_3h": "3-hour return", "usd_60m": "dollars traded, last 60 min", "log_bid_sz0": "size at the best bid"}
mi = list(csv.DictReader(open(os.path.join(SRC, "datamining_2026-10-26", "15m", "mutual_information.csv"))))[:12]
labels = [MI_NAMES.get(r["feature"], r["feature"]) for r in mi]
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, xbar, width=0.76\\textwidth, height=8.6cm, xmin=0, xmax=34, ymajorgrids=false, xmajorgrids, y dir=reverse, ytick={%s},\n" % ",".join("{%s}" % l for l in labels)
s += "  symbolic y coords={%s}, xlabel={share of the label's uncertainty removed by the feature alone (\\%%)}, bar width=7pt, bar shift=0pt, enlarge y limits=0.055,\n" % ",".join("{%s}" % l for l in labels)
s += "  y axis line style={draw=none}, ytick style={draw=none}, yticklabel style={font=\\footnotesize, text width=4.2cm, align=right},\n"
s += "  nodes near coords, nodes near coords align={horizontal}, every node near coord/.append style={font=\\scriptsize, color=vizInk, /pgf/number format/fixed, /pgf/number format/precision=0},\n"
s += "  legend style={at={(0.5,1.02)}, anchor=south, font=\\footnotesize}, legend columns=5, legend image code/.code={\\draw[#1, draw=none] (0cm,-0.1cm) rectangle (0.3cm,0.1cm);}]\n"
for g, gname, col in GROUPS:
    pts = " ".join("(%.1f,{%s})" % (100 * float(r["mi_nats"]) / H_ENT["15m"], MI_NAMES.get(r["feature"], r["feature"])) for r in mi if r["group"] == g)
    if pts:
        s += "\\addplot[fill=%s, draw=none] coordinates {%s};\n\\addlegendentry{%s}\n" % (col, pts, gname)
    else:
        s += "\\addlegendimage{fill=%s, draw=none}\n\\addlegendentry{%s}\n" % (col, gname)
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_structure_mi.tex", s)

# ── K. across the horizons: three panels ───────────────────────────────────────────────
X = "symbolic x coords={1 min, 15 min, 1 h, 6 h, 12 h}, xtick=data, enlarge x limits=0.12"
lift = [1.06, 1.26, 1.39, 1.45, 1.42]
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, width=0.5\\textwidth, height=5cm, %s, ymin=1, ymax=1.6, ylabel={lift of the label (1 = chance)}, xlabel={horizon $H$}]\n" % X
s += "\\addplot[vizA, thick, mark=*, mark size=1.8pt] coordinates {%s};\n" % " ".join("(%s,%.2f)" % (lab, v) for (h, lab), v in zip(HS, lift))
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_structure_hz_a.tex", s)
SEPH = {"Mean Reversion": [0.536, 0.537, 0.531, 0.507, 0.507], "best linear model": [0.504, 0.568, 0.563, 0.544, 0.543],
        "decision tree, depth 6": [0.575, 0.571, 0.554, 0.531, 0.531], "nearest neighbours": [0.508, 0.574, 0.566, 0.543, 0.546],
        "LightGBM (the control)": [0.609, 0.616, 0.602, 0.572, 0.572]}
COLS = ["vizInk", "vizA", "vizB", "vizC", "vizD"]; MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*"]
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, width=0.6\\textwidth, height=5.2cm, %s, ymin=0.48, ymax=0.64, ylabel={balanced accuracy, all samples}, xlabel={horizon $H$}, legend pos=outer north east, legend cell align=left, legend style={font=\\scriptsize}]\n" % X
for (name, vals), col, mk in zip(SEPH.items(), COLS, MARKS):
    s += "\\addplot[%s, thick, mark=%s, mark size=1.6pt] coordinates {%s};\n\\addlegendentry{%s}\n" % (col, mk, " ".join("(%s,%.3f)" % (lab, v) for (h, lab), v in zip(HS, vals)), name)
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_structure_hz_b.tex", s)
# best feature of each group, as a share of the label's uncertainty, per horizon
best = {g: [] for g, _, _ in GROUPS}
for h, lab in HS:
    rows_h = list(csv.DictReader(open(os.path.join(SRC, "datamining_2026-10-26", h, "mutual_information.csv"))))
    for g, _, _ in GROUPS:
        best[g].append(100 * max(float(r["mi_nats"]) for r in rows_h if r["group"] == g) / H_ENT[h])
s = "\\begin{tikzpicture}\n\\begin{axis}[thesis, width=0.6\\textwidth, height=5.2cm, %s, ymin=0, ymax=50, ylabel={best feature of the group (\\%% of uncertainty)}, xlabel={horizon $H$}, legend pos=outer north east, legend cell align=left, legend style={font=\\scriptsize}]\n" % X
for (g, gname, col), mk in zip(GROUPS, MARKS):
    s += "\\addplot[%s, thick, mark=%s, mark size=1.6pt] coordinates {%s};\n\\addlegendentry{%s}\n" % (col, mk, " ".join("(%s,%.1f)" % (lab, v) for (h, lab), v in zip(HS, best[g])), gname)
s += "\\end{axis}\n\\end{tikzpicture}\n"
w("fig_structure_hz_c.tex", s)
print("best per group:", {g: [round(v, 1) for v in vs] for g, vs in best.items()})

# ── I. the depth-3 tree at 15 min, drawn ───────────────────────────────────────────────
tree = r"""\begin{tikzpicture}[
  font=\footnotesize, level distance=16mm, edge from parent/.style={draw, vizGrid, line width=0.8pt},
  q/.style={draw=vizInk, line width=0.6pt, rounded corners=3pt, align=center, inner sep=4pt, text width=34mm, fill=white},
  leafD/.style={draw=none, fill=vizA, text=white, rounded corners=3pt, inner sep=4pt, minimum width=14mm, font=\footnotesize\scshape},
  leafC/.style={draw=none, fill=vizB, text=white, rounded corners=3pt, inner sep=4pt, minimum width=14mm, font=\footnotesize\scshape},
  leafU/.style={draw=none, fill=vizC, text=white, rounded corners=3pt, inner sep=4pt, minimum width=14mm, font=\footnotesize\scshape},
  lab/.style={font=\scriptsize, color=vizInk, fill=white, inner sep=1pt},
  level 1/.style={sibling distance=80mm}, level 2/.style={sibling distance=40mm}, level 3/.style={sibling distance=19mm}]
\node[q] {has the mid moved in the last 17 minutes?}
  child { node[q] {more than 8 moves in the last hour?}
    child { node[q] {was the last move down?}
      child { node[leafU] {up} edge from parent node[lab, left] {yes} }
      child { node[leafD] {down} edge from parent node[lab, right] {no} }
      edge from parent node[lab, left] {no} }
    child { node[q] {last move less than 2 minutes ago?}
      child { node[leafU] {up} edge from parent node[lab, left] {yes} }
      child { node[leafD] {down} edge from parent node[lab, right] {no} }
      edge from parent node[lab, right] {yes} }
    edge from parent node[lab, left, xshift=-4pt] {yes} }
  child { node[leafC, text width=30mm, align=center, font=\footnotesize] {\textsc{const}\\[1pt]{\scriptsize\normalfont on every branch below}}
    edge from parent node[lab, right, xshift=4pt] {no} };
\end{tikzpicture}
"""
w("fig_structure_tree.tex", tree)
