"""Summary figures of the wallet-market transaction graph from the CSV outputs of
graph_animation.py (stages S).  usage: python3 graph_summary_plots.py <dir with the csv/json> <out dir>"""
import sys, os, json, csv, collections, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
A, B, C, D, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#d9d7d2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def ccdf(vals_counts):
    v = np.array([x for x, _ in vals_counts], float); c = np.array([n for _, n in vals_counts], float)
    o = np.argsort(v); v, c = v[o], c[o]
    tail = np.cumsum(c[::-1])[::-1] / c.sum()
    return v, tail


dd = collections.defaultdict(list)
for r in csv.DictReader(open(f"{SRC}/degree_distributions.csv")): dd[r["quantity"]].append((int(r["value"]), int(r["count"])))
H = {k: np.array([float(r[k]) for r in csv.DictReader(open(f"{SRC}/hourly.csv"))]) for k in
     ["t", "events", "taker_fills", "wallets", "markets", "edges", "new_wallets", "new_markets", "new_edges", "components", "giant_share", "usdc"]}
Dg = {k: np.array([float(r[k]) for r in csv.DictReader(open(f"{SRC}/daily_growth.csv"))]) for k in ["t", "cum_wallets", "cum_markets", "cum_edges", "components", "giant_share"]}
st = json.load(open(f"{SRC}/stats.json"))
th = (H["t"] - H["t"][0]) / 86400; td = (Dg["t"] - H["t"][0]) / 86400 + 1
days = np.arange(0, th[-1] + 1, 3); dl = [time.strftime("%d %b", time.gmtime(H["t"][0] + d * 86400)) for d in days]

fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.6))
# (a) degree CCDF
a = ax[0, 0]
for key, lab, col in [("wallet_degree", "wallets: markets traded", A), ("market_degree", "markets: wallets seen", B), ("edge_multiplicity", "edges: fills per pair", C)]:
    v, t = ccdf(dd[key]); a.plot(v, t, color=col, lw=2, label=lab)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("degree / multiplicity"); a.set_ylabel("share of nodes (edges) with at least x")
a.set_title("(a) heavy tails: degrees and edge weights", loc="left", fontsize=9.5); a.legend(fontsize=8); a.grid(color=GRID, lw=0.5)
# (b) hourly activity
a = ax[0, 1]
a.fill_between(th, 0, H["events"] / 1e3, color=D, alpha=0.3, lw=0); a.plot(th, H["events"] / 1e3, color=D, lw=1.2)
a.set_ylabel("fill legs per hour (thousands)"); a.set_xticks(days); a.set_xticklabels(dl, fontsize=7.5)
a.set_title("(b) activity: fill legs per hour", loc="left", fontsize=9.5); a.grid(color=GRID, lw=0.5); a.set_ylim(0)
# (c) active nodes per hour
a = ax[0, 2]
a.plot(th, H["wallets"] / 1e3, color=A, lw=1.2, label="active wallets"); a.plot(th, H["markets"] / 1e3, color=B, lw=1.2, label="active markets")
a.set_ylabel("per hour (thousands)"); a.set_xticks(days); a.set_xticklabels(dl, fontsize=7.5); a.legend(fontsize=8)
a.set_title("(c) how many nodes are awake in an hour", loc="left", fontsize=9.5); a.grid(color=GRID, lw=0.5); a.set_ylim(0)
# (d) components of the hourly graph
a = ax[1, 0]
a.plot(th, H["components"], color=INK, lw=1.2)
a.set_ylabel("connected components of the hourly graph"); a.set_xticks(days); a.set_xticklabels(dl, fontsize=7.5)
a.set_title("(d) components of the hourly graph", loc="left", fontsize=9.5); a.grid(color=GRID, lw=0.5); a.set_ylim(0)
a2 = ax[1, 1]
a2.plot(th, H["giant_share"] * 100, color=C, lw=1.2)
a2.set_ylabel("giant component, % of the hour's active nodes"); a2.set_xticks(days); a2.set_xticklabels(dl, fontsize=7.5); a2.set_ylim(0, 100)
a2.set_title("(e) giant component, hourly graph", loc="left", fontsize=9.5); a2.grid(color=GRID, lw=0.5)
# (f) cumulative growth
a = ax[1, 2]
a.plot(td, Dg["cum_wallets"] / 1e3, color=A, lw=2, label="wallets seen so far (k)")
a.plot(td, Dg["cum_markets"] / 1e3, color=B, lw=2, label="markets seen so far (k)")
a.plot(td, Dg["cum_edges"] / 1e5, color=C, lw=2, label="edges seen so far (100k)")
a.set_xlabel("day of the window"); a.set_xticks(days); a.set_xticklabels(dl, fontsize=7.5); a.legend(fontsize=8)
a.set_title("(f) the graph keeps growing: new wallets every day", loc="left", fontsize=9.5); a.grid(color=GRID, lw=0.5); a.set_ylim(0)
fig.suptitle(f"Polymarket wallet × market transaction graph, {time.strftime('%d %b', time.gmtime(H['t'][0]))}–{time.strftime('%d %b %Y', time.gmtime(H['t'][-1]))}: "
             f"{st['n_wallets']:,} wallets, {st['n_markets']:,} markets, {st['n_edges']:,} edges, {st['n_events']:,} fill legs, {st['n_components']} connected component", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}/graph_summary.png", dpi=170); print("written", f"{OUT}/graph_summary.png")

# text summary of the hourly series
print("hourly: events mean %.0f min %.0f max %.0f; wallets mean %.0f (min %.0f max %.0f); markets mean %.0f; edges mean %.0f; components mean %.1f (min %.0f max %.0f); giant share mean %.3f min %.3f; new wallets/h mean %.0f; new edges/h mean %.0f (share of hour's edges %.3f)" % (
    H["events"].mean(), H["events"].min(), H["events"].max(), H["wallets"].mean(), H["wallets"].min(), H["wallets"].max(), H["markets"].mean(), H["edges"].mean(),
    H["components"].mean(), H["components"].min(), H["components"].max(), H["giant_share"].mean(), H["giant_share"].min(), H["new_wallets"].mean(), H["new_edges"].mean(), (H["new_edges"] / np.maximum(H["edges"], 1)).mean()))
print("daily growth:", [(int(w), int(m), int(e), int(c), round(g, 3)) for w, m, e, c, g in zip(Dg["cum_wallets"], Dg["cum_markets"], Dg["cum_edges"], Dg["components"], Dg["giant_share"])])
# hour-of-day profile
hod = ((H["t"] // 3600) % 24).astype(int)
prof = np.array([H["events"][hod == h].mean() for h in range(24)])
print("hour-of-day profile (UTC), fill legs/h in k:", np.round(prof / 1e3).astype(int).tolist(), "peak h", int(prof.argmax()), "trough h", int(prof.argmin()), "ratio %.2f" % (prof.max() / prof.min()))
