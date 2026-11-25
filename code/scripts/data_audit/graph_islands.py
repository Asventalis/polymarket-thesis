"""Islands of the market map: HDBSCAN on the 2-D market positions of a layout of graph_animation.py /
graph_layout_anchor.py, with the category mix, the frequent words, the date range and the Up/Down
interval length of every island.  usage: ./run_dev.sh graph_islands.py --layout layout_anchor.npz [--min-size 150]
writes market_clusters<suffix>.csv and market_cluster_labels<suffix>.npy (suffix = layout name after 'layout_anchor')."""
import argparse, re, collections, sys
sys.path.insert(0, "/data/v.pampinella/dm_pkgs")
import numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN

OUT = "/data/v.pampinella/data_audit/graphviz"
p = argparse.ArgumentParser(); p.add_argument("--layout", default="layout_anchor.npz"); p.add_argument("--min-size", type=int, default=150); a = p.parse_args()
suffix = a.layout.replace("layout_anchor", "").replace(".npz", "")
lay = np.load(f"{OUT}/{a.layout}"); xy = lay["xy"]; nW = int(lay["nW"]); XM = xy[nW:]
meta = pd.read_parquet(f"{OUT}/market_meta.parquet").sort_values("mi"); e = np.load(f"{OUT}/edges.npz"); fills = e["fillsM"]; deg = e["degM"]
lab = HDBSCAN(min_cluster_size=a.min_size, min_samples=10).fit(XM).labels_
print("clusters", lab.max() + 1, "noise share %.3f" % (lab < 0).mean(), "fills in clusters %.3f" % (fills[lab >= 0].sum() / fills.sum()))
stop = set("will the be in on of to a an and or by at for vs is win at than more less over under up down yes no price this that with".split())


def tokens(q):
    q = re.sub(r"[0-9\$,\.\%\:\(\)\?\-]+", " ", str(q)); return [t for t in q.split() if t.lower() not in stop and len(t) > 1]


def interval(q):
    mm = re.search(r"(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)", str(q))
    if not mm: return None
    h1, m1, p1, h2, m2, p2 = mm.groups()
    t1 = (int(h1) % 12 + (12 if p1 == "PM" else 0)) * 60 + int(m1); t2 = (int(h2) % 12 + (12 if p2 == "PM" else 0)) * 60 + int(m2)
    return (t2 - t1) % 1440


rows = []
for c in range(lab.max() + 1):
    m = np.flatnonzero(lab == c); qs = meta.question.values[m]; cats = meta.category.values[m]; cc = collections.Counter(cats)
    tok = collections.Counter(t for q in qs[:4000] for t in tokens(q))
    days = [int(x) for q in qs for x in re.findall(r"March (\d+)", str(q))]
    iv = collections.Counter(x for q in qs[:3000] if (x := interval(q)) is not None)
    rows.append(dict(cluster=c, n=len(m), fills=int(fills[m].sum()), wallets_med=int(np.median(deg[m])), cx=float(XM[m, 0].mean()), cy=float(XM[m, 1].mean()),
                     top_cat=cc.most_common(1)[0][0], purity=cc.most_common(1)[0][1] / len(m), cats="; ".join(f"{k} {v / len(m):.2f}" for k, v in cc.most_common(3)),
                     updown_share=float(np.mean(["Up or Down" in str(q) for q in qs])), day_p10=np.percentile(days, 10) if days else None, day_p90=np.percentile(days, 90) if days else None,
                     interval_min=iv.most_common(1)[0][0] if iv else None, tokens=", ".join(k for k, v in tok.most_common(8)),
                     examples=" | ".join(str(meta.question.values[i])[:60] for i in m[::max(1, len(m) // 3)][:3])))
df = pd.DataFrame(rows).sort_values("n", ascending=False)
df.to_csv(f"{OUT}/market_clusters{suffix}.csv", index=False); np.save(f"{OUT}/market_cluster_labels{suffix}.npy", lab)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df[["cluster", "n", "fills", "top_cat", "purity", "cats", "updown_share", "day_p10", "day_p90", "interval_min"]].head(45).to_string(index=False))
print(); print(df[["cluster", "n", "tokens", "examples"]].head(45).to_string(index=False))
