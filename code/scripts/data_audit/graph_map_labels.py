"""Island labels at the margins of the market map, with leader lines (figures of the thesis).
Step 1 (server, where the layouts live): python graph_map_labels.py centroids layout_anchor_cat.npz market_cluster_labels_cat.npy > islands_cat.json
Step 2 (anywhere): python graph_map_labels.py draw islands_cat.json static_map_light_catmetric.png island_labels_cat.json out_prefix
writes <prefix>_A_margins.png (labels at the margins) and <prefix>_B_translucent.png (translucent boxes on the map)."""
import sys
if sys.argv[1] == "centroids":
    import numpy as np, json, os
    D = "/data/v.pampinella/data_audit/graphviz/"; layout, clusters = sys.argv[2], sys.argv[3]
    lay=np.load(D + layout); xy=lay["xy"]; nW=int(lay["nW"]); XM=xy[nW:]
    lab=np.load(D + clusters); e=np.load(D+"edges.npz"); fills=e["fillsM"]
    lo,hi=np.quantile(xy,0.003,axis=0),np.quantile(xy,0.997,axis=0)
    out={}
    for l in np.unique(lab):
        if l<0: continue
        m=lab==l
        out[int(l)]=dict(x=float(np.median(XM[m,0])),y=float(np.median(XM[m,1])),n=int(m.sum()),fills=float(fills[m].sum()))
    print(json.dumps(dict(lo=lo.tolist(),hi=hi.tolist(),islands=out)))
    sys.exit(0)
import json, numpy as np, matplotlib
SP, ISL, BG, NAMES, PFX = ".", *sys.argv[2:6]
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
d = json.load(open(ISL)); names = json.load(open(NAMES))
W, H = 1440, 1080; margin = 0.03
lo, hi = np.array(d["lo"]), np.array(d["hi"]); span = (hi - lo).max(); c = (lo + hi) / 2; s = min(W, H) * (1 - 2 * margin) / span
def px(x, y): return (x - c[0]) * s + W / 2, (c[1] - y) * s + H / 2
bg = np.asarray(Image.open(BG))
isl = []
for k, name in names.items():
    it = d["islands"].get(k)
    if not it: continue
    x, y = px(it["x"], it["y"])
    if 0 <= x < W and 0 <= y < H: isl.append((x, y, name, it["n"]))
isl.sort(key=lambda t: -t[3])
print(len(isl), "labels placed of", len(names))

# ---- variant A: labels at the margins with leader lines
PAD = 310
import textwrap
wrap = lambda t: "\n".join(textwrap.wrap(t, 24))
fig = plt.figure(figsize=((W + 2 * PAD) / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-PAD, W + PAD); ax.set_ylim(H, 0); ax.axis("off")
ax.imshow(bg, extent=(0, W, H, 0), interpolation="bilinear")
left = sorted([t for t in isl if t[0] < W / 2], key=lambda t: t[1]); right = sorted([t for t in isl if t[0] >= W / 2], key=lambda t: t[1])
def spread(items, top=60, bottom=H - 60, gap=56):
    ys = [t[1] for t in items]
    # push apart so that consecutive labels are at least `gap` apart, keeping them near their islands
    ys = np.clip(np.array(ys, float), top, bottom)
    for _ in range(200):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                m = (ys[i] + ys[i - 1]) / 2; ys[i - 1] = m - gap / 2; ys[i] = m + gap / 2
        ys = np.clip(ys, top, bottom)
    return ys
for items, side in ((left, "L"), (right, "R")):
    ys = spread(items)
    for (x, y, name, n), ly in zip(items, ys):
        lx = -30 if side == "L" else W + 30
        ax.plot([x, lx], [y, ly], color="#666666", lw=0.7, alpha=0.65, zorder=3, solid_capstyle="round")
        ax.plot([x], [y], marker="o", ms=4.5, mfc="white", mec="#333333", mew=1.0, zorder=4)
        ax.text(lx - (8 if side == "L" else -8), ly, wrap(name), fontsize=11, color="#222222", ha="right" if side == "L" else "left", va="center", family="DejaVu Sans", zorder=5, linespacing=1.1)
fig.savefig(f"{PFX}_A_margins.png", dpi=100, facecolor="white"); plt.close(fig)

# ---- variant B: labels on the map, translucent box, slight offset above the island centre
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
ax.imshow(bg, extent=(0, W, H, 0), interpolation="bilinear")
boxes = []
for (x, y, name, n) in isl:
    w_, h_ = 7.2 * len(name) + 16, 26
    for dy in (-34, 34, -70, 70, 0, -106, 106):
        box = (x - w_ / 2, y + dy - h_ / 2, x + w_ / 2, y + dy + h_ / 2)
        if not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes): break
    boxes.append(box)
    ax.text(x, y + dy, name, fontsize=11.5, color="#111111", ha="center", va="center", family="DejaVu Sans", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=(1, 1, 1, 0.55), ec=(0.2, 0.2, 0.2, 0.5), lw=0.8))
    ax.plot([x], [y], marker="o", ms=4, mfc="white", mec="#333333", mew=0.9, zorder=4)
fig.savefig(f"{PFX}_B_translucent.png", dpi=100, facecolor="white"); plt.close(fig)
