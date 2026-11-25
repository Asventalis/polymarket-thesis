"""Animate a Polymarket order book from replayed frames (JSON from vp_replay.py)."""
import argparse, json, bisect
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from datetime import datetime

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BID, ASK = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated)

ap = argparse.ArgumentParser()
ap.add_argument("json"); ap.add_argument("out")
ap.add_argument("--step", type=int, default=2, help="seconds of data per video frame")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--halfwidth", type=float, default=0.06, help="half width of the price window (ignored with --full)")
ap.add_argument("--full", action="store_true", help="fixed price axis from 0 to 1; levels grouped in bins of --bin")
ap.add_argument("--bin", type=float, default=0.005, help="bin width on the price axis with --full")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
ap.add_argument("--still", type=int, default=None, help="render a single frame index to PNG")
ap.add_argument("--width", type=float, default=12.8); ap.add_argument("--height", type=float, default=7.2)
ap.add_argument("--dpi", type=int, default=100)
ap.add_argument("--subtitle", default="")
a = ap.parse_args()

k = a.width / 12.8   # font scale so a narrower figure keeps the same proportions
d = json.load(open(a.json))
frames = d["frames"][a.start:a.end]
T = [datetime.fromisoformat(f["t"]) for f in frames]
bb = np.array([f["bids"][0][0] for f in frames]); ba = np.array([f["asks"][0][0] for f in frames])
mid = (bb + ba) / 2
# smoothed window centre so the price axis glides instead of jumping
centre = np.empty_like(mid); c = mid[0]
for i, m in enumerate(mid):
    c = c + 0.08 * (m - c); centre[i] = c
def binned(levels):
    """sum the sizes of the levels that fall in the same bin; returns {bin centre: size}"""
    agg = {}
    for p, v in levels:
        c = (int(p / a.bin + 1e-9) + 0.5) * a.bin
        agg[c] = agg.get(c, 0.0) + v
    return agg
def compact(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.1f}K" if v >= 1e4 else f"{v:,.0f}")
if a.ymax is None:
    if a.full:
        sz = [v for f in frames for v in list(binned(f["bids"]).values()) + list(binned(f["asks"]).values())]
        a.ymax = float(np.ceil(np.percentile(sz, 90) / 1000) * 1000)
    else:
        sz = [v for f, m in zip(frames, mid) for p, v in f["bids"] + f["asks"] if abs(p - m) <= a.halfwidth]
        y99 = np.percentile(sz, 99); a.ymax = float(np.ceil(y99 / 100) * 100)
tick = a.bin if a.full else 0.001

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11 * k, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(a.width, a.height), dpi=a.dpi, facecolor=SURF)
gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.1], hspace=0.42, left=0.07 / k, right=0.97, top=0.78, bottom=0.09)
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
for axx in (ax, ax2):
    axx.set_facecolor(SURF)
    for s in ("top", "right"): axx.spines[s].set_visible(False)
    axx.grid(axis="y", color=GRID, linewidth=1); axx.set_axisbelow(True)
    axx.tick_params(length=0)

fig.text(0.07 / k, 0.95, d["question"], fontsize=15 * k, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.07 / k, 0.905, a.subtitle, fontsize=10.5 * k, color=INK2, ha="left", va="top")
clock = fig.text(0.97, 0.95, "", fontsize=13 * k, color=INK, ha="right", va="top", family="DejaVu Sans Mono")
quote = fig.text(0.97, 0.905, "", fontsize=10.5 * k, color=INK2, ha="right", va="top", family="DejaVu Sans Mono")

ax.set_ylim(0, a.ymax); ax.set_ylabel("resting size (shares)"); ax.set_xlabel("price (USDC per share)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}" if a.full else f"{v:.3f}"))
if a.full:
    ax.set_xlim(0, 1); ax.set_xticks(np.arange(0, 1.01, 0.1))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BID, label="bids (buy orders)"), Patch(color=ASK, label="asks (sell orders)")],
           loc="upper left", bbox_to_anchor=(0.062 / k, 0.875), ncol=2, frameon=False, fontsize=10 * k, labelcolor=INK2, handlelength=1.2, columnspacing=1.6)
midline = ax.axvline(mid[0], color=MUTED, linewidth=1)
midlab = ax.text(mid[0], a.ymax * 0.97, "", ha="center", va="top", fontsize=9.5 * k, color=INK2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=SURF, edgecolor="none"))
bars = []; caps = []

# lower panel: mid over the window with the bid-ask band and a cursor
ax2.fill_between(T, bb, ba, color=GRID, linewidth=0)
ax2.plot(T, mid, color=INK2, linewidth=2, solid_joinstyle="round")
ax2.set_xlim(T[0], T[-1]); lo, hi = bb.min(), ba.max(); pad = (hi - lo) * 0.08
ax2.set_ylim(lo - pad, hi + pad); ax2.set_ylabel("mid price")
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
cursor = ax2.axvline(T[0], color=MUTED, linewidth=1)
dot, = ax2.plot([T[0]], [mid[0]], marker="o", markersize=9, markerfacecolor=INK2, markeredgecolor=SURF, markeredgewidth=2, linestyle="none")

def draw(i):
    global bars, caps
    for b in bars: b.remove()
    for t in caps: t.remove()
    bars, caps = [], []
    f = frames[i]; c = centre[i]
    if a.full:
        # whole 0-1 axis: bids and asks binned; a bin holding both sides stacks the ask on the bid
        bb_ = binned(f["bids"]); aa_ = binned(f["asks"]); gap = a.ymax * 0.004
        sides = ((bb_, BID, {}), (aa_, ASK, bb_))
        tall = []
        for agg, col, under in sides:
            if not agg: continue
            ps = list(agg); vs = np.array([agg[p] for p in ps])
            bot = np.array([min(under.get(p, 0.0), a.ymax) + (gap if p in under else 0.0) for p in ps])
            cont = ax.bar(ps, np.minimum(vs, a.ymax - bot), bottom=bot, width=tick * 0.78, color=col, linewidth=0, align="center")
            bars.extend(cont.patches)
            tall.extend((p, agg[p] + under.get(p, 0.0)) for p in ps if agg[p] + under.get(p, 0.0) > a.ymax)
        placed = []
        for p, v in sorted(set(tall), key=lambda x: -x[1]):          # label clipped bins, biggest first, no overlaps
            if all(abs(p - q) >= 2.6 * a.bin for q in placed):
                caps.append(ax.text(p, a.ymax * 1.01, compact(v), ha="center", va="bottom", fontsize=8 * k, color=INK2, rotation=90, clip_on=False))
                placed.append(p)
    else:
        ax.set_xlim(c - a.halfwidth, c + a.halfwidth)
        for side, col in ((f["bids"], BID), (f["asks"], ASK)):
            pts = [(p, v) for p, v in side if abs(p - c) <= a.halfwidth + tick]
            if not pts: continue
            ps, vs = zip(*pts)
            cont = ax.bar(ps, np.minimum(vs, a.ymax), width=tick * 0.78, color=col, linewidth=0, align="center")
            bars.extend(cont.patches)
            for p, v in pts:
                if v > a.ymax:
                    caps.append(ax.text(p, a.ymax * 1.01, f"{v:,.0f}", ha="center", va="bottom", fontsize=8 * k, color=INK2, rotation=90, clip_on=False))
    midline.set_xdata([mid[i], mid[i]]); midlab.set_x(mid[i]); midlab.set_text(f"mid {mid[i]:.4f}".rstrip("0"))
    clock.set_text(T[i].strftime("%d %b %Y  %H:%M:%S UTC"))
    quote.set_text(f"best bid {bb[i]:.3f}   best ask {ba[i]:.3f}   spread {ba[i]-bb[i]:.3f}")
    cursor.set_xdata([T[i], T[i]]); dot.set_data([T[i]], [mid[i]])
    return bars

if a.still is not None:
    draw(a.still); fig.savefig(a.out, facecolor=SURF); print("wrote", a.out)
else:
    idx = list(range(0, len(frames), a.step))
    anim = FuncAnimation(fig, draw, frames=idx, blit=False)
    if a.out.endswith(".gif"):
        anim.save(a.out, writer=PillowWriter(fps=a.fps), savefig_kwargs={"facecolor": SURF})
    else:
        anim.save(a.out, writer=FFMpegWriter(fps=a.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]),
                  savefig_kwargs={"facecolor": SURF})
    print("wrote", a.out, len(idx), "frames")
