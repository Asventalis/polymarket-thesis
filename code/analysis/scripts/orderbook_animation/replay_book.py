"""Replay the order book of one token over a few hours and dump one frame per second.

Runs on the server inside the container, e.g.

    cd /data/v.pampinella/agy_project_dev && ./run_dev.sh replay_book.py \
        --market 0xd4e77ba6f29fc093509d24f508631abd445ecf506bbdc9c4c80e60256a318527 \
        --token 12403602920039269077597917340921667997547115084613238528792639013246536343316 \
        --question "Will no Fed rate cuts happen in 2026?" \
        --day 2026-03-20 --warmup-hour 13 --hour 14 --out /tmp/vp_book_fed.json

The book is rebuilt the way the feature pipeline does it (lob_builder.py): the latest
`book_snapshot` of the token replaces the whole book, every later `price_change` row sets
the resting size at one price (size 0 removes the level).  Prices are rounded to the tick
(0.001) so that a level coming from the snapshot and the same level coming from a change
share one key.  Frames are emitted once per second over the chosen hour, starting from the
first snapshot seen in the warm-up hour or later.
"""
import argparse, json, sys
from datetime import datetime, timedelta
import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/data/v.pampinella/agy_project/data/derived/partitioned")
ap.add_argument("--market", required=True); ap.add_argument("--token", required=True)
ap.add_argument("--question", default="")
ap.add_argument("--day", default="2026-03-20"); ap.add_argument("--warmup-hour", type=int, default=13)
ap.add_argument("--hour", type=int, default=14)
ap.add_argument("--levels", type=int, default=25, help="levels kept per side in each frame")
ap.add_argument("--out", required=True)
a = ap.parse_args()

day = datetime.strptime(a.day, "%Y-%m-%d")
parts = [f"year={day.year}/month={day.month}/day={day.day}/hour={h}" for h in range(a.warmup_hour, a.hour + 1)]
con = duckdb.connect()
ob = " union all ".join(f"select * from read_parquet('{a.base}/orderbook/{p}/*.parquet')" for p in parts)
sn = " union all ".join(f"select * from read_parquet('{a.base}/snapshots/{p}/*.parquet')" for p in parts)
ev = con.execute(f"""select timestamp_received ts, change_price p, change_size s, change_side side, best_bid, best_ask
  from ({ob}) where market_id = '{a.market}' and token_id = '{a.token}' order by ts""").fetchall()
snaps = [(ts, json.loads(d)) for ts, d in con.execute(
    f"select timestamp_received ts, data from ({sn}) where market_id = '{a.market}' order by ts").fetchall()]
snaps = [(ts, d) for ts, d in snaps if d.get("token_id") == a.token]
print("events", len(ev), "snapshots for the token", len(snaps), "side", snaps[0][1].get("side") if snaps else None, file=sys.stderr)

stream = [(ts, 0, ("snap", d)) for ts, d in snaps] + [(r[0], 1, ("ev", r)) for r in ev]
stream.sort(key=lambda x: (x[0], x[1]))          # a snapshot comes before changes at the same instant
bids, asks = {}, {}
t0 = day + timedelta(hours=a.hour); t1 = t0 + timedelta(hours=1)
frames = []; next_t = t0; had_snap = False
def emit(t):
    sb = sorted(bids.items(), key=lambda x: -x[0])[:a.levels]; sa = sorted(asks.items())[:a.levels]
    frames.append({"t": t.isoformat(), "bids": [[round(p, 3), round(v, 2)] for p, v in sb],
                   "asks": [[round(p, 3), round(v, 2)] for p, v in sa]})
for ts, _, (kind, payload) in stream:
    while next_t <= ts and next_t < t1:
        if had_snap: emit(next_t)
        next_t += timedelta(seconds=1)
    if ts >= t1: break
    if kind == "snap":
        bids.clear(); asks.clear()
        bids.update({round(float(p), 3): float(v) for p, v in payload.get("bids", [])})
        asks.update({round(float(p), 3): float(v) for p, v in payload.get("asks", [])})
        had_snap = True
    else:
        _, p, s, side, _, _ = payload
        if p is None or s is None or side is None: continue
        book = bids if side.upper() == "BUY" else asks
        if s <= 0: book.pop(round(float(p), 3), None)
        else: book[round(float(p), 3)] = float(s)
while next_t < t1:
    if had_snap: emit(next_t)
    next_t += timedelta(seconds=1)

out = {"market": a.market, "token": a.token, "question": a.question, "side": snaps[0][1].get("side") if snaps else None,
       "frames": frames,
       "events": [[str(r[0]), float(r[1]) if r[1] is not None else None, str(r[3]),
                   float(r[4]) if r[4] is not None else None, float(r[5]) if r[5] is not None else None] for r in ev]}
json.dump(out, open(a.out, "w"))
print("frames", len(frames), "first", frames[0]["t"] if frames else None, file=sys.stderr)
