"""
tsfm_cache.py — step 2 of the foundation-model study: zero-shot Chronos-2 forecasts of the mid
price for EVERY market that is active, on a grid of ``--grid-min`` minutes, over whole days, saved
to a cache that ``train_full.py --tsfm-dir`` turns into input features (``TSFMIndex``).

    PYTHONPATH=/data/v.pampinella/tsfm_pkgs HF_HOME=/data/v.pampinella/hf_cache \
    ./run_dev.sh tsfm_cache.py --horizon 3600 --step-min 1 --steps 15,30,60 --device cuda:3 \
        --days 2026-03-06..2026-03-13 --out-dir /data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h

Per day D and grid time g in [D 00:00, D 24:00): a forecast is made for every market that has at
least one LOB snapshot in [g, g + grid) (the markets the training loop will ask about in that
interval) and at least ``--min-known`` known context points.  The context is ``--context`` mids
sampled every ``--step-min`` minutes, each the last mid strictly before the grid time (NaN where
the market had no snapshot yet), exactly as in ``tsfm_zero_shot.py``; the forecast covers
horizon / step steps and the quantiles 10/50/90 at the steps listed in ``--steps`` (minutes
ahead) are kept.  Which markets get a forecast depends on their activity AFTER g, but the
forecast itself only reads prices before g: a feature read at block t >= g is causal.

Output: ``day=YYYY-MM-DD.parquet`` (market_id, t_grid, mid_last, known, q{Q}_{S} ...) and
``meta.json``.  Days already on disk are skipped (resumable); prints "ALL DONE" at the end.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/v.pampinella/tsfm_pkgs")
os.environ.setdefault("HF_HOME", "/data/v.pampinella/hf_cache")

from polymarket_gnn.data.lob_layout import MID_IDX  # noqa: E402
from polymarket_gnn.data.tsfm_index import DAY_NS, HOUR_NS, MIN_NS, day_file, quantile_columns  # noqa: E402

QS = [0.1, 0.5, 0.9]
QLAB = [10, 50, 90]


def parse_days(spec):
    if ".." in spec:
        a, b = spec.split("..")
        return list(pd.date_range(a, b, freq="D"))
    return [pd.Timestamp(s) for s in spec.split(",")]


def hour_paths(base_dir, k0, k1):
    out = []
    for k in range(k0, k1 + 1):
        ts = pd.Timestamp(k * HOUR_NS)
        p = base_dir / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"
        if p.exists():
            out.append(str(p))
    return out


def load_day(base_dir, day0_ns, step_ns, grid_ns, context):
    """
    Mid matrix M (markets x context grid) and activity matrix A (markets x forecast grid) of one day.
    The context grid runs from day0 - (context - 1) * step to day0 + 24h (every step),
    the forecast grid from day0 to day0 + 24h (every grid).
    """
    cgrid = np.arange(day0_ns - (context - 1) * step_ns, day0_ns + DAY_NS + step_ns, step_ns, dtype=np.int64)
    fgrid = np.arange(day0_ns, day0_ns + DAY_NS, grid_ns, dtype=np.int64)
    paths = hour_paths(base_dir, cgrid[0] // HOUR_NS, (day0_ns + DAY_NS - 1) // HOUR_NS)
    files = ", ".join(f"'{p}'" for p in paths)
    rows = duckdb.query(
        f"SELECT market_id, epoch_ns(timestamp) AS ts, features[{MID_IDX + 1}] AS mid "
        f"FROM read_parquet([{files}]) ORDER BY market_id, ts"
    ).df()
    m_codes, m_uniq = pd.factorize(rows["market_id"].values, sort=True)
    ts, mid = rows["ts"].values.astype(np.int64), rows["mid"].values.astype(np.float32)
    starts = np.searchsorted(m_codes, np.arange(len(m_uniq)), side="left")
    ends = np.searchsorted(m_codes, np.arange(len(m_uniq)), side="right")
    M = np.full((len(m_uniq), len(cgrid)), np.nan, dtype=np.float32)
    A = np.zeros((len(m_uniq), len(fgrid)), dtype=bool)
    for i, (a, b) in enumerate(zip(starts, ends)):
        pos = np.searchsorted(ts[a:b], cgrid, side="left") - 1       # last snapshot strictly before the grid time
        ok = pos >= 0
        M[i, ok] = mid[a:b][pos[ok]]
        lo = np.searchsorted(ts[a:b], fgrid, side="left")
        hi = np.searchsorted(ts[a:b], fgrid + grid_ns, side="left")
        A[i] = hi > lo                                               # a snapshot in [g, g + grid)
    return m_uniq, cgrid, fgrid, M, A, len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, help="forecast horizon in seconds")
    ap.add_argument("--step-min", type=int, default=1, help="context / forecast step in minutes")
    ap.add_argument("--grid-min", type=int, default=1, help="forecast grid in minutes (multiple of --step-min)")
    ap.add_argument("--steps", type=str, default="15,30,60", help="minutes ahead whose quantiles are kept")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--min-known", type=int, default=1, help="minimum known context points for a forecast")
    ap.add_argument("--days", type=str, required=True, help="'2026-03-06..2026-03-13' or comma-separated dates")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--base-dir", type=Path, default=Path("/data/v.pampinella/agy_project/data/derived"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="debug: forecast at most this many (market, grid) pairs per day")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_ns, grid_ns = args.step_min * MIN_NS, args.grid_min * MIN_NS
    assert args.horizon % (args.step_min * 60) == 0 and args.grid_min % args.step_min == 0
    pred_len = args.horizon // (args.step_min * 60)
    steps = [int(s) for s in args.steps.split(",")]
    assert all(s % args.step_min == 0 and 0 < s // args.step_min <= pred_len for s in steps)
    step_idx = [s // args.step_min - 1 for s in steps]
    cols = quantile_columns(steps)                   # q50_S, q10_S, q90_S per step (the order TSFMIndex expects)
    col_q = [QLAB.index(int(c.split("_")[0][1:])) for c in cols]
    col_s = [steps.index(int(c.split("_")[1])) for c in cols]
    meta = {"model": args.model, "horizon": args.horizon, "step_min": args.step_min, "grid_min": args.grid_min,
            "context": args.context, "steps": steps, "quantiles": QLAB, "min_known": args.min_known,
            "columns": cols, "note": "forecast at t_grid uses mids strictly before t_grid; quantiles are price levels"}
    meta_path = args.out_dir / "meta.json"
    if meta_path.exists():
        old = json.load(open(meta_path))
        for k in ("horizon", "step_min", "grid_min", "context", "steps", "model"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"{meta_path} has {k}={old.get(k)!r}, this run {meta[k]!r}: use another --out-dir")
    else:
        json.dump(meta, open(meta_path, "w"), indent=2)

    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(args.threads)
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device, dtype=getattr(torch, args.dtype))
    offs = np.arange(-(args.context - 1), 1)

    for date in parse_days(args.days):
        out_path = day_file(args.out_dir, date)
        if out_path.exists():
            print(f"[{date.date()}] already on disk, skipping", flush=True)
            continue
        t0 = time.time()
        day0 = int(date.normalize().value)
        m_uniq, cgrid, fgrid, M, A, n_files = load_day(args.base_dir, day0, step_ns, grid_ns, args.context)
        c0 = int(np.searchsorted(cgrid, fgrid[0]))                   # context column of the first forecast grid time
        cstep = grid_ns // step_ns
        known = np.zeros_like(A, dtype=np.int16)
        cum = np.cumsum(~np.isnan(M), axis=1)
        for jg in range(len(fgrid)):
            c_end = c0 + jg * cstep
            known[:, jg] = cum[:, c_end] - (cum[:, c_end - args.context] if c_end - args.context >= 0 else 0)
        rows_i, cols_j = np.nonzero(A & (known >= args.min_known))
        if args.max_pairs:
            rows_i, cols_j = rows_i[:args.max_pairs], cols_j[:args.max_pairs]
        n = len(rows_i)
        print(f"[{date.date()}] {len(m_uniq)} markets from {n_files} hour files, {int(A.sum())} active (market, grid) pairs, "
              f"{n} with a context ({time.time() - t0:.0f}s)", flush=True)
        Q = np.zeros((n, len(cols)), dtype=np.float32)
        mid_last = np.zeros(n, dtype=np.float32)
        t1 = time.time()
        for b in range(0, n, args.batch_size):
            ri, cj = rows_i[b:b + args.batch_size], cols_j[b:b + args.batch_size]
            X = M[ri[:, None], (c0 + cj * cstep)[:, None] + offs[None, :]]           # (B, context), last = mid before g
            mid_last[b:b + len(ri)] = X[:, -1]
            with torch.inference_mode():
                q, _ = pipe.predict_quantiles([X[i] for i in range(len(ri))], prediction_length=pred_len,
                                              quantile_levels=QS, batch_size=len(ri))
            qa = torch.stack([qi[0] for qi in q]).float().cpu().numpy()               # (B, pred_len, 3)
            Q[b:b + len(ri)] = qa[:, step_idx][:, col_s, col_q]
            if (b // args.batch_size) % 50 == 0:
                done = b + len(ri)
                rate = done / max(time.time() - t1, 1e-9)
                print(f"[{date.date()}] {done}/{n} forecasts, {rate:.0f}/s, ETA {(n - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
        df = pd.DataFrame({"market_id": m_uniq[rows_i], "t_grid": fgrid[cols_j], "mid_last": mid_last,
                           "known": known[rows_i, cols_j]})
        for k, c in enumerate(cols):
            df[c] = Q[:, k]
        df.to_parquet(out_path, index=False)
        print(f"[{date.date()}] saved {n} forecasts to {out_path} in {time.time() - t0:.0f}s total "
              f"(median |q50_{steps[-1]} - mid| = {np.median(np.abs(df[f'q50_{steps[-1]}'] - df['mid_last'])):.4f})", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
