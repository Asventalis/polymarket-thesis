"""
verify_optimizations.py

Checks on REAL data that the two performance optimisations do not change behaviour.

  buffer   FastLOBWindowBuffer vs LOBWindowBuffer, block by block over consecutive
           hours (block times from the transactions, hour reloads as in training):
           same active markets, bit-identical windows.
               python verify_optimizations.py buffer --day 6 --hours 0,1

  engine   Dense CausalDynamicGNN.forward vs the row-sparse engine in one process, on real
           blocks, with identical parameters and dropout masks; prints the per-block logit
           difference, the gradient difference at each TBPTT segment and the final state
           difference.  In float64 they agree to ~1e-15.
               python verify_optimizations.py engine --dtype f64 --mode train --blocks 21

  compare  Two training runs (e.g. dense+reference vs sparse+fast, same seed):
           per-block loss and confusion matrices of the training and validation
           histories.
               python verify_optimizations.py compare --a ref --b new
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path("/data/v.pampinella/agy_project/data/derived")


def lob_path(day, hour):
    ts = pd.Timestamp(year=2026, month=3, day=day, hour=hour)
    return BASE / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"


def buffer_paths(ts):
    prev = ts - pd.Timedelta(hours=1)
    p = BASE / f"tensors/year={prev.year}/month={prev.month}/day={prev.day}/hour={prev.hour}/lob_features.parquet"
    cur = BASE / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"
    return ([p] if p.exists() else []) + [cur]


def cmd_buffer(args):
    from polymarket_gnn.data.lob_window_buffer import FastLOBWindowBuffer, LOBWindowBuffer
    hours = [int(h) for h in args.hours.split(",")]
    first = pd.Timestamp(year=2026, month=3, day=args.day, hour=hours[0])
    t0 = time.time(); ref = LOBWindowBuffer.from_parquet(buffer_paths(first), K=args.window); t_ref_load = time.time() - t0
    t0 = time.time(); fast = FastLOBWindowBuffer.from_parquet(buffer_paths(first), K=args.window); t_fast_load = time.time() - t0
    ref.reset_to_time(first.value); fast.reset_to_time(first.value)
    assert ref._open_markets == fast._open_markets
    prev = first.value
    n_blocks = n_samples = n_padded = 0
    t_ref = t_fast = 0.0
    for h in hours:
        ts = pd.Timestamp(year=2026, month=3, day=args.day, hour=h)
        if ts != first:
            t0 = time.time(); ref.reload_lob_data(buffer_paths(ts)); t_ref_load += time.time() - t0
            t0 = time.time(); fast.reload_lob_data(buffer_paths(ts)); t_fast_load += time.time() - t0
        txn = BASE / f"partitioned/transactions/year=2026/month=3/day={args.day}/hour={h}"
        blocks = duckdb.query(f"SELECT block_number, min(timestamp) ts FROM read_parquet('{txn}/**/*.parquet') GROUP BY 1 ORDER BY 1").df()
        for t_ns in blocks["ts"].values.astype("datetime64[ns]").astype(np.int64).tolist():
            if t_ns <= prev:
                continue
            t0 = time.time(); a_ref, w_ref = ref.load_step(prev, t_ns); t_ref += time.time() - t0
            t0 = time.time(); a_fast, w_fast = fast.load_step(prev, t_ns); t_fast += time.time() - t0
            if sorted(a_ref) != sorted(a_fast):
                raise SystemExit(f"ACTIVE SET MISMATCH at {pd.Timestamp(t_ns)}: {len(a_ref)} vs {len(a_fast)}")
            if a_ref:
                o_ref, o_fast = np.argsort(a_ref), np.argsort(a_fast)
                if not np.array_equal(w_ref[o_ref], w_fast[o_fast]):
                    raise SystemExit(f"WINDOW MISMATCH at {pd.Timestamp(t_ns)}")
                n_samples += len(a_ref)
                n_padded += int((np.abs(w_ref).sum(axis=2) == 0).any(axis=1).sum())
            prev = t_ns
            n_blocks += 1
        assert ref._open_markets == fast._open_markets
    print(f"OK day {args.day} hours {hours}: {n_blocks} blocks, {n_samples} windows bit-identical "
          f"({n_padded} with zero padding) | load_step: reference {1000 * t_ref / n_blocks:.1f} ms/block, "
          f"fast {1000 * t_fast / n_blocks:.2f} ms/block | loading: reference {t_ref_load:.1f}s, fast {t_fast_load:.1f}s")


def cmd_compare(args):
    for kind in ("loss", "val"):
        a = json.load(open(f"{kind}_history_{args.a}.json")); b = json.load(open(f"{kind}_history_{args.b}.json"))
        assert len(a) == len(b) and len(a) > 0, f"{kind}: {len(a)} vs {len(b)} entries"
        assert [e["t"] for e in a] == [e["t"] for e in b]
        la, lb = np.array([e["loss"] for e in a]), np.array([e["loss"] for e in b])
        cma, cmb = np.array([e["cm"] for e in a]), np.array([e["cm"] for e in b])
        n = cma.sum()
        moved = np.abs(cma - cmb).sum() / 2
        rel = np.abs(la - lb) / np.maximum(np.abs(la), 1e-12)
        print(f"{kind:>4}: {len(a)} blocks, {int(n)} samples | loss: max |Δ|={np.abs(la - lb).max():.2e} "
              f"max rel={rel.max():.2e} mean rel={rel.mean():.2e} | predictions that differ: {int(moved)} of {int(n)} "
              f"({100 * moved / n:.4f}%) | same n per block: {bool((cma.sum((1, 2)) == cmb.sum((1, 2))).all())}")


def cmd_engine(args):
    """Dense forward vs sparse engine in ONE process: same parameters, same real inputs, same dropout masks."""
    import torch
    from polymarket_gnn.data.graph_builder import EDGE_DIM, EdgeBuilder
    from polymarket_gnn.data.lob_window_buffer import FastLOBWindowBuffer, OUTPUT_DIM
    from polymarket_gnn.models.causal_gnn import CausalDynamicGNN
    from polymarket_gnn.models.sparse_state import DecayClock, LazyNodeState, sparse_step
    torch.set_num_threads(8)
    B = BASE
    DT = torch.float64 if args.dtype == "f64" else torch.float32
    MODE = args.mode
    NB = args.blocks
    mm = pd.read_parquet(B / "id_mappings/market_to_idx.parquet"); wm = pd.read_parquet(B / "id_mappings/wallet_to_idx.parquet")
    m2i = dict(zip(mm["market_id"], mm["market_idx"])); w2i = dict(zip(wm["wallet"], wm["wallet_idx"]))
    torch.manual_seed(0)
    feat = torch.load(B / "market_features.pt", map_location="cpu")
    proj = torch.nn.Linear(feat.shape[1], 128)
    with torch.no_grad():
        h_m = torch.clamp(torch.nan_to_num(proj(feat), nan=0.0, posinf=0.0, neginf=0.0), -10, 10).to(DT)
    h_w = (torch.randn(len(w2i), 128) * 0.01).to(DT)
    model = CausalDynamicGNN(lob_dim=OUTPUT_DIM, hidden_dim=128, edge_dim=EDGE_DIM).to(DT)
    model.train(MODE == "train")
    clock = DecayClock("cpu"); sm, sw = LazyNodeState(h_m, clock), LazyNodeState(h_w, clock)   # default (learned) decay
    eb = EdgeBuilder(m2i, w2i)
    first = pd.Timestamp("2026-03-06 00:00")
    buf = FastLOBWindowBuffer.from_parquet([B / "tensors/year=2026/month=3/day=6/hour=0/lob_features.parquet"], K=10)
    buf.reset_to_time(first.value)
    txn = duckdb.query(f"SELECT * FROM read_parquet('{B}/partitioned/transactions/year=2026/month=3/day=6/hour=0/**/*.parquet') ORDER BY block_number, timestamp").df()
    prev = first.value; nb = 0
    tb_d = torch.zeros((), dtype=DT); tb_s = torch.zeros((), dtype=DT)
    for bn, bt in txn.groupby("block_number", sort=True):
        t = int(bt["timestamp"].iloc[0].value)
        if t <= prev: continue
        ids, w = buf.load_step(prev, t)
        keep = sorted((i for i, m in enumerate(ids) if m in m2i), key=lambda i: m2i[ids[i]])
        ids = [ids[i] for i in keep]; w = w[keep]
        aidx = torch.tensor([m2i[m] for m in ids], dtype=torch.long)
        win = torch.clamp(torch.nan_to_num(torch.tensor(w, dtype=DT)), -100, 100) if ids else torch.empty((0, 10, OUTPUT_DIM), dtype=DT)
        ei, ef = eb(bt); ef = torch.clamp(torch.nan_to_num(ef), -100, 100).to(DT)
        dt = (t - prev) / 1e9
        y = torch.randint(0, 3, (len(aidx),), generator=torch.Generator().manual_seed(nb))
        with torch.set_grad_enabled(MODE == "train"):
            st = torch.get_rng_state()
            dtt = torch.tensor(dt, dtype=DT)
            ld, h_m, h_w = model(win, aidx, h_m.clone(), h_w.clone(), ei, ef, dtt, dtt, aidx)
            torch.set_rng_state(st)                      # identical dropout masks for the sparse pass
            ls = sparse_step(model, clock, sm, sw, win, aidx, ei, ef, dt, aidx)
        d = (ld - ls).abs().max().item() if len(aidx) else 0.0
        n_dup = ei.size(1) - torch.unique(ei, dim=1).size(1) if ei.size(1) else 0
        if nb < 6 or nb % 20 == 0 or d > 1e-3:
            print(f"block {nb:3d} dt={dt:4.1f} act={len(aidx):3d} E={ei.size(1):4d} dupE={n_dup:3d} max|Δlogit|={d:.2e} |logit|max={ld.abs().max().item() if len(aidx) else 0:.2f}")
        if MODE == "train" and len(aidx):
            tb_d = tb_d + torch.nn.functional.cross_entropy(ld, y); tb_s = tb_s + torch.nn.functional.cross_entropy(ls, y)
        if nb > 0 and nb % 5 == 0:
            if MODE == "train" and tb_d.requires_grad:
                model.zero_grad(); tb_d.backward(); gd = [p.grad.clone() for p in model.parameters()]
                model.zero_grad(); tb_s.backward(); gs = [p.grad.clone() for p in model.parameters()]
                rel = max(((a - b).abs().max() / (a.abs().max() + 1e-30)).item() for a, b in zip(gd, gs))
                print(f"   segment end block {nb}: max relative grad diff = {rel:.2e}")
                tb_d = torch.zeros((), dtype=DT); tb_s = torch.zeros((), dtype=DT)
            h_m = torch.nan_to_num(h_m.detach(), nan=0.0); h_w = torch.nan_to_num(h_w.detach(), nan=0.0)
            sm.commit(); sw.commit(); clock.commit()
        prev = t; nb += 1
        if nb >= NB: break
    dm = (h_m - sm.materialize()).abs().max().item(); dw = (h_w - sw.materialize()).abs().max().item()
    print(f"FINAL state max|Δ|: markets {dm:.2e} wallets {dw:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buffer"); b.add_argument("--day", type=int, default=13); b.add_argument("--hours", default="5,6"); b.add_argument("--window", type=int, default=10)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    e = sub.add_parser("engine"); e.add_argument("--dtype", choices=["f32", "f64"], default="f64")
    e.add_argument("--mode", choices=["eval", "train"], default="train"); e.add_argument("--blocks", type=int, default=21)
    args = ap.parse_args()
    {"buffer": cmd_buffer, "compare": cmd_compare, "engine": cmd_engine}[args.cmd](args)
