"""
compare_runs.py

Multi-seed comparison of the full model (LOB + graph) against the LOB-only
ablation, with an hour-block bootstrap on the difference.

    python compare_runs.py --tag lb                 # val_history_<H>_lb_s<seed>.json vs ..._lb_ablation_s<seed>.json
    python compare_runs.py --tag lb --horizons 1h 6h

For each horizon it prints, on all samples and on the deduplicated subset:
  * per-seed balanced accuracy / macro-F1 / accuracy of both variants, mean ± std
  * the paired difference full − ablation per seed
  * a 95% bootstrap CI of the difference in balanced accuracy, resampling the
    validation HOURS (not samples) with replacement and pooling the seeds that
    exist for both variants.  Samples within an hour are strongly dependent,
    so the hour is the smallest unit that can be treated as roughly
    exchangeable.  With one validation day there are only 24 such units: the
    interval is honest but wide, and cannot rule out day-specific effects.
  * the causal baselines and the seed-0 runs without lookback features, for reference.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from polymarket_gnn.evaluation.metrics import cm_to_metrics

HOUR_NS = 3_600_000_000_000


def load_hourly_cms(path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """{'all'|'dedup': {hour_key: 3x3 cm}} from a val_history file."""
    with open(path) as f:
        data = json.load(f)
    out = {"all": {}, "dedup": {}}
    for d in data:
        hk = int(d["t"]) // HOUR_NS
        for key, field in (("all", "cm"), ("dedup", "cm_dedup")):
            if field in d:
                cm = np.asarray(d[field], dtype=np.float64)
                out[key][hk] = out[key].get(hk, 0) + cm
    return out


def find_runs(directory: Path, horizon: str, tag: str, ablation: bool) -> Dict[int, Path]:
    mid = f"_{tag}" if tag else ""
    suffix = "_ablation" if ablation else ""
    runs = {}
    for p in glob.glob(str(directory / f"val_history_{horizon}{mid}{suffix}_s*.json")):
        m = re.search(rf"val_history_{re.escape(horizon)}{re.escape(mid)}{suffix}_s(\d+)\.json$", p)
        if m:
            runs[int(m.group(1))] = Path(p)
    return runs


def total_cm(hourly: Dict[int, np.ndarray]) -> np.ndarray:
    return sum(hourly.values()) if hourly else np.zeros((3, 3))


def bal_acc(cm: np.ndarray) -> float:
    rows = cm.sum(axis=1)
    return float(np.mean([cm[i, i] / rows[i] if rows[i] > 0 else 0.0 for i in range(3)]))


def bootstrap_diff(full: List[Dict[int, np.ndarray]], abl: List[Dict[int, np.ndarray]],
                   n_boot: int = 5000, seed: int = 0):
    """95% CI of bal_acc(full) − bal_acc(ablation), resampling hours, pooling seeds."""
    hours = sorted(set.intersection(*[set(h) for h in full + abl]))
    if len(hours) < 2:
        return None
    f = np.stack([sum(run[h] for run in full) for h in hours])   # (n_hours, 3, 3)
    a = np.stack([sum(run[h] for run in abl) for h in hours])
    rng = np.random.default_rng(seed)
    point = bal_acc(f.sum(0)) - bal_acc(a.sum(0))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(hours), size=len(hours))
        diffs[b] = bal_acc(f[idx].sum(0)) - bal_acc(a[idx].sum(0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_hours": len(hours),
            "p_le_0": float((diffs <= 0).mean())}


def fmt(vals: List[float]) -> str:
    if not vals:
        return "   n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lb", help="run tag between horizon and seed ('' for none)")
    ap.add_argument("--horizons", nargs="+", default=["1h", "6h", "12h"])
    ap.add_argument("--dir", type=Path, default=Path("."))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = {}
    print("\n" + "=" * 108)
    print(f"FULL (LOB + graph) vs ABLATION (LOB only) — tag='{args.tag}' — multi-seed")
    print("=" * 108)
    for H in args.horizons:
        full_paths = find_runs(args.dir, H, args.tag, ablation=False)
        abl_paths = find_runs(args.dir, H, args.tag, ablation=True)
        print(f"\n{'─' * 100}\n  HORIZON {H}   full seeds={sorted(full_paths)}   ablation seeds={sorted(abl_paths)}\n{'─' * 100}")
        if not full_paths and not abl_paths:
            print("  (no runs found)")
            continue
        full = {s: load_hourly_cms(p) for s, p in full_paths.items()}
        abl = {s: load_hourly_cms(p) for s, p in abl_paths.items()}
        # a run still in training has an empty val_history: do not report it as 0.0
        pending = sorted([f"full s{s}" for s, r in full.items() if not r["all"]] +
                         [f"ablation s{s}" for s, r in abl.items() if not r["all"]])
        full = {s: r for s, r in full.items() if r["all"]}
        abl = {s: r for s, r in abl.items() if r["all"]}
        if pending:
            print(f"  (still running, no validation yet: {', '.join(pending)})")
        common = sorted(set(full) & set(abl))
        report[H] = {}
        for subset in ("all", "dedup"):
            print(f"\n  [{subset}]")
            rows = {}
            for name, runs in (("full", full), ("ablation", abl)):
                ms = {s: cm_to_metrics(total_cm(r[subset])) for s, r in runs.items()}
                rows[name] = ms
                for s in sorted(ms):
                    m = ms[s]
                    print(f"    {name:<9} seed {s}: bal_acc={m['balanced_accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  n={m['n']}")
                print(f"    {name:<9} mean   : bal_acc={fmt([m['balanced_accuracy'] for m in ms.values()])}   "
                      f"macroF1={fmt([m['macro_f1'] for m in ms.values()])}   acc={fmt([m['accuracy'] for m in ms.values()])}")
            entry = {k: {str(s): {"balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
                         for s, m in v.items()} for k, v in rows.items()}
            if common:
                d = [rows["full"][s]["balanced_accuracy"] - rows["ablation"][s]["balanced_accuracy"] for s in common]
                print(f"    Δ bal_acc (full − ablation) per seed {common}: {[round(x, 4) for x in d]}   mean={np.mean(d):+.4f}")
                bs = bootstrap_diff([full[s][subset] for s in common], [abl[s][subset] for s in common], args.n_boot)
                if bs:
                    verdict = "graph helps" if bs["lo"] > 0 else ("graph hurts" if bs["hi"] < 0 else "not distinguishable from 0")
                    print(f"    hour-block bootstrap ({bs['n_hours']} hours, seeds pooled): Δ={bs['point']:+.4f}  "
                          f"95% CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]  → {verdict}")
                    entry["bootstrap"] = bs
                entry["delta_per_seed"] = dict(zip(map(str, common), d))
            report[H][subset] = entry

        # reference rows
        print("\n  [reference, all samples]")
        bl_path = args.dir / f"baseline_results_{H}.json"
        if bl_path.exists():
            bl = json.load(open(bl_path))
            if bl.get("meta", {}).get("format") == 2:
                for name in ("Mean Reversion H (causal)", "Momentum H (causal)", "Always Constant"):
                    m = bl["baselines"][name]["all"]
                    md = bl["baselines"][name]["dedup"]
                    print(f"    {name:<28} bal_acc={m['balanced_accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}   [dedup bal_acc={md['balanced_accuracy']:.4f}]")
        for label, fname in (("no-lookback full (seed 0)", f"val_history_{H}.json"),
                             ("no-lookback ablation (seed 0)", f"val_history_{H}_ablation.json")):
            p = args.dir / fname
            if p.exists():
                h = load_hourly_cms(p)
                m, md = cm_to_metrics(total_cm(h["all"])), cm_to_metrics(total_cm(h["dedup"]))
                print(f"    {label:<28} bal_acc={m['balanced_accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}   [dedup bal_acc={md['balanced_accuracy']:.4f}]")
    print("\n" + "=" * 108 + "\n")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
