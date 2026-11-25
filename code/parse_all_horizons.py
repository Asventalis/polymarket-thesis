"""
parse_all_horizons.py

Prints the comparison table (CT-DGNN full / ablation / causal baselines) for
the 1h, 6h, 12h horizons from the JSON files in the current directory.

Reads both the new baseline format ("format": 2, metrics on all + dedup
samples) and the legacy one.  Legacy baseline files are flagged: their
"Momentum" / "Mean Reversion" rows used the previous block's true label and
all legacy results (models included) were computed on tensor column 0, which
is not the mid price.
"""
import json
import sys
from pathlib import Path

import numpy as np

from polymarket_gnn.evaluation.metrics import cm_to_metrics, format_metrics_row

HORIZONS = sys.argv[1:] or ["1h", "6h", "12h"]


def model_metrics(path: Path):
    """Aggregate per-block confusion matrices of a val_history file."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    cm_all = np.zeros((3, 3))
    cm_dedup = np.zeros((3, 3))
    has_dedup = False
    losses = []
    for d in data:
        if "cm" in d:
            cm_all += np.array(d["cm"])
        if "cm_dedup" in d:
            cm_dedup += np.array(d["cm_dedup"])
            has_dedup = True
        if "loss" in d:
            losses.append(d["loss"])
    out = {"all": cm_to_metrics(cm_all), "loss": float(np.mean(losses)) if losses else float("nan")}
    out["dedup"] = cm_to_metrics(cm_dedup) if has_dedup else None
    return out


def print_models(label):
    for fname, name in [(f"val_history_{label}.json", "CT-DGNN (LOB + graph)"),
                        (f"val_history_{label}_ablation.json", "CT-DGNN ablation (LOB only)")]:
        m = model_metrics(Path(fname))
        if m is None:
            print(f"  {name:<34} ({fname} not found)")
            continue
        print(format_metrics_row(name, m["all"]) + f"  loss={m['loss']:.4f}")
        if m["dedup"] is not None:
            print(format_metrics_row(name + " [dedup]", m["dedup"]))
        else:
            print(f"  {'':<34} (legacy file: no dedup metrics, targets on tensor column 0 = NOT the mid price)")


def print_baselines(label):
    path = Path(f"baseline_results_{label}.json")
    if not path.exists():
        print(f"  ({path} not found)")
        return
    with open(path) as f:
        bl = json.load(f)
    if bl.get("meta", {}).get("format") == 2:
        meta = bl["meta"]
        print(f"  (val days {meta['val_days']}, prior days {meta['prior_days']}, eps={meta['const_eps']}, "
              f"close_filter={meta['close_filter']}, n={meta['n_samples']})")
        for subset in ("all", "dedup"):
            print(f"  [{subset}]")
            for name, res in bl["baselines"].items():
                print(format_metrics_row(name, res[subset]))
    else:
        print("  LEGACY baseline file — 'Momentum'/'Mean Reversion' here used the previous block's TRUE label (leaky)")
        for name, res in bl.items():
            print(format_metrics_row(name + " [legacy]", cm_to_metrics(res["cm"])))


print("\n" + "=" * 110)
print("CT-DGNN full model vs LOB-only ablation vs causal baselines")
print("=" * 110)
for label in HORIZONS:
    print(f"\n{'─' * 100}\n  HORIZON: {label}\n{'─' * 100}")
    print("\n  [Deep learning models]")
    print_models(label)
    print("\n  [Baselines]")
    print_baselines(label)
print("\n" + "=" * 110 + "\n")
