"""
baseline_evaluation.py

Causal (leakage-free) baselines for the 3-class mid-price direction task.

Usage:
    python baseline_evaluation.py --horizon 3600     # 1 hour
    python baseline_evaluation.py --horizon 21600    # 6 hours
    python baseline_evaluation.py --horizon 43200    # 12 hours

Sample definition (identical to train_full.py): one sample per (block, market)
where the market is OPEN per the lifecycle file, has a LOB snapshot in
[t_prev_block, t_block), is in the market index mapping, and has a label
(mid price observable at t and at t+H, see MidPriceIndex.labels).

Baselines (all strictly causal w.r.t. t):
  Always Constant               predict CONSTANT
  Majority Class (train prior)  predict the most frequent class on --prior-days
  Random Uniform                uniform over the 3 classes
  Class Prior (train)           sample from the --prior-days class distribution
  Momentum H (causal)           sign(mid(t) - mid(t - H))        (horizon-matched)
  Momentum last-tick (causal)   sign of the last observed snapshot-to-snapshot change
  Mean Reversion H (causal)     opposite of Momentum H

The previous version of this script fed the *true label of the previous block*
back as the "momentum" prediction.  That label spans (t_prev, t_prev + H),
i.e. almost the entire future window of the current sample, which is why it
scored 93-95%.  It was not a baseline, it was the answer key.

Metrics are reported on all samples and on a deduplicated subset (one sample
per market every --dedup-minutes), see DedupTracker.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from polymarket_gnn.data.price_index import LifecycleTracker, MidPriceIndex
from polymarket_gnn.evaluation.baselines import (
    CONST, DedupTracker, mean_reversion_from_change, momentum_from_change, parse_days,
)
from polymarket_gnn.evaluation.metrics import (
    CLASS_NAMES, N_CLASSES, cm_to_metrics, confusion_matrix, format_metrics_row,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("/data/v.pampinella/agy_project/data/derived")
YEAR, MONTH = 2026, 3


def txn_blocks_for_hour(base_dir: Path, year: int, month: int, day: int, hour: int) -> np.ndarray:
    """Sorted int64 ns timestamps of the blocks of one hour (first tx timestamp per block)."""
    txn_path = base_dir / f"partitioned/transactions/year={year}/month={month}/day={day}/hour={hour}"
    df = duckdb.query(
        f"SELECT block_number, min(timestamp) AS ts FROM read_parquet('{txn_path}/**/*.parquet') "
        f"GROUP BY block_number ORDER BY block_number"
    ).df()
    if df.empty:
        return np.empty(0, dtype=np.int64)
    return df["ts"].values.astype("datetime64[ns]").astype(np.int64)


def iter_blocks(
    base_dir: Path,
    days: List[int],
    index: MidPriceIndex,
    lifecycle: LifecycleTracker,
    known_markets: set,
) -> Iterator[Tuple[int, int, List[str]]]:
    """Yields (t_prev_ns, t_ns, active_market_ids) per block, mirroring train_full.py."""
    prev = None
    for day in days:
        for hour in range(24):
            try:
                block_ts = txn_blocks_for_hour(base_dir, YEAR, MONTH, day, hour)
            except Exception as e:  # missing hour directory
                logger.warning(f"No transactions for {YEAR}-{MONTH:02d}-{day:02d} {hour:02d}:00: {e}")
                continue
            if len(block_ts) == 0:
                continue
            index.set_hour(YEAR, MONTH, day, hour)
            hour_start = pd.Timestamp(year=YEAR, month=MONTH, day=day, hour=hour).value
            if prev is None:
                prev = hour_start
                lifecycle.reset_to(hour_start)
            for t_ns in block_ts.tolist():
                if t_ns <= prev:
                    prev = max(prev, t_ns)
                    continue
                lifecycle.advance(t_ns)
                active = [m for m in index.active_between(prev, t_ns) if m in known_markets]
                yield prev, t_ns, active
                prev = t_ns


def main():
    parser = argparse.ArgumentParser(description="Causal baseline evaluation")
    parser.add_argument("--horizon", type=int, default=3600, help="prediction horizon in seconds")
    parser.add_argument("--val-days", type=str, default="13", help="validation days of March 2026, e.g. '13' or '21-23'")
    parser.add_argument("--prior-days", type=str, default="12", help="days used to estimate the class prior (must precede val)")
    parser.add_argument("--const-eps", type=float, default=0.0, help="|Δmid| <= eps → CONSTANT")
    parser.add_argument("--dedup-minutes", type=float, default=10.0, help="min gap between two samples of the same market in the dedup metrics")
    parser.add_argument("--close-filter", choices=["last_tx", "none"], default="last_tx",
                        help="'last_tx' drops samples whose market's last trade precedes t+H (selection on a future event, kept for comparability); 'none' keeps them")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out", type=Path, default=None, help="output JSON (default baseline_results_<label>.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=int, default=None, help="debug: stop each pass after N hours")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    horizon_ns = args.horizon * 10**9
    horizon_hours = int(np.ceil(args.horizon / 3600))
    label = f"{args.horizon // 3600}h" if args.horizon >= 3600 else f"{args.horizon}s"
    val_days = parse_days(args.val_days)
    prior_days = parse_days(args.prior_days)
    if max(prior_days) >= min(val_days):
        raise SystemExit("--prior-days must strictly precede --val-days (no peeking at the evaluation period)")
    out_path = args.out or Path(f"baseline_results_{label}.json")
    logger.info(f"Horizon {label} ({args.horizon}s) | val days {val_days} | prior days {prior_days} | eps={args.const_eps} | close_filter={args.close_filter}")

    lifecycle = LifecycleTracker(args.base_dir / "market_lifecycle_events.parquet")
    index = MidPriceIndex(
        args.base_dir, hours_back=horizon_hours, hours_ahead=horizon_hours,
        lifecycle=lifecycle, close_filter=args.close_filter, const_eps=args.const_eps,
    )
    known_markets = set(pd.read_parquet(args.base_dir / "id_mappings/market_to_idx.parquet")["market_id"])

    # ── Pass 1: class prior on the training period ───────────────────────────
    t0 = time.time()
    prior_counts = np.zeros(N_CLASSES, dtype=np.int64)
    n_hours, last_hour = 0, None
    for prev, t_ns, active in iter_blocks(args.base_dir, prior_days, index, lifecycle, known_markets):
        if index._current_key != last_hour:
            last_hour = index._current_key
            n_hours += 1
            if args.max_hours and n_hours > args.max_hours:
                break
        if not active:
            continue
        y = index.labels(active, t_ns, horizon_ns)
        y = y[~np.isnan(y)].astype(np.int64)
        prior_counts += np.bincount(y, minlength=N_CLASSES)
    prior = prior_counts / max(prior_counts.sum(), 1)
    majority = int(np.argmax(prior_counts))
    logger.info(
        f"Prior pass done in {time.time() - t0:.0f}s: counts={prior_counts.tolist()} "
        f"prior={dict(zip(CLASS_NAMES, np.round(prior, 4).tolist()))} majority={CLASS_NAMES[majority]}"
    )

    # ── Pass 2: evaluate baselines on validation ─────────────────────────────
    names = [
        "Always Constant",
        "Majority Class (train prior)",
        "Random Uniform",
        "Class Prior (train)",
        "Momentum H (causal)",
        "Momentum last-tick (causal)",
        "Mean Reversion H (causal)",
    ]
    cms_all = {n: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for n in names}
    cms_dedup = {n: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for n in names}
    dedup = DedupTracker(int(args.dedup_minutes * 60 * 10**9))
    n_blocks = n_samples = 0
    n_mom_unavailable = 0
    t0 = time.time()
    n_hours, last_hour = 0, None
    for prev, t_ns, active in iter_blocks(args.base_dir, val_days, index, lifecycle, known_markets):
        if index._current_key != last_hour:
            last_hour = index._current_key
            n_hours += 1
            if args.max_hours and n_hours > args.max_hours:
                break
            logger.info(f"Evaluating hour {pd.Timestamp(last_hour * 3_600_000_000_000)} | blocks so far {n_blocks} | samples {n_samples}")
        n_blocks += 1
        if not active:
            continue
        y_all = index.labels(active, t_ns, horizon_ns)
        valid = ~np.isnan(y_all)
        if not valid.any():
            continue
        act = [m for m, v in zip(active, valid) if v]
        y = y_all[valid].astype(np.int64)
        n = len(y)
        n_samples += n

        past_h = index.past_change(act, t_ns, horizon_ns)
        last_tick = index.last_change(act, t_ns)
        n_mom_unavailable += int(np.isnan(past_h).sum())

        preds = {
            "Always Constant": np.full(n, CONST, dtype=np.int64),
            "Majority Class (train prior)": np.full(n, majority, dtype=np.int64),
            "Random Uniform": rng.integers(0, N_CLASSES, size=n),
            "Class Prior (train)": rng.choice(N_CLASSES, size=n, p=prior),
            "Momentum H (causal)": momentum_from_change(past_h, args.const_eps),
            "Momentum last-tick (causal)": momentum_from_change(last_tick, args.const_eps),
            "Mean Reversion H (causal)": mean_reversion_from_change(past_h, args.const_eps),
        }
        keep = dedup.keep_mask(act, t_ns)
        for name, p in preds.items():
            cms_all[name] += confusion_matrix(y, p)
            if keep.any():
                cms_dedup[name] += confusion_matrix(y[keep], p[keep])

    logger.info(f"Evaluation pass done in {time.time() - t0:.0f}s: {n_blocks} blocks, {n_samples} samples")

    # ── Report ───────────────────────────────────────────────────────────────
    results = {
        "meta": {
            "horizon_s": args.horizon, "label": label,
            "val_days": val_days, "prior_days": prior_days,
            "const_eps": args.const_eps, "close_filter": args.close_filter,
            "dedup_minutes": args.dedup_minutes, "seed": args.seed,
            "prior_counts": prior_counts.tolist(), "majority_class": CLASS_NAMES[majority],
            "n_blocks": n_blocks, "n_samples": n_samples,
            "momentum_H_unavailable_frac": n_mom_unavailable / max(n_samples, 1),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "format": 2,
        },
        "baselines": {
            name: {"all": cm_to_metrics(cms_all[name]), "dedup": cm_to_metrics(cms_dedup[name])}
            for name in names
        },
    }
    print(f"\n{'=' * 100}")
    print(f"CAUSAL BASELINES — horizon {label} — validation days {val_days} (March 2026) — eps={args.const_eps}")
    print(f"{'=' * 100}")
    for subset in ("all", "dedup"):
        print(f"\n[{subset}]  n={results['baselines'][names[0]][subset]['n']}")
        for name in names:
            print(format_metrics_row(name, results["baselines"][name][subset]))
    print()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
