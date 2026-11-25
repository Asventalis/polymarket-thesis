"""
train_full.py

Causal, block-by-block training of the CT-DGNN (CausalDynamicGNN) on the
Polymarket wallet-market transaction graph with LOB window features.

Per Polygon block (≈2 s):
    A. LOB update      — markets with new LOB snapshots in [t_prev, t) get a
                         K-snapshot window (55 dims) → GRU → residual state update
    B. transaction MP  — wallet→market→wallet attention message passing on the
                         block's transactions (disabled with --ablation)
    C. prediction      — 3-class direction (0=DOWN, 1=CONSTANT, 2=UP) of the
                         TRUE mid price (tensor column 40) at t + horizon

Targets come from MidPriceIndex, which sees every hour between t and t+H and
returns NaN when t+H is not observable.  Earlier versions read column 0 of the
tensor as the price (it is the relative best-bid deviation) and only loaded
hours h and h+H.

Usage:
    python train_full.py --horizon 3600                 # 1h, full model
    python train_full.py --horizon 21600 --ablation     # 6h, LOB-only
    python train_full.py --horizon 3600 --max-hours 2 --device cuda:1   # smoke test
    python train_full.py --horizon 21600 --lookback-hours 1,6,12 --seed 1   # with long-lookback features
    python train_full.py --horizon 3600 --focal-alpha auto --undersample-ratio 1.5   # imbalance handling
    python train_full.py --horizon 3600 --task regression                  # signed-delta regression + calibrated tau
    python train_full.py ... --init-from model_<label>_final.pt --train-days '' --warm-days 22-23 --val-days 24-26 \
        --run-label <label>_test --save-val-predictions      # evaluation of a checkpoint (same model flags as the run)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from polymarket_gnn.data.graph_builder import EDGE_DIM, EdgeBuilder, load_market_orientation
from polymarket_gnn.data.wallet_skill import BEHAVIOUR_COLUMNS, N_BEHAVIOUR_FEATURES, N_SKILL_FEATURES
from polymarket_gnn.data.lob_layout import FEATURE_DIM, signed_log1p_columns
from polymarket_gnn.data.lob_window_buffer import FastLOBWindowBuffer, LOBWindowBuffer, OUTPUT_DIM
from polymarket_gnn.data.price_index import HOUR_NS, LifecycleTracker, MidPriceIndex
from polymarket_gnn.evaluation.baselines import DedupTracker, parse_days
from polymarket_gnn.evaluation.metrics import N_CLASSES
from polymarket_gnn.models.ema import WeightEMA
from polymarket_gnn.models.sparse_state import DecayClock, HierarchicalMemory, make_lazy_states, sparse_step
from polymarket_gnn.training.objectives import (
    FocalLoss, ThresholdCalibrator, classes_from_delta, classify_delta, default_thresholds,
    inverse_frequency_weights, parse_alpha, undersample_mask,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("/data/v.pampinella/agy_project/data/derived")
YEAR, MONTH = 2026, 3


def lob_path(base_dir: Path, year: int, month: int, day: int, hour: int) -> Path:
    return base_dir / f"tensors/year={year}/month={month}/day={day}/hour={hour}/lob_features.parquet"


def txn_path(base_dir: Path, year: int, month: int, day: int, hour: int) -> Path:
    return base_dir / f"partitioned/transactions/year={year}/month={month}/day={day}/hour={hour}"


def hour_paths_for_buffer(base_dir: Path, ts: pd.Timestamp) -> List[Path]:
    """
    LOB files to keep resident for hour ``ts``: the previous hour (so that the
    K-snapshot windows do not restart from zero padding at every hour boundary)
    and the current hour.  Targets are NOT read from the buffer any more.
    """
    paths = []
    prev = ts - pd.Timedelta(hours=1)
    p_prev = lob_path(base_dir, prev.year, prev.month, prev.day, prev.hour)
    if p_prev.exists():
        paths.append(p_prev)
    paths.append(lob_path(base_dir, ts.year, ts.month, ts.day, ts.hour))
    return paths


def confusion_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor) -> List[List[int]]:
    """3x3 cm[true][pred] with a single device sync."""
    cm = torch.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES)
    return cm.view(N_CLASSES, N_CLASSES).tolist()


def build_schedule(train_days: List[int], val_days: List[int], max_hours: Optional[int],
                   warm_days: Optional[List[int]] = None) -> List[Tuple[int, int, int, int, bool, bool]]:
    """Hours in time order as (year, month, day, hour, is_train, record).

    Training hours learn (is_train) and are not recorded in the validation history; warm-up hours
    (``warm_days``, evaluation from a checkpoint) run the model without learning only to rebuild the
    node states, the memory and the buffers, and are not recorded either; validation hours are recorded.
    """
    train = [(YEAR, MONTH, d, h, True, False) for d in train_days for h in range(24)]
    warm = [(YEAR, MONTH, d, h, False, False) for d in (warm_days or []) for h in range(24)]
    val = [(YEAR, MONTH, d, h, False, True) for d in val_days for h in range(24)]
    if max_hours:
        train, warm, val = train[:max_hours], warm[:max_hours], val[:max_hours]
    return train + warm + val


def main():
    parser = argparse.ArgumentParser(description="CT-DGNN training")
    parser.add_argument("--horizon", type=int, default=3600, help="prediction horizon in seconds (3600, 21600, 43200)")
    parser.add_argument("--ablation", action="store_true", help="disable graph message passing (LOB-only model)")
    parser.add_argument("--train-days", type=str, default="6-12", help="training days of March 2026")
    parser.add_argument("--val-days", type=str, default="13", help="validation days of March 2026")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="evaluation from a checkpoint (model_<label>_final.pt): the weights are loaded and nothing "
                             "is learned; use with --train-days '' and --warm-days, and give --run-label")
    parser.add_argument("--warm-days", type=str, default="",
                        help="days streamed before --val-days without learning and without recording predictions, to "
                             "rebuild the node states, the hierarchical memory and the LOB buffers that a checkpoint "
                             "does not hold (e.g. '22-23' before a test on 24-26)")
    parser.add_argument("--const-eps", type=float, default=0.0, help="|Δmid| <= eps → CONSTANT")
    parser.add_argument("--close-filter", choices=["last_tx", "none"], default="last_tx")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=0, help="width of the predictor's hidden layer(s) (0 = hidden_dim // 2)")
    parser.add_argument("--head-layers", type=int, default=1, help="number of hidden layers of the predictor")
    parser.add_argument("--window", type=int, default=10, help="K LOB snapshots per window")
    parser.add_argument("--tbptt", type=int, default=5, help="blocks per truncated-BPTT segment")
    parser.add_argument("--accum", type=int, default=20, help="TBPTT segments per optimizer step")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="decay of an exponential moving average of the weights along the training pass (updated after "
                             "every optimizer step; the bias-corrected average replaces the weights before the validation "
                             "hours and is what the final checkpoint holds); 0 = off")
    parser.add_argument("--dedup-minutes", type=float, default=10.0)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=int, default=None, help="smoke test: first N train hours + first N val hours")
    parser.add_argument("--epochs", type=int, default=1,
                        help="passes over the training hours before the validation hours; at the start of every pass "
                             "after the first the states, the hierarchical memory, the decay clock and the LOB buffer "
                             "are reset to their initial values and the optimizer is kept (sparse engine only)")
    parser.add_argument("--decay-mode", choices=["learned", "none", "multiscale"], default="learned",
                        help="memory of the node states between events: 'learned' = one learnable scalar decay "
                             "(original); 'none' = no decay, GRU gates only; 'multiscale' = groups of state "
                             "dimensions with fixed half-lives (--decay-half-lives)")
    parser.add_argument("--decay-half-lives", type=str, default="60,3600,86400,inf",
                        help="multiscale: half-lives in seconds of the market-state groups ('inf' = no decay)")
    parser.add_argument("--wallet-half-lives", type=str, default="",
                        help="multiscale: half-lives of the wallet-state groups (default: same as markets)")
    parser.add_argument("--state-engine", choices=["sparse", "dense"], default="sparse",
                        help="'sparse': row-sparse node states with lazy decay (same mathematics, no full-matrix "
                             "copies per block); 'dense': original reference loop with gradient checkpointing")
    parser.add_argument("--lob-buffer", choices=["fast", "reference"], default="fast",
                        help="'fast': vectorised LOB window lookup (bit-identical windows); 'reference': original")
    parser.add_argument("--save-val-predictions", action="store_true",
                        help="write val_predictions_<label>.parquet with one row per validation sample "
                             "(t, market_idx, y_true, y_pred, delta, past_delta, mid, dedup) for per-market analyses")
    parser.add_argument("--max-blocks-per-hour", type=int, default=None, help="debug: process only the first N blocks of each hour")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                        help="'regression': predict the signed change mid(t+H)-mid(t) (Huber loss) and classify it "
                             "with a threshold calibrated on the last training day(s)")
    parser.add_argument("--focal-alpha", type=str, default="2,1,2",
                        help="class weights 'DOWN,CONST,UP', 'none' (unweighted) or 'auto' (inverse of the running "
                             "training class frequency)")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--undersample-ratio", type=float, default=0.0,
                        help="> 0: undersample frequent classes in the loss; each sample of class c is kept with "
                             "probability min(1, ratio * f_min / f_c).  1 = balanced in expectation, 0 = off")
    parser.add_argument("--delta-scale", type=float, default=100.0,
                        help="regression target = scale * Δmid (100 → one cent of probability = 1.0, so typical "
                             "non-zero moves are O(1))")
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber transition point, in scaled units")
    parser.add_argument("--init-tau", type=float, default=0.005, help="regression: threshold used before calibration (raw Δmid)")
    parser.add_argument("--calib-days", type=int, default=1, help="regression: number of final training days used to calibrate tau")
    parser.add_argument("--memory-periods", type=str, default="",
                        help="hierarchical market memory: comma-separated periods in seconds (e.g. '60,3600'). "
                             "Adds one market state per period, each with its own GRU cell: the first is refreshed "
                             "from the per-block state once per period, each next one from the previous level; "
                             "the predictor reads all the states.  Message passing stays on the per-block state.")
    parser.add_argument("--edge-orientation", choices=["token1", "book"], default="token1",
                        help="'token1' = raw token_direction edge feature (+1 for token 1 of every market, original "
                             "behaviour); 'book' = +1 when the traded token is the one the order book / target of the "
                             "market refers to (token 1 in Yes/No markets, token 2 in all the others)")
    parser.add_argument("--edge-skill", choices=["none", "real", "placebo"], default="none",
                        help="append to every edge the causal track record of its wallet at horizon H (past trades "
                             "whose outcome is already realised), pre-computed by build_wallet_skill.py; 'placebo' = "
                             "the record of another wallet (fixed permutation).  Needs --edge-orientation book")
    parser.add_argument("--skill-dir", type=Path, default=Path("wallet_skill"))
    parser.add_argument("--edge-behaviour", action="store_true",
                        help="append to every edge the past behaviour of its wallet (activity, maker share, "
                             "one-sidedness, size, number of markets; no outcomes), from build_wallet_skill.py --behaviour")
    parser.add_argument("--behaviour-dir", type=Path, default=Path("wallet_behaviour"))
    parser.add_argument("--skip-lookback", action="store_true",
                        help="direct linear path from the lookback features to the output, zero-initialised "
                             "(the model then contains Mean Reversion as a special case); needs --lookback-hours")
    parser.add_argument("--skip-init", choices=["zero", "meanrev"], default="zero",
                        help="'meanrev': the direct path starts as Mean Reversion on the lookback closest to the horizon")
    parser.add_argument("--lookback-logodds", action="store_true",
                        help="also append the log-odds price features (level and change per lookback) to every "
                             "LOB window row; they enter the direct path too.  Needs --lookback-hours")
    parser.add_argument("--tsfm-dir", type=Path, default=None,
                        help="cache of foundation-model forecasts written by tsfm_cache.py: appends to every LOB "
                             "window row [scaled change of q50/q10/q90 per step, sign, available] of the last "
                             "forecast made at most --tsfm-max-stale-min before the block; they enter the direct "
                             "path too.  Needs --lookback-hours")
    parser.add_argument("--hand-dir", type=Path, default=None,
                        help="directory written by hand_features.py: appends the standardised hand-made features of "
                             "the last tensor row (book shape, activity clock, trades, static fields; the no-state "
                             "control's inputs) to every LOB window row; they enter the encoder and the direct path")
    parser.add_argument("--tree-dir", type=Path, default=None,
                        help="directory written by tree_features.py: appends the cross-fitted LightGBM log-probabilities "
                             "of the last tensor row [lp_down, lp_const, lp_up, available] to every LOB window row "
                             "(they enter the encoder, the direct path and, with --extra-to-predictor, the predictor)")
    parser.add_argument("--news-dir", type=Path, default=None,
                        help="directory written by news_features.py (rows --tensor-days): appends the standardised GDELT "
                             "news-volume features of the last tensor row (+ available) to every LOB window row; "
                             "they enter the encoder, the direct path and, with --extra-to-predictor, the predictor")
    parser.add_argument("--tree-offset", action="store_true",
                        help="add the tree log-probabilities of --tree-dir to the output logits: the model starts from "
                             "the trees' solution and learns the residual (zeros where no tree row exists)")
    parser.add_argument("--extra-to-predictor", action="store_true",
                        help="the extra columns (hand / forecast / log-odds features) are also read by the predictor "
                             "MLP next to the states, so their nonlinear interactions do not depend on the state")
    parser.add_argument("--tsfm-max-stale-min", type=float, default=3.0,
                        help="a cached forecast older than this (minutes) at block time is not used")
    parser.add_argument("--global-state", action="store_true",
                        help="the predictor also reads the mean state of the markets active in the block")
    parser.add_argument("--lob-encoder", choices=["gru", "timefirst"], default="gru",
                        help="window encoder: 'gru' reads the K snapshots as K feature vectors; 'timefirst' reads "
                             "every LOB channel as a series of length K first, then mixes the channels (use with "
                             "a longer --window, e.g. 64)")
    parser.add_argument("--time-dim", type=int, default=32, help="per-channel token size of the timefirst encoder")
    parser.add_argument("--state-update", choices=["gru", "ctan"], default="gru",
                        help="'gru' = GRU cell + LayerNorm (original); 'ctan' = anti-symmetric update "
                             "h + eps*tanh((W - W^T - gamma*I) h + x) for market and wallet states")
    parser.add_argument("--ctan-eps", type=float, default=0.5)
    parser.add_argument("--ctan-gamma", type=float, default=0.1)
    parser.add_argument("--log-sizes", action="store_true",
                        help="apply sign(x)*log1p(|x|) to the heavy-tailed LOB columns (sizes, depths, OFI, "
                             "volume, relative ask prices) instead of letting the ±100 clamp saturate them")
    parser.add_argument("--lookback-hours", type=str, default="",
                        help="comma-separated lookbacks in hours (e.g. '1,6,12'): appends causal "
                             "[Δmid, sign, available] features per lookback to every LOB window row")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    horizon_ns = args.horizon * 10**9
    horizon_hours = int(math.ceil(args.horizon / 3600))
    horizon_label = f"{args.horizon // 3600}h" if args.horizon >= 3600 else f"{args.horizon}s"
    lookback_hours = [float(x) for x in args.lookback_hours.split(",") if x.strip()]
    lookbacks_ns = [int(h * HOUR_NS) for h in lookback_hours]
    n_logodds = 1 + len(lookbacks_ns) if args.lookback_logodds else 0
    if args.lookback_logodds and not lookbacks_ns:
        raise SystemExit("--lookback-logodds needs --lookback-hours")
    tsfm_index = None
    n_tsfm = 0
    if args.tsfm_dir is not None:
        if not lookbacks_ns:
            raise SystemExit("--tsfm-dir needs --lookback-hours")
        from polymarket_gnn.data.tsfm_index import TSFMIndex
        tsfm_index = TSFMIndex(args.tsfm_dir, max_stale_ns=int(args.tsfm_max_stale_min * 60 * 10**9))
        n_tsfm = tsfm_index.n_features
    hand_index = None
    n_hand = 0
    if args.hand_dir is not None:
        if not lookbacks_ns:
            raise SystemExit("--hand-dir needs --lookback-hours")
        from polymarket_gnn.data.hand_index import HandFeatureIndex
        hand_index = HandFeatureIndex(args.hand_dir)
        n_hand = hand_index.n_features
    tree_index = None
    n_tree = 0
    if args.tree_dir is not None:
        if not lookbacks_ns:
            raise SystemExit("--tree-dir needs --lookback-hours")
        from polymarket_gnn.data.tree_index import TreeIndex
        tree_index = TreeIndex(args.tree_dir)
        n_tree = tree_index.n_features
    if args.tree_offset and (tree_index is None or args.task == "regression"):
        raise SystemExit("--tree-offset needs --tree-dir and the classification task")
    news_index = None
    n_news = 0
    if args.news_dir is not None:
        if not lookbacks_ns:
            raise SystemExit("--news-dir needs --lookback-hours")
        from polymarket_gnn.data.news_index import NewsIndex
        news_index = NewsIndex(args.news_dir)
        n_news = news_index.n_features
    n_extra = n_hand + n_logodds + n_tsfm + n_tree + n_news          # columns between the LOB features and the lookback triples
    if args.extra_to_predictor and not n_extra:
        raise SystemExit("--extra-to-predictor needs extra columns (--hand-dir, --tsfm-dir or --lookback-logodds)")
    lob_dim = OUTPUT_DIM + n_extra + 3 * len(lookbacks_ns)
    regression = args.task == "regression"
    default_label = (horizon_label + ("_lb" if lookbacks_ns else "") + ("_log" if args.log_sizes else "")
                     + ("_tf" if args.lob_encoder == "timefirst" else "") + (f"_w{args.window}" if args.window != 10 else "")
                     + ({"learned": "", "none": "_dnone", "multiscale": "_dms"}[args.decay_mode])
                     + ("_hm" if args.memory_periods.strip() else "")
                     + ("_ctan" if args.state_update == "ctan" else "")
                     + ("_skip" if args.skip_lookback else "") + ("_mr" if args.skip_init == "meanrev" else "")
                     + ("_lo" if args.lookback_logodds else "") + ("_tsfm" if tsfm_index is not None else "")
                     + ("_hand" if hand_index is not None else "") + ("_xp" if args.extra_to_predictor else "")
                     + ("_tree" if tree_index is not None else "") + ("_tro" if args.tree_offset else "")
                     + ("_news" if news_index is not None else "")
                     + (f"_hd{args.head_dim}" if args.head_dim else "") + (f"_hl{args.head_layers}" if args.head_layers != 1 else "")
                     + (f"_ep{args.epochs}" if args.epochs != 1 else "")
                     + ("_glob" if args.global_state else "")
                     + ("_ob" if args.edge_orientation == "book" else "")
                     + ({"none": "", "real": "_sk", "placebo": "_skp"}[args.edge_skill])
                     + ("_beh" if args.edge_behaviour else "")
                     + ("_reg" if regression else "")
                     + ("_aw" if args.focal_alpha.replace(" ", "") != "2,1,2" and not regression else "")
                     + ("_us" if args.undersample_ratio > 0 else "")
                     + ("_ablation" if args.ablation else ""))
    if args.seed != 0:
        default_label += f"_s{args.seed}"
    run_label = args.run_label or default_label
    days_or_none = lambda spec: parse_days(spec) if str(spec).strip() else []        # parse_days rejects ''
    train_days, val_days, warm_days = days_or_none(args.train_days), days_or_none(args.val_days), days_or_none(args.warm_days)
    eval_only = args.init_from is not None
    if eval_only:
        if train_days:
            raise SystemExit("--init-from evaluates a checkpoint: give --train-days '' (and --warm-days to rebuild the states)")
        if not args.run_label:
            raise SystemExit("--init-from needs an explicit --run-label (the files of the original run must not be overwritten)")
        if regression:
            raise SystemExit("--init-from is not supported for --task regression (tau is not restored)")
        if args.ema > 0:
            raise SystemExit("--init-from: give --ema 0 (an EMA checkpoint already holds the averaged weights)")
        if args.epochs != 1:
            raise SystemExit("--init-from: --epochs has no meaning without training days")
        if not args.init_from.is_file():
            raise SystemExit(f"{args.init_from} not found")
    elif not train_days:
        raise SystemExit("--train-days is empty (evaluation of a checkpoint needs --init-from)")
    if not val_days:
        raise SystemExit("--val-days is empty")
    if train_days and max(train_days) >= min(warm_days or val_days):
        raise SystemExit("--train-days must strictly precede --warm-days and --val-days")
    if warm_days and max(warm_days) >= min(val_days):
        raise SystemExit("--warm-days must strictly precede --val-days")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"device={device} | horizon={horizon_label} | ablation={args.ablation} | run_label={run_label} | "
                f"train={train_days} warm={warm_days} val={val_days}" + (f" | init_from={args.init_from}" if eval_only else ""))

    with open(args.out_dir / f"run_config_{run_label}.json", "w") as f:
        json.dump({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "started_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    # ── 1. Mappings ──────────────────────────────────────────────────────────
    market_mapping = pd.read_parquet(args.base_dir / "id_mappings/market_to_idx.parquet")
    wallet_mapping = pd.read_parquet(args.base_dir / "id_mappings/wallet_to_idx.parquet")
    market_to_idx = dict(zip(market_mapping["market_id"], market_mapping["market_idx"]))
    wallet_to_idx = dict(zip(wallet_mapping["wallet"], wallet_mapping["wallet_idx"]))
    n_markets, n_wallets = len(market_to_idx), len(wallet_to_idx)
    logger.info(f"markets={n_markets} wallets={n_wallets}")

    # ── 2. Node state ────────────────────────────────────────────────────────
    # The initial market state is a FIXED random projection of the static
    # market features (390 dims).  It is not trained: the state is detached
    # every TBPTT segment, so gradients could only ever reach the projection
    # through the first segment.  It only sets a deterministic starting point.
    hidden = args.hidden_dim
    market_init = torch.load(args.base_dir / "market_features.pt", map_location="cpu")
    projector = nn.Linear(market_init.shape[1], hidden)
    with torch.no_grad():
        h_market = projector(market_init).to(device)
        h_market = torch.clamp(torch.nan_to_num(h_market, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
    h_wallet = torch.randn(n_wallets, hidden, device=device) * 0.01
    del projector, market_init

    # ── 3. Model ─────────────────────────────────────────────────────────────
    from polymarket_gnn.models.causal_gnn import CausalDynamicGNN
    use_skill = args.edge_skill != "none"
    if use_skill and (args.edge_orientation != "book" or args.ablation):
        raise SystemExit("--edge-skill needs --edge-orientation book and the graph (no --ablation)")
    skill_root = args.skill_dir / f"h{horizon_label}"
    if use_skill and not skill_root.is_dir():
        raise SystemExit(f"{skill_root} not found: run build_wallet_skill.py --horizon {args.horizon}")
    if args.edge_behaviour and args.ablation:
        raise SystemExit("--edge-behaviour needs the graph (no --ablation)")
    if args.edge_behaviour and not args.behaviour_dir.is_dir():
        raise SystemExit(f"{args.behaviour_dir} not found: run build_wallet_skill.py --behaviour")
    edge_dim = EDGE_DIM + (N_SKILL_FEATURES if use_skill else 0) + (N_BEHAVIOUR_FEATURES if args.edge_behaviour else 0)
    memory_periods = [float(x) for x in args.memory_periods.split(",") if x.strip()]
    if (memory_periods or args.skip_lookback or args.global_state) and args.state_engine != "sparse":
        raise SystemExit("--memory-periods, --skip-lookback and --global-state need --state-engine sparse")
    if args.skip_lookback and not lookbacks_ns:
        raise SystemExit("--skip-lookback needs --lookback-hours")
    market_hl = [float(x) for x in args.decay_half_lives.split(",") if x.strip()]
    wallet_hl = [float(x) for x in args.wallet_half_lives.split(",") if x.strip()] or None
    model = CausalDynamicGNN(lob_dim=lob_dim, hidden_dim=hidden, edge_dim=edge_dim,
                             out_dim=1 if regression else N_CLASSES, decay_mode=args.decay_mode,
                             market_half_lives=market_hl if args.decay_mode == "multiscale" else None,
                             wallet_half_lives=wallet_hl if args.decay_mode == "multiscale" else None,
                             memory_periods=memory_periods or None,
                             skip_dim=3 * len(lookbacks_ns) if args.skip_lookback else 0, skip_extra=n_extra,
                             skip_init_lookback=(int(np.argmin([abs(lb - horizon_ns) for lb in lookbacks_ns]))
                                                 if args.skip_lookback and args.skip_init == "meanrev" else None),
                             predictor_extra=n_extra if args.extra_to_predictor else 0,
                             global_state=args.global_state, state_update=args.state_update,
                             ctan_eps=args.ctan_eps, ctan_gamma=args.ctan_gamma,
                             lob_encoder=args.lob_encoder, window=args.window, n_series=OUTPUT_DIM,
                             time_dim=args.time_dim, head_dim=args.head_dim or None, head_layers=args.head_layers).to(device)
    if regression:
        # Start from "no change": with a random head the initial predictions are O(1) while the
        # targets are mostly 0, and the first thousands of blocks are spent learning the bias.
        nn.init.zeros_(model.predictor.net[-1].weight)
        nn.init.zeros_(model.predictor.net[-1].bias)
    if eval_only:
        state = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded weights from {args.init_from}: {sum(v.numel() for v in state.values()):,} parameters in "
                    f"{len(state)} tensors (strict)")
        del state
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = WeightEMA(model, args.ema) if args.ema > 0 else None
    auto_alpha = args.focal_alpha.strip().lower() == "auto"
    criterion = FocalLoss(alpha=parse_alpha(args.focal_alpha, device), gamma=args.focal_gamma, reduction="none")
    # imbalance handling state (training only, causal: counts are updated after use)
    train_class_counts = torch.zeros(N_CLASSES, dtype=torch.float32, device=device)
    sampler_gen = torch.Generator(device=device)
    sampler_gen.manual_seed(args.seed)
    # regression: threshold calibration on the last training day(s), diagnostic sweep on validation
    tau = args.init_tau
    calibration = {"calibrated": False, "tau": tau}
    calib_day_set = set(train_days[-args.calib_days:]) if regression else set()
    calibrator = ThresholdCalibrator(default_thresholds(), device) if regression else None
    val_sweep = ThresholdCalibrator(default_thresholds(), device) if regression else None
    orientation = None
    if args.edge_orientation == "book":
        orientation = load_market_orientation(args.base_dir.parent / "universe_markets.parquet")
        known = [orientation[m] for m in market_to_idx if m in orientation]
        logger.info(f"edge orientation 'book': {len(known)} of {n_markets} markets in the universe file, "
                    f"{sum(o == -1 for o in known)} with the book on token 2 (missing markets keep the raw direction)")
    edge_builder = EdgeBuilder(market_to_idx, wallet_to_idx, orientation, skill=use_skill, behaviour=args.edge_behaviour)

    # ── 4. Data structures ───────────────────────────────────────────────────
    lifecycle = LifecycleTracker(args.base_dir / "market_lifecycle_events.parquet")
    hours_back = int(math.ceil(max(lookback_hours))) if lookback_hours else 0
    if args.save_val_predictions:
        hours_back = max(hours_back, horizon_hours)      # needed for past_delta = mid(t) - mid(t - H)
    price_index = MidPriceIndex(args.base_dir, hours_back=hours_back, hours_ahead=horizon_hours,
                                lifecycle=lifecycle, close_filter=args.close_filter, const_eps=args.const_eps)
    schedule = build_schedule(train_days, val_days, args.max_hours, warm_days)
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.epochs > 1:
        if args.state_engine != "sparse":
            raise SystemExit("--epochs > 1 needs --state-engine sparse")
        schedule = [s for s in schedule if s[4]] * args.epochs + [s for s in schedule if not s[4]]
    first = pd.Timestamp(year=schedule[0][0], month=schedule[0][1], day=schedule[0][2], hour=schedule[0][3])
    buffer_cls = FastLOBWindowBuffer if args.lob_buffer == "fast" else LOBWindowBuffer
    buffer = buffer_cls.from_parquet(hour_paths_for_buffer(args.base_dir, first), K=args.window)
    buffer.reset_to_time(first.value)

    # ── 5. Loop state ────────────────────────────────────────────────────────
    block_count = total_lob_updates = total_edges = 0
    tbptt_loss = torch.tensor(0.0, device=device)
    loss_history, val_history = [], []
    prev_block_ts: Optional[int] = None
    val_dedup = DedupTracker(int(args.dedup_minutes * 60 * 10**9))
    empty_edges = (torch.empty((2, 0), dtype=torch.long, device=device),
                   torch.empty((0, edge_dim), dtype=torch.float32, device=device))
    sparse = args.state_engine == "sparse"
    epoch = 0
    if sparse:
        clock = DecayClock(device)
        state_m, state_w = make_lazy_states(model, h_market, h_wallet, clock)
        memory = HierarchicalMemory(model, n_markets, clock, device, seed=args.seed) if memory_periods else None
        if args.epochs > 1:
            h_market0, h_wallet0 = h_market.cpu(), h_wallet.cpu()      # initial states, for the resets between epochs
        del h_market, h_wallet

    val_pred_rows = []          # per-block tuples of arrays, only with --save-val-predictions

    def save_val_predictions():
        if not (args.save_val_predictions and val_pred_rows):
            return
        cols = ["t", "market_idx", "y_true", "y_pred", "delta", "past_delta", "mid", "dedup", "p_down", "p_const", "p_up"]
        dtypes = [np.int64, np.int32, np.int8, np.int8, np.float32, np.float32, np.float32, bool, np.float32, np.float32, np.float32]
        data = {c: np.concatenate([r[i] for r in val_pred_rows]).astype(dt) for i, (c, dt) in enumerate(zip(cols, dtypes))}
        out = args.out_dir / f"val_predictions_{run_label}.parquet"
        pd.DataFrame(data).to_parquet(out, index=False)
        logger.info(f"Saved {len(data['t'])} validation predictions to {out}")

    def save_histories():
        if regression:
            with open(args.out_dir / f"calibration_{run_label}.json", "w") as f:
                json.dump({**calibration,
                           "note": "tau chosen on training data only; 'val_sweep' is a diagnostic and must not be used to select tau",
                           "val_sweep": [{k: v for k, v in r.items() if k != "cm"} for r in val_sweep.table()] if val_sweep.n else []},
                          f, indent=2)
        with open(args.out_dir / f"loss_history_{run_label}.json", "w") as f:
            json.dump(loss_history, f)
        with open(args.out_dir / f"val_history_{run_label}.json", "w") as f:
            json.dump(val_history, f)

    for year, month, day, hour, is_train, record in schedule:
        ts_hour = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        model.train(is_train)
        if is_train and ts_hour == first and block_count > 0:
            # ── new pass over the training hours: everything that carries time is reset, the weights stay ──
            epoch += 1
            clock = DecayClock(device)
            state_m, state_w = make_lazy_states(model, h_market0.to(device), h_wallet0.to(device), clock)
            memory = HierarchicalMemory(model, n_markets, clock, device, seed=args.seed + epoch) if memory_periods else None
            buffer.reload_lob_data(hour_paths_for_buffer(args.base_dir, first))
            buffer.reset_to_time(first.value)
            prev_block_ts = None
            tbptt_loss = torch.tensor(0.0, device=device)
            optimizer.zero_grad()
            logger.info(f"=== epoch {epoch + 1}/{args.epochs}: states, memory, clock and buffer reset; optimizer kept ===")
        if not is_train:
            tbptt_loss = torch.tensor(0.0, device=device)
            if ema is not None and not ema.applied:
                logger.info(f"EMA weights swapped in for validation: {ema.apply(model)}")
            if regression and not calibration["calibrated"]:
                best = calibrator.best("balanced_accuracy")
                if best is None:
                    logger.warning(f"No calibration samples: keeping tau={tau}")
                else:
                    tau = best["tau"]
                    calibration.update({"tau": tau, "n_calibration_samples": calibrator.n,
                                        "calibration_days": sorted(calib_day_set),
                                        "train_balanced_accuracy_at_tau": best["balanced_accuracy"],
                                        "train_sweep": [{k: v for k, v in r.items() if k != "cm"} for r in calibrator.table()]})
                    logger.info(f"Calibrated tau={tau:.6f} on {calibrator.n} training samples "
                                f"(days {sorted(calib_day_set)}), train bal_acc={best['balanced_accuracy']:.4f}")
                calibration["calibrated"] = True
        logger.info(f"--- {'TRAINING' if is_train else ('VALIDATION' if record else 'WARM-UP')} hour {ts_hour} ---")

        t0 = time.time()
        extra_cols, extra_joins = "", ""
        if args.edge_behaviour:
            beh_hour = args.behaviour_dir / f"day={day}" / f"hour={hour}"
            if not beh_hour.is_dir():
                raise SystemExit(f"no wallet behaviour for {ts_hour} in {beh_hour}")
            extra_cols += "".join(f", b.{c}" for c in BEHAVIOUR_COLUMNS)
            extra_joins += (f"LEFT JOIN read_parquet('{beh_hour}/*.parquet') b "
                            f"ON t.wallet = b.wallet AND t.timestamp = b.timestamp ")
        if use_skill:
            skill_hour = skill_root / f"day={day}" / f"hour={hour}"
            if not skill_hour.is_dir():
                raise SystemExit(f"no wallet skill for {ts_hour} in {skill_hour}")
            pre = "p_" if args.edge_skill == "placebo" else ""
            extra_cols += f", s.{pre}wins AS skill_wins, s.{pre}losses AS skill_losses"
            extra_joins += (f"LEFT JOIN read_parquet('{skill_hour}/*.parquet') s "
                            f"ON t.wallet = s.wallet AND t.role = s.role AND t.timestamp = s.timestamp ")
        try:
            txn_df = duckdb.query(
                f"WITH t AS (SELECT * FROM read_parquet('{txn_path(args.base_dir, year, month, day, hour)}/**/*.parquet')) "
                f"SELECT t.*{extra_cols} FROM t {extra_joins}"
                # All rows of a block share one timestamp, so "ORDER BY block_number, timestamp" leaves
                # the order inside a block to the thread scheduling of the query and it changes from
                # process to process (verified).  The model is invariant to edge order mathematically,
                # but dropout masks and floating-point sums are not: the tie-breakers make runs reproducible.
                f"ORDER BY t.block_number, t.timestamp, t.tx_hash, t.order_index, t.wallet, t.role, t.token_id, t.side, t.tokens, t.price"
            ).df()
        except Exception as e:
            logger.warning(f"No transactions for {ts_hour}: {e}")
            continue
        if txn_df.empty:
            continue
        logger.info(f"Loaded {len(txn_df)} transactions in {time.time() - t0:.2f}s")
        if args.edge_behaviour:
            if txn_df["beh_n"].isna().any():
                raise SystemExit(f"{int(txn_df['beh_n'].isna().sum())} transactions of {ts_hour} without a behaviour row")
            logger.info(f"wallet behaviour: {(txn_df['beh_n'] > 0).mean():.3f} of the transactions come from a wallet seen before")
        if use_skill:
            if txn_df["skill_wins"].isna().any():
                raise SystemExit(f"{int(txn_df['skill_wins'].isna().sum())} transactions of {ts_hour} without a skill row")
            logger.info(f"wallet skill ({args.edge_skill}): {((txn_df['skill_wins'] + txn_df['skill_losses']) > 0).mean():.3f} "
                        f"of the transactions come from a wallet with a track record")

        if ts_hour != first:
            buffer.reload_lob_data(hour_paths_for_buffer(args.base_dir, ts_hour))
        price_index.set_hour(year, month, day, hour)
        if tsfm_index is not None:
            tsfm_index.set_hour(year, month, day, hour)
        if hand_index is not None:
            hand_index.set_hour(year, month, day, hour)
        if tree_index is not None:
            tree_index.set_hour(year, month, day, hour)
        if news_index is not None:
            news_index.set_hour(year, month, day, hour)
        if prev_block_ts is None:
            prev_block_ts = ts_hour.value

        blocks_this_hour = 0
        for block_number, block_txns in txn_df.groupby("block_number", sort=True):
            t_ns = int(block_txns["timestamp"].iloc[0].value)
            if t_ns <= prev_block_ts:
                prev_block_ts = max(prev_block_ts, t_ns)
                continue
            if args.max_blocks_per_hour and blocks_this_hour >= args.max_blocks_per_hour:
                break
            blocks_this_hour += 1

            # ── A. LOB windows for active markets ─────────────────────────────
            t1 = time.time()
            active_ids, windows = buffer.load_step(prev_block_ts, t_ns)
            tree_offset_np = None
            if active_ids:
                # Canonical order (ascending market index).  The reference buffer iterates a Python
                # set, whose order changes from process to process; a fixed order makes runs
                # reproducible and the two buffers / two state engines directly comparable.
                keep = sorted((i for i, m in enumerate(active_ids) if m in market_to_idx),
                              key=lambda i: market_to_idx[active_ids[i]])
                active_ids = [active_ids[i] for i in keep]
                windows = windows[keep]
            if active_ids and args.log_sizes:
                windows = signed_log1p_columns(windows)
            if active_ids and lookbacks_ns:
                ctx = price_index.lookback_features(active_ids, t_ns, lookbacks_ns)        # (N, 3L), causal
                if n_logodds:
                    ctx = np.concatenate([price_index.logodds_features(active_ids, t_ns, lookbacks_ns), ctx], axis=1)
                if tsfm_index is not None:
                    mid_now = np.array([price_index.mid_before(m, t_ns) for m in active_ids], dtype=np.float64)
                    ctx = np.concatenate([tsfm_index.features(active_ids, t_ns, mid_now), ctx], axis=1)
                if hand_index is not None:
                    ctx = np.concatenate([hand_index.features(active_ids, t_ns), ctx], axis=1)
                if news_index is not None:
                    ctx = np.concatenate([news_index.features(active_ids, t_ns), ctx], axis=1)
                if tree_index is not None:
                    ctx = np.concatenate([tree_index.features(active_ids, t_ns), ctx], axis=1)
                    if args.tree_offset:
                        tree_offset_np = tree_index.offset(active_ids, t_ns)
                windows = np.concatenate([windows, np.repeat(ctx[:, None, :], args.window, axis=1)], axis=2)
            if active_ids:
                active_idx = torch.tensor([market_to_idx[m] for m in active_ids], dtype=torch.long, device=device)
                windows_gpu = torch.tensor(windows, dtype=torch.float32, device=device)
                windows_gpu = torch.clamp(torch.nan_to_num(windows_gpu, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
                total_lob_updates += len(active_idx)
            else:
                active_idx = torch.empty(0, dtype=torch.long, device=device)
                windows_gpu = torch.empty((0, args.window, lob_dim), dtype=torch.float32, device=device)
            lob_time = time.time() - t1
            if tree_offset_np is not None and len(tree_offset_np) != len(active_ids):
                tree_offset_np = None

            # ── B. Transaction edges ──────────────────────────────────────────
            t1 = time.time()
            if args.ablation:
                edge_index, edge_features = empty_edges
            else:
                edge_index, edge_features = edge_builder(block_txns, device=device)
                edge_features = torch.clamp(torch.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
            edge_time = time.time() - t1

            # ── C. Targets ────────────────────────────────────────────────────
            delta_np = price_index.future_change(active_ids, t_ns, horizon_ns) if active_ids else np.empty(0, dtype=np.float64)
            valid_np = ~np.isnan(delta_np)
            valid_mask = torch.from_numpy(valid_np).to(device)
            valid_active_idx = active_idx[valid_mask]
            valid_targets = torch.from_numpy(classes_from_delta(delta_np[valid_np], args.const_eps)).to(device)
            valid_delta = torch.from_numpy(delta_np[valid_np].astype(np.float32)).to(device)

            delta_t = torch.tensor((t_ns - prev_block_ts) / 1e9, dtype=torch.float32, device=device)
            if not sparse:
                h_market_in, h_wallet_in = h_market.clone(), h_wallet.clone()

            with torch.set_grad_enabled(is_train):
                if sparse:
                    logits = sparse_step(model, clock, state_m, state_w, windows_gpu, active_idx,
                                         edge_index, edge_features, delta_t.item(), valid_active_idx, memory)
                elif is_train:
                    h_market_in.requires_grad_(True)   # checkpoint needs ≥1 input with grad
                    logits, h_market_out, h_wallet_out = checkpoint(
                        model, windows_gpu, active_idx, h_market_in, h_wallet_in,
                        edge_index, edge_features, delta_t, delta_t, valid_active_idx,
                        use_reentrant=False,
                    )
                else:
                    logits, h_market_out, h_wallet_out = model(
                        windows_gpu, active_idx, h_market_in, h_wallet_in,
                        edge_index, edge_features, delta_t, delta_t, valid_active_idx,
                    )
            if torch.isnan(logits).any():
                raise ValueError(f"NaN logits at block {block_count}")
            if tree_offset_np is not None and len(valid_active_idx) > 0:
                logits = logits + torch.from_numpy(tree_offset_np[valid_np]).to(device)    # residual over the trees
            if not sparse:
                h_market, h_wallet = h_market_out, h_wallet_out

            step_loss = 0.0
            if len(valid_active_idx) > 0:
                if regression:
                    pred_scaled = logits.squeeze(1)
                    per_sample = F.smooth_l1_loss(pred_scaled, valid_delta * args.delta_scale,
                                                  beta=args.huber_beta, reduction="none")
                    pred_delta = pred_scaled.detach() / args.delta_scale
                    preds = classify_delta(pred_delta, tau)
                    if is_train and day in calib_day_set:
                        calibrator.update(pred_delta, valid_targets)
                    elif not is_train:
                        val_sweep.update(pred_delta, valid_targets)
                else:
                    if is_train and auto_alpha:
                        criterion.alpha = inverse_frequency_weights(train_class_counts)
                    per_sample = criterion(logits, valid_targets)
                    preds = torch.argmax(logits, dim=1)
                loss = per_sample.mean()          # reported loss: always over all labelled samples
                if is_train:
                    if args.undersample_ratio > 0:
                        keep_loss = undersample_mask(valid_targets, train_class_counts, args.undersample_ratio, sampler_gen)
                        train_loss = per_sample[keep_loss].mean() if keep_loss.any() else per_sample.sum() * 0.0
                    else:
                        train_loss = loss
                    tbptt_loss = tbptt_loss + train_loss
                    train_class_counts += torch.bincount(valid_targets, minlength=N_CLASSES).to(torch.float32)
                step_loss = loss.item()
                cm = confusion_from_tensors(valid_targets, preds)
                entry = {"block": block_count, "t": t_ns, "epoch": epoch, "loss": step_loss,
                         "accuracy": sum(cm[i][i] for i in range(N_CLASSES)) / len(valid_targets), "cm": cm}
                if regression:
                    entry["mae"] = (pred_delta - valid_delta).abs().mean().item()
                if is_train:
                    loss_history.append(entry)
                elif record:
                    valid_ids = [m for m, v in zip(active_ids, valid_np) if v]
                    keep = val_dedup.keep_mask(valid_ids, t_ns)
                    if args.save_val_predictions:
                        probs = (torch.softmax(logits.detach().float(), dim=1).cpu().numpy() if not regression
                                 else np.full((len(valid_ids), 3), np.nan, dtype=np.float32))
                        val_pred_rows.append((
                            np.full(len(valid_ids), t_ns, dtype=np.int64),
                            valid_active_idx.cpu().numpy(),
                            valid_targets.cpu().numpy(),
                            preds.cpu().numpy(),
                            delta_np[valid_np],
                            price_index.past_change(valid_ids, t_ns, horizon_ns),      # causal, NaN if unobservable
                            np.array([price_index.mid_before(m, t_ns) for m in valid_ids], dtype=np.float32),
                            keep,
                            probs[:, 0], probs[:, 1], probs[:, 2],
                        ))
                    if keep.any():
                        keep_t = torch.from_numpy(keep).to(device)
                        entry["cm_dedup"] = confusion_from_tensors(valid_targets[keep_t], preds[keep_t])
                    val_history.append(entry)

            # ── TBPTT ─────────────────────────────────────────────────────────
            if block_count > 0 and block_count % args.tbptt == 0:
                if is_train:
                    if tbptt_loss.requires_grad and not torch.isnan(tbptt_loss):
                        (tbptt_loss / args.accum).backward()
                    if block_count % (args.tbptt * args.accum) == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        if ema is not None:
                            ema.update(model)
                    tbptt_loss = torch.tensor(0.0, device=device)
                if sparse:
                    state_m.commit(); state_w.commit(); clock.commit()
                    if memory is not None:
                        memory.commit()
                else:
                    h_market = torch.nan_to_num(h_market.detach(), nan=0.0)
                    h_wallet = torch.nan_to_num(h_wallet.detach(), nan=0.0)

            prev_block_ts = t_ns
            block_count += 1
            total_edges += edge_index.size(1)
            if block_count % 100 == 0:
                logger.info(
                    f"Block {block_count:6d} | blk={block_number} | t={pd.Timestamp(t_ns)} | "
                    f"LOB={len(active_idx):3d} labelled={len(valid_active_idx):3d} | edges={edge_index.size(1):4d} | "
                    f"loss={step_loss:.4f} | lob_t={lob_time * 1000:.1f}ms edge_t={edge_time * 1000:.1f}ms"
                )

        logger.info(f"Finished hour {ts_hour} | blocks={block_count} | LOB_updates={total_lob_updates} | edges={total_edges}")
        if hour == 23 and is_train:
            ckpt = args.out_dir / f"model_{run_label}_day_{day:02d}.pt"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Saved daily checkpoint: {ckpt}")
            save_histories()

    if not eval_only:
        torch.save(model.state_dict(), args.out_dir / f"model_{run_label}_final.pt")
    save_histories()
    save_val_predictions()
    logger.info(f"Results saved to {args.out_dir / f'val_history_{run_label}.json'}")


if __name__ == "__main__":
    main()
