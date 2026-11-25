#!/bin/bash
# run_tsfm_runs.sh — foundation-model step 2: cache zero-shot Chronos-2 forecasts for every active
# market on a 5-minute grid (train 6–12 March + validation 13 March), then train the best LOB-only
# configuration (five lookbacks, zero-init direct path, global state, hierarchical memory) with the
# forecast features appended to the windows and to the direct path.
#   ./run_tsfm_runs.sh cuda:N H cache DAYS   forecasts of the given days ('2026-03-06..2026-03-09')
#   ./run_tsfm_runs.sh cuda:N H train        the training run (the cache must be complete)
#   ./run_tsfm_runs.sh cuda:N H all          both, all days, one GPU
# Measured on a shared V100: ~600 forecasts/s in float32 (batch 1024, context 512), 2.6M active
# (market, 5-minute) pairs per day at 1h → ~1.2 h per day; the cache of 8 days is split over two
# GPUs (one job per GPU) and the training waits for both logs (queue_after.sh):
#   nohup ./run_tsfm_runs.sh cuda:1 3600 cache 2026-03-06..2026-03-09 > tsfm_cache_a.log 2>&1 < /dev/null &
#   nohup ./run_tsfm_runs.sh cuda:3 3600 cache 2026-03-10..2026-03-13 > tsfm_cache_b.log 2>&1 < /dev/null &
#   nohup ./queue_after.sh tsfm_cache_a.log ./queue_after.sh tsfm_cache_b.log \
#         ./run_tsfm_runs.sh cuda:3 3600 train > tsfm_1h.log 2>&1 < /dev/null &
set -e
DEVICE="${1:-cuda:3}"; H="${2:-3600}"; MODE="${3:-all}"; DAYS="${4:-2026-03-06..2026-03-13}"; SEED="${SEED:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=/data/v.pampinella/tsfm_pkgs HF_HOME=/data/v.pampinella/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# BATCH (default 1024) can be lowered on a GPU crowded by other users (batch 1024 needs ~4 GB)
HL="$((H / 3600))h"
CACHE="/data/v.pampinella/agy_project/data/derived/tsfm/chronos2_${HL}"
# context step 1 min at 1h (60 forecast steps), 5 min at 12h (144 steps); forecast grid 5 min in both cases
if [ "$H" = "3600" ]; then STEP=1; STEPS="15,30,60"; else STEP=5; STEPS="60,180,360,720"; fi
GRID=5; STALE=$((GRID + 3))
if [ "$MODE" = "cache" ] || [ "$MODE" = "all" ]; then
  echo ">>> cache $CACHE days $DAYS on $DEVICE  ($(date))"
  ./run_dev.sh tsfm_cache.py --horizon "$H" --step-min $STEP --grid-min $GRID --steps "$STEPS" --device "$DEVICE" \
      --days "$DAYS" --out-dir "$CACHE" --batch-size "${BATCH:-1024}"
fi
if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
  for D in 06 07 08 09 10 11 12 13; do
    [ -f "$CACHE/day=2026-03-$D.parquet" ] || { echo "!!! $CACHE/day=2026-03-$D.parquet missing"; exit 1; }
  done
  LABEL="${HL}_lb_hm_sg_lb5_tsfm_ablation_pred_s${SEED}"
  if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done"; else
    echo ">>> $LABEL on $DEVICE  ($(date))"
    ./run_dev.sh train_full.py --ablation --device "$DEVICE" --seed "$SEED" --horizon "$H" --lookback-hours 0.25,1,3,6,12 \
        --skip-lookback --global-state --memory-periods 60,3600 --tsfm-dir "$CACHE" --tsfm-max-stale-min $STALE \
        --save-val-predictions --run-label "$LABEL"
    echo ">>> $LABEL DONE  ($(date))"
  fi
fi
echo ">>> [tsfm $HL $MODE] ALL DONE ($(date))"
