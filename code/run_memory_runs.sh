#!/bin/bash
# run_memory_runs.sh — node-state memory study against the 'learned' scalar decay of the
# reference runs <H>_lb_pred_s0 / <H>_lb_ablation_pred_s0.  Seed 0, lookback features,
# reference loss, per-sample validation predictions saved.  One chain per GPU.
#
# Variants (run in the order hm, dnone, dnone_hm, dms; each for every horizon before moving to the next):
#   dnone     no decay, memory managed by the GRU gates alone
#   hm        hierarchical memory: per-block state + 1-minute state + 1-hour state, each level with
#             its own GRU cell (minute reads the per-block state, hour reads the minute state),
#             predictor on the three states; per-block state with the reference learned decay
#   dnone_hm  hierarchical memory on top of a per-block state without decay
#   dms       one state, groups of dimensions with half-lives 1 min / 1 h / 1 day / none (shared weights)
#
#   nohup ./run_memory_runs.sh full cuda:1 > memory_full.log 2>&1 < /dev/null &
#   nohup ./run_memory_runs.sh lob  cuda:3 > memory_lob.log  2>&1 < /dev/null &
#
# The LOB-only chain is needed to measure the graph contribution (full − LOB-only) under each memory mode.
set -e
CHAIN="$1"; DEVICE="$2"; HORIZONS="${3:-3600 43200}"; VARIANTS="${4:-hm dnone dnone_hm dms}"
if [ -z "$CHAIN" ] || [ -z "$DEVICE" ]; then echo "usage: $0 full|lob cuda:N [\"horizons\"] [\"variants\"]"; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
FLAG=""; SUF=""; if [ "$CHAIN" = "lob" ]; then FLAG="--ablation"; SUF="_ablation"; fi
for V in $VARIANTS; do
  case "$V" in
    dnone)    OPTS="--decay-mode none" ;;
    hm)       OPTS="--memory-periods 60,3600" ;;
    dnone_hm) OPTS="--decay-mode none --memory-periods 60,3600" ;;
    dms)      OPTS="--decay-mode multiscale --decay-half-lives 60,3600,86400,inf" ;;
    *) echo "unknown variant $V"; exit 1 ;;
  esac
  for H in $HORIZONS; do
    LABEL="$((H / 3600))h_lb_${V}${SUF}_pred_s0"
    if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done, skipping"; continue; fi
    echo ">>> $LABEL on $DEVICE  ($(date))"
    ./run_dev.sh train_full.py $FLAG --device "$DEVICE" --seed 0 --horizon $H --lookback-hours 1,6,12 \
        $OPTS --save-val-predictions --run-label "$LABEL"
    echo ">>> $LABEL DONE  ($(date))"
  done
done
echo ">>> [$CHAIN] ALL DONE ($(date))"
