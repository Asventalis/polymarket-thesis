#!/bin/bash
# run_lookback.sh — multi-seed study with long-lookback features (Δmid at 1h/6h/12h).
# One chain per GPU; seeds are the OUTER loop so that a complete 3-horizon
# comparison for seed 0 is available first.
#
#   nohup ./run_lookback.sh full cuda:3 "0 1 2"     > lookback_full.log     2>&1 < /dev/null &
#   nohup ./run_lookback.sh ablation cuda:1 "0 1 2" > lookback_ablation.log 2>&1 < /dev/null &
#
# Labels: <H>_lb_s<seed> and <H>_lb_ablation_s<seed>  → read by compare_runs.py --tag lb
set -e
CHAIN="$1"; DEVICE="$2"; SEEDS="${3:-0 1 2}"; LOOKBACK="${LOOKBACK:-1,6,12}"
if [ -z "$CHAIN" ] || [ -z "$DEVICE" ]; then echo "usage: $0 full|ablation cuda:N \"seeds\""; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
FLAG=""; SUF=""; if [ "$CHAIN" = "ablation" ]; then FLAG="--ablation"; SUF="_ablation"; fi
for SEED in $SEEDS; do
  for H in 3600 21600 43200; do
    LABEL="$((H / 3600))h_lb${SUF}_s${SEED}"
    if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> [$CHAIN] $LABEL already done, skipping"; continue; fi
    echo ">>> [$CHAIN] $LABEL on $DEVICE  ($(date))"
    ./run_dev.sh train_full.py --horizon $H $FLAG --device "$DEVICE" --seed "$SEED" \
        --lookback-hours "$LOOKBACK" --run-label "$LABEL"
    echo ">>> [$CHAIN] $LABEL DONE  ($(date))"
  done
done
echo ">>> [$CHAIN] ALL DONE ($(date))"
