#!/bin/bash
# run_best_runs.sh — full-length runs (train 6–12 March, validate on 13 March, seed 0) of the
# configuration that came out of the short tests of 12/10/2026: lookback features with the
# direct path to the output, global state, hierarchical market memory (60 s, 3600 s).
# One chain per GPU; the LOB-only chain measures the graph contribution.
#   nohup ./run_best_runs.sh full cuda:1 > best_full.log 2>&1 < /dev/null &
#   nohup ./run_best_runs.sh lob  cuda:3 > best_lob.log  2>&1 < /dev/null &
# Extra options for train_full.py (and a label suffix) can be given as 4th and 5th argument;
# the seed can be changed with the SEED environment variable.
set -e
CHAIN="$1"; DEVICE="$2"; HORIZONS="${3:-3600 43200}"; EXTRA="$4"; XSUF="$5"; SEED="${SEED:-0}"
if [ -z "$CHAIN" ] || [ -z "$DEVICE" ]; then echo "usage: $0 full|lob cuda:N [\"horizons\"] [\"extra options\" label_suffix]"; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "$CHAIN" = "lob" ]; then FLAG="--ablation"; SUF="_ablation"; else FLAG="--edge-orientation book"; SUF="_ob"; fi
for H in $HORIZONS; do
  LABEL="$((H / 3600))h_lb_hm_sg${XSUF}${SUF}_pred_s${SEED}"
  if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done, skipping"; continue; fi
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py $FLAG --device "$DEVICE" --seed "$SEED" --horizon $H --lookback-hours 1,6,12 \
      --skip-lookback --global-state --memory-periods 60,3600 $EXTRA --save-val-predictions --run-label "$LABEL"
  echo ">>> $LABEL DONE  ($(date))"
done
echo ">>> [$CHAIN] ALL DONE ($(date))"
