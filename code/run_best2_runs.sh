#!/bin/bash
# run_best2_runs.sh — second round of full-length LOB-only runs (train 6–12 March, validate on
# 13 March, seed 0), adding what helped in the short tests of 12/10/2026 to the configuration of
# run_best_runs.sh: direct path initialised as Mean Reversion, lookbacks 0.25/1/3/6/12 h.
# Two chains, one per GPU, with and without the hierarchical memory:
#   nohup ./run_best2_runs.sh hm   cuda:1 > best2_hm.log   2>&1 < /dev/null &
#   nohup ./run_best2_runs.sh nohm cuda:3 > best2_nohm.log 2>&1 < /dev/null &
set -e
CHAIN="$1"; DEVICE="$2"; HORIZONS="${3:-3600 43200}"
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$CHAIN" in
  hm)   OPTS="--memory-periods 60,3600"; TAG="hm_" ;;
  nohm) OPTS=""; TAG="" ;;
  *) echo "usage: $0 hm|nohm cuda:N [\"horizons\"]"; exit 1 ;;
esac
for H in $HORIZONS; do
  LABEL="$((H / 3600))h_lb5_${TAG}sg_mr_ablation_pred_s0"
  if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done, skipping"; continue; fi
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py --ablation --device "$DEVICE" --seed 0 --horizon $H --lookback-hours 0.25,1,3,6,12 \
      --skip-lookback --skip-init meanrev --global-state $OPTS --save-val-predictions --run-label "$LABEL"
  echo ">>> $LABEL DONE  ($(date))"
done
echo ">>> [$CHAIN] ALL DONE ($(date))"
