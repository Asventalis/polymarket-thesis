#!/bin/bash
# run_supervisor_variants.sh — the supervisor's suggestions (24/09/2026) as one chain of
# single-change experiments on top of the lookback configuration (LOB-only model, seed 0).
# Reference run for all of them: <H>_lb_ablation_s0 (focal alpha 2,1,2, no undersampling).
#
#   nohup ./run_supervisor_variants.sh cuda:1 "3600"            > variants.log 2>&1 < /dev/null &
#   nohup ./run_supervisor_variants.sh cuda:1 "3600 21600 43200" > variants.log 2>&1 < /dev/null &
#
# Variants (label suffix):
#   aw    class weights = inverse running class frequency          (--focal-alpha auto)
#   us    unweighted focal loss + undersampling of frequent classes (--focal-alpha none --undersample-ratio 1.5)
#   reg   regression on the signed change, tau calibrated on the last training day
# Add --full as third argument to train the LOB + graph model instead of the LOB-only one.
set -e
DEVICE="$1"; HORIZONS="${2:-3600}"; MODE="${3:---ablation}"
if [ -z "$DEVICE" ]; then echo "usage: $0 cuda:N \"horizons_in_seconds\" [--full]"; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "$MODE" = "--full" ]; then FLAG=""; SUF=""; else FLAG="--ablation"; SUF="_ablation"; fi
run() {  # run <label> <extra args...>
  local LABEL="$1"; shift
  if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done, skipping"; return; fi
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py $FLAG --device "$DEVICE" --seed 0 --lookback-hours 1,6,12 --run-label "$LABEL" "$@"
  echo ">>> $LABEL DONE  ($(date))"
}
for H in $HORIZONS; do
  HL="$((H / 3600))h"
  run "${HL}_lb_aw${SUF}_s0"  --horizon $H --focal-alpha auto
  run "${HL}_lb_us${SUF}_s0"  --horizon $H --focal-alpha none --undersample-ratio 1.5
  run "${HL}_lb_reg${SUF}_s0" --horizon $H --task regression
done
echo ">>> ALL DONE ($(date))"
