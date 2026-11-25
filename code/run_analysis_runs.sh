#!/bin/bash
# run_analysis_runs.sh — runs that dump per-sample validation predictions
# (train_full.py --save-val-predictions) for the per-market breakdown, plus the
# first models at short horizons.  One chain per GPU, seed 0, lookback features on.
#
#   nohup ./run_analysis_runs.sh full cuda:1 > analysis_full.log 2>&1 < /dev/null &
#   nohup ./run_analysis_runs.sh lob  cuda:3 > analysis_lob.log  2>&1 < /dev/null &
#
# full : LOB + graph at 1h, 6h, 12h (reference loss), then 12h with undersampling (best 12h variant)
# lob  : LOB only at 1h, then 5 min / 1 min / 2 min, then 6h, 12h, then the short horizons with undersampling
# Short horizons use lookbacks of 1 min, 5 min and 1 h (label infix "slb").
set -e
CHAIN="$1"; DEVICE="$2"
if [ -z "$CHAIN" ] || [ -z "$DEVICE" ]; then echo "usage: $0 full|lob cuda:N"; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
LB="1,6,12"; SLB="0.0166667,0.0833333,1"
run() {  # run <label> <args...>
  local LABEL="$1"; shift
  if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done, skipping"; return; fi
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py --device "$DEVICE" --seed 0 --save-val-predictions --run-label "$LABEL" "$@"
  echo ">>> $LABEL DONE  ($(date))"
}
if [ "$CHAIN" = "full" ]; then
  for H in 3600 21600 43200; do run "$((H / 3600))h_lb_pred_s0" --horizon $H --lookback-hours $LB; done
  run "12h_lb_us_pred_s0" --horizon 43200 --lookback-hours $LB --focal-alpha none --undersample-ratio 1.5
else
  run "1h_lb_ablation_pred_s0" --horizon 3600 --lookback-hours $LB --ablation
  for H in 300 60 120; do run "${H}s_slb_ablation_pred_s0" --horizon $H --lookback-hours $SLB --ablation; done
  for H in 21600 43200; do run "$((H / 3600))h_lb_ablation_pred_s0" --horizon $H --lookback-hours $LB --ablation; done
  for H in 300 60 120; do run "${H}s_slb_us_ablation_pred_s0" --horizon $H --lookback-hours $SLB --ablation --focal-alpha none --undersample-ratio 1.5; done
fi
echo ">>> [$CHAIN] ALL DONE ($(date))"
