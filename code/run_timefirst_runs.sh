#!/usr/bin/env bash
# Full-length run of the best LOB-only configuration (five lookbacks, zero-init direct path, global
# state, hierarchical memory) with a different LOB window encoder / window length.
#   ./run_timefirst_runs.sh cuda:N H ENCODER WINDOW     ENCODER gru|timefirst, WINDOW = snapshots per window
# e.g.  nohup ./run_timefirst_runs.sh cuda:3 3600 timefirst 64 > tf64_1h.log 2>&1 < /dev/null &
#       nohup ./run_timefirst_runs.sh cuda:1 3600 gru 32       > gru32_1h.log 2>&1 < /dev/null &
# Labels: <H>h_lb_hm_sg_lb5_<tf|gru><WINDOW>_ablation_pred_s<SEED>; env SEED (default 0), TIME_DIM (32).
set -u
DEVICE="${1:-cuda:3}"; H="${2:-3600}"; ENC="${3:-timefirst}"; W="${4:-64}"; SEED="${SEED:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HL="$((H / 3600))h"
TAG=$([ "$ENC" = "timefirst" ] && echo "tf" || echo "gru")
LABEL="${HL}_lb_hm_sg_lb5_${TAG}${W}_ablation_pred_s${SEED}"
if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done"; else
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py --ablation --device "$DEVICE" --seed "$SEED" --horizon "$H" --lookback-hours 0.25,1,3,6,12 \
      --skip-lookback --global-state --memory-periods 60,3600 --lob-encoder "$ENC" --window "$W" \
      --time-dim "${TIME_DIM:-32}" --save-val-predictions --run-label "$LABEL"
  echo ">>> $LABEL DONE  ($(date))"
fi
echo ">>> [$LABEL] ALL DONE ($(date))"
