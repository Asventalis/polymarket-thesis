#!/usr/bin/env bash
# Full-length run of the best LOB-only configuration (five lookbacks, zero-init direct path, global
# state, hierarchical memory, time-first encoder K=64) WITH the hand-made features of hand_features.py
# in the window rows, the direct path and (XP=1, default) the predictor.
#   ./run_hand_runs.sh cuda:N H        env SEED (0), XP (1), ENC (timefirst), W (64), HAND_DIR,
#                                      LOSS ("" = focal 2/1/2 γ=2; "bce" = class-balanced CE: --focal-alpha auto --focal-gamma 0),
#                                      HEAD ("" = default head; "256x2" = --head-dim 256 --head-layers 2),
#                                      EPOCHS (1; N = N passes over the training days, --epochs N)
# e.g.  nohup ./run_hand_runs.sh cuda:3 3600 > hand_1h.log 2>&1 < /dev/null &
# Label: <H>h_lb_hm_sg_lb5_<tf|gru><W>_hand[_xp][_bce][_hd<D>_hl<L>][_ep<N>]_ablation_pred_s<SEED>
set -u
DEVICE="${1:-cuda:3}"; H="${2:-3600}"; SEED="${SEED:-0}"; XP="${XP:-1}"; ENC="${ENC:-timefirst}"; W="${W:-64}"
LOSS="${LOSS:-}"; HEAD="${HEAD:-}"; EPOCHS="${EPOCHS:-1}"
HAND_DIR="${HAND_DIR:-/data/v.pampinella/agy_project/data/derived/hand_features}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HL="$((H / 3600))h"
TAG=$([ "$ENC" = "timefirst" ] && echo "tf" || echo "gru")
EXTRA=""; SUFFIX=""
if [ "$LOSS" = "bce" ]; then EXTRA="$EXTRA --focal-alpha auto --focal-gamma 0"; SUFFIX="${SUFFIX}_bce"; fi
if [ -n "$HEAD" ]; then D="${HEAD%x*}"; L="${HEAD#*x}"; EXTRA="$EXTRA --head-dim $D --head-layers $L"; SUFFIX="${SUFFIX}_hd${D}_hl${L}"; fi
if [ "$EPOCHS" != "1" ]; then EXTRA="$EXTRA --epochs $EPOCHS"; SUFFIX="${SUFFIX}_ep${EPOCHS}"; fi
LABEL="${HL}_lb_hm_sg_lb5_${TAG}${W}_hand$([ "$XP" = "1" ] && echo "_xp")${SUFFIX}_ablation_pred_s${SEED}"
if [ -f "model_${LABEL}_final.pt" ]; then echo ">>> $LABEL already done"; else
  echo ">>> $LABEL on $DEVICE  ($(date))"
  ./run_dev.sh train_full.py --ablation --device "$DEVICE" --seed "$SEED" --horizon "$H" --lookback-hours 0.25,1,3,6,12 \
      --skip-lookback --global-state --memory-periods 60,3600 --lob-encoder "$ENC" --window "$W" \
      --hand-dir "$HAND_DIR" $([ "$XP" = "1" ] && echo "--extra-to-predictor") $EXTRA \
      --save-val-predictions --run-label "$LABEL"
  echo ">>> $LABEL DONE  ($(date))"
fi
echo ">>> [$LABEL] ALL DONE ($(date))"
