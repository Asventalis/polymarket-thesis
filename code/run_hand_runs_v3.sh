#!/usr/bin/env bash
# (v3 = v2 + TREE_DIR / TREE_OFFSET: the deep model as the residual of the cross-fitted LightGBM of tree_features.py)
# Full-length run of the best LOB-only configuration (five lookbacks, zero-init direct path, global
# state, hierarchical memory, time-first encoder K=64) WITH the hand-made features of hand_features.py
# in the window rows, the direct path and (XP=1, default) the predictor.
#   ./run_hand_runs.sh cuda:N H        env SEED (0), XP (1), ENC (timefirst), W (64), HAND_DIR,
#                                      LOSS ("" = focal 2/1/2 γ=2; "bce" = class-balanced CE: --focal-alpha auto --focal-gamma 0),
#                                      HEAD ("" = default head; "256x2" = --head-dim 256 --head-layers 2),
#                                      EPOCHS (1; N = N passes over the training days, --epochs N),
#                                      LR ("" = 1e-4; X = --lr X, suffix _lrX), ACCUM ("" = 20; N = --accum N, suffix _accN),
#                                      EMA ("" = off; D = --ema D, suffix _emaD: weight average along the pass used for validation),
#                                      TREE_DIR ("" = off; dir of tree_features.py → --tree-dir, suffix _tree), TREE_OFFSET (1 = --tree-offset, suffix _tro)
# e.g.  nohup ./run_hand_runs.sh cuda:3 3600 > hand_1h.log 2>&1 < /dev/null &
# Label: <H>h_lb_hm_sg_lb5_<tf|gru><W>_hand[_xp][_tree[_tro]][_bce][_hd<D>_hl<L>][_ep<N>][_lr<X>][_acc<N>][_ema<D>]_ablation_pred_s<SEED>
# (v3 of run_hand_runs.sh: a separate file because v2 was still being read by running jobs)
set -u
DEVICE="${1:-cuda:3}"; H="${2:-3600}"; SEED="${SEED:-0}"; XP="${XP:-1}"; ENC="${ENC:-timefirst}"; W="${W:-64}"
LOSS="${LOSS:-}"; HEAD="${HEAD:-}"; EPOCHS="${EPOCHS:-1}"; LR="${LR:-}"; ACCUM="${ACCUM:-}"; EMA="${EMA:-}"
TREE_DIR="${TREE_DIR:-}"; TREE_OFFSET="${TREE_OFFSET:-1}"
HAND_DIR="${HAND_DIR:-/data/v.pampinella/agy_project/data/derived/hand_features}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HL="$((H / 3600))h"
TAG=$([ "$ENC" = "timefirst" ] && echo "tf" || echo "gru")
EXTRA=""; SUFFIX=""
if [ -n "$TREE_DIR" ]; then EXTRA="$EXTRA --tree-dir $TREE_DIR"; SUFFIX="${SUFFIX}_tree"; if [ "$TREE_OFFSET" = "1" ]; then EXTRA="$EXTRA --tree-offset"; SUFFIX="${SUFFIX}_tro"; fi; fi
if [ "$LOSS" = "bce" ]; then EXTRA="$EXTRA --focal-alpha auto --focal-gamma 0"; SUFFIX="${SUFFIX}_bce"; fi
if [ -n "$HEAD" ]; then D="${HEAD%x*}"; L="${HEAD#*x}"; EXTRA="$EXTRA --head-dim $D --head-layers $L"; SUFFIX="${SUFFIX}_hd${D}_hl${L}"; fi
if [ "$EPOCHS" != "1" ]; then EXTRA="$EXTRA --epochs $EPOCHS"; SUFFIX="${SUFFIX}_ep${EPOCHS}"; fi
if [ -n "$LR" ]; then EXTRA="$EXTRA --lr $LR"; SUFFIX="${SUFFIX}_lr${LR}"; fi
if [ -n "$ACCUM" ]; then EXTRA="$EXTRA --accum $ACCUM"; SUFFIX="${SUFFIX}_acc${ACCUM}"; fi
if [ -n "$EMA" ]; then EXTRA="$EXTRA --ema $EMA"; SUFFIX="${SUFFIX}_ema${EMA}"; fi
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
