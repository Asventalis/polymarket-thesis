#!/usr/bin/env bash
# queue_residual.sh DEVICE H TREE_DIR [WAIT_FILE ...]
# Waits until TREE_DIR/meta.json (the end of a tree_features.py build) and every WAIT_FILE exist,
# then runs the two-pass residual run of run_hand_runs_v4.sh (EPOCHS=2, TREE_DIR, --tree-offset).
# Launched 24 Nov 2026 for the horizons the thesis lacks (15 min, 6 h, 12 h).
set -u
DEVICE="$1"; H="$2"; TREE="$3"; shift 3
cd "$(dirname "${BASH_SOURCE[0]}")"
while [ ! -f "$TREE/meta.json" ] || { for f in "$@"; do [ -f "$f" ] || exit_wait=1; done; [ "${exit_wait:-0}" = 1 ]; }; do
  exit_wait=0; sleep 300
done
echo ">>> queue: prerequisites ready at $(date); launching residual run H=$H on $DEVICE"
EPOCHS=2 TREE_DIR="$TREE" TREE_OFFSET=1 ./run_hand_runs_v4.sh "$DEVICE" "$H"
