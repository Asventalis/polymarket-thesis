#!/usr/bin/env bash
# queue_residual2.sh DEVICE H TREE_DIR MIN_FREE_MIB [WAIT_FILE ...]
# Like queue_residual.sh, plus: waits until the GPU has at least MIN_FREE_MIB free (the GPUs are shared).
# Launched 24 Nov 2026 for the 6 h and 12 h residual runs.
set -u
DEVICE="$1"; H="$2"; TREE="$3"; MINFREE="$4"; shift 4
IDX="${DEVICE#cuda:}"
cd "$(dirname "${BASH_SOURCE[0]}")"
ready() {
  [ -f "$TREE/meta.json" ] || return 1
  for f in "$@"; do [ -f "$f" ] || return 1; done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$IDX" | tr -d " ")
  [ "${free:-0}" -ge "$MINFREE" ]
}
while ! ready "$@"; do sleep 300; done
echo ">>> queue: prerequisites ready at $(date) (GPU $IDX free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $IDX)); launching residual run H=$H on $DEVICE"
EPOCHS=2 TREE_DIR="$TREE" TREE_OFFSET=1 ./run_hand_runs_v4.sh "$DEVICE" "$H"
