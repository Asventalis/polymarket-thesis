#!/usr/bin/env bash
# queue_tests_new.sh DEVICE LABEL[:TREE_DIR] ...  (25 Nov 2026)
# Test evaluation (warm-up 21-23, test 24-26) of the runs launched on 24-25 Nov, in order: each waits for
# model_<label>_final.pt and, for the residual runs, for TREE_DIR/meta.json; idempotent (skips if the
# predictions already exist).  Configs and checkpoints are read from this directory (--src-dir).
D=$1; shift
cd /data/v.pampinella/agy_project_eval
T=/data/v.pampinella/agy_project/data/derived
for spec in "$@"; do
  L=${spec%%:*}; TD=${spec#*:}; [ "$TD" = "$L" ] && TD=""
  while [ ! -f "model_${L}_final.pt" ] || { [ -n "$TD" ] && [ ! -f "$TD/meta.json" ]; }; do sleep 300; done
  if [ -f "test_runs/val_predictions_${L}_test.parquet" ]; then echo ">>> $L: test predictions exist"; continue; fi
  SET="--set hand_dir=$T/hand_features_test"; [ -n "$TD" ] && SET="$SET --set tree_dir=$TD"
  echo ">>> eval $L on $D ($(date))"
  ./run_dev.sh eval_checkpoint.py --src-dir /data/v.pampinella/agy_project_eval --device "$D" --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs $SET --labels "$L"
  echo ">>> eval $L exit $? ($(date))"
done
echo ">>> QUEUE DONE ($(date))"
