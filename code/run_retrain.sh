#!/bin/bash
# run_retrain.sh — re-train the 3-horizon study on the CORRECT target (mid price,
# tensor column 40) with the fixed pipeline.  Two independent chains, one per GPU:
#   chain "full"     : 1h → 6h → 12h, LOB + graph
#   chain "ablation" : 1h → 6h → 12h, LOB only
#
# Usage (from /data/v.pampinella/agy_project_dev, each ~14h per horizon):
#   nohup ./run_retrain.sh full cuda:3     > retrain_full.log     2>&1 &
#   nohup ./run_retrain.sh ablation cuda:1 > retrain_ablation.log 2>&1 &
set -e
CHAIN="$1"; DEVICE="$2"
if [ -z "$CHAIN" ] || [ -z "$DEVICE" ]; then
    echo "usage: $0 full|ablation cuda:N"; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")"
FLAG=""; [ "$CHAIN" = "ablation" ] && FLAG="--ablation"
for H in 3600 21600 43200; do
    echo ">>> [$CHAIN] horizon ${H}s on $DEVICE  ($(date))"
    ./run_dev.sh train_full.py --horizon $H $FLAG --device "$DEVICE"
    echo ">>> [$CHAIN] horizon ${H}s DONE  ($(date))"
done
echo ">>> [$CHAIN] ALL DONE ($(date))"
