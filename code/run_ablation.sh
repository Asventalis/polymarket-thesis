#!/bin/bash
# run_ablation.sh
# Runs 3 ablation (LOB-only) training runs sequentially,
# then prints the combined comparison table.
#
# The baseline results from the previous run are reused (no need to re-run them).

set -e
cd /data/v.pampinella/agy_project

echo ""
echo "========================================================"
echo " ABLATION STUDY: LOB-only model (no graph) — 3 horizons"
echo " $(date)"
echo "========================================================"

echo ""
echo ">>> [1/3] Ablation Training — 1h horizon  ($(date))"
python train_full.py --horizon 3600 --ablation
echo ">>> [1/3] DONE"

echo ""
echo ">>> [2/3] Ablation Training — 6h horizon  ($(date))"
python train_full.py --horizon 21600 --ablation
echo ">>> [2/3] DONE"

echo ""
echo ">>> [3/3] Ablation Training — 12h horizon  ($(date))"
python train_full.py --horizon 43200 --ablation
echo ">>> [3/3] DONE"

echo ""
echo ">>> Generating final ablation comparison report..."
python parse_all_horizons.py

echo ""
echo "========================================================"
echo " ABLATION DONE — $(date)"
echo "========================================================"
