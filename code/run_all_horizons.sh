#!/bin/bash
# run_all_horizons.sh
# Runs 3 training runs + 3 baseline evaluations sequentially.
# Each run uses a different prediction horizon (1h, 6h, 12h).
# Markets that close before the horizon are automatically excluded.
#
# Usage: ./run.sh bash run_all_horizons.sh

set -e  # Exit immediately on any error

LOGDIR="/data/v.pampinella/agy_project"
cd "$LOGDIR"

echo ""
echo "========================================================"
echo " HORIZON STUDY: Starting 3-horizon experiment"
echo " $(date)"
echo "========================================================"

# ─── 1-HOUR HORIZON ──────────────────────────────────────────
echo ""
echo ">>> [1/6] Training CT-DGNN — 1h horizon  ($(date))"
python train_full.py --horizon 3600
echo ">>> [1/6] DONE"

echo ""
echo ">>> [2/6] Baselines — 1h horizon  ($(date))"
python baseline_evaluation.py --horizon 3600
echo ">>> [2/6] DONE"

# ─── 6-HOUR HORIZON ──────────────────────────────────────────
echo ""
echo ">>> [3/6] Training CT-DGNN — 6h horizon  ($(date))"
python train_full.py --horizon 21600
echo ">>> [3/6] DONE"

echo ""
echo ">>> [4/6] Baselines — 6h horizon  ($(date))"
python baseline_evaluation.py --horizon 21600
echo ">>> [4/6] DONE"

# ─── 12-HOUR HORIZON ─────────────────────────────────────────
echo ""
echo ">>> [5/6] Training CT-DGNN — 12h horizon  ($(date))"
python train_full.py --horizon 43200
echo ">>> [5/6] DONE"

echo ""
echo ">>> [6/6] Baselines — 12h horizon  ($(date))"
python baseline_evaluation.py --horizon 43200
echo ">>> [6/6] DONE"

# ─── FINAL AGGREGATED REPORT ─────────────────────────────────
echo ""
echo ">>> Generating final comparison report..."
python parse_all_horizons.py

echo ""
echo "========================================================"
echo " ALL DONE — $(date)"
echo "========================================================"
