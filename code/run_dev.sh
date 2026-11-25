#!/bin/bash
# run_dev.sh — run a command inside the PyTorch Apptainer image using the MAIN
# project's virtualenv but THIS directory's sources.  Lets a development copy
# (e.g. /data/v.pampinella/agy_project_dev) be tested without touching the
# code that a running experiment in /data/v.pampinella/agy_project will import.
#
#   ./run_dev.sh -m pytest tests -q
#   ./run_dev.sh baseline_evaluation.py --horizon 3600
#   ./run_dev.sh train_full.py --horizon 3600 --max-hours 2 --device cuda:1
DEV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="${MAIN_DIR:-/data/v.pampinella/agy_project}"
cd "$DEV_DIR"
exec apptainer exec --nv --bind /data/v.pampinella "$MAIN_DIR/pytorch.sif" \
    env PYTHONPATH="$DEV_DIR/src" "$MAIN_DIR/.venv/bin/python" "$@"
