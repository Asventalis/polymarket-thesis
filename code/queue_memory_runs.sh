#!/bin/bash
# queue_memory_runs.sh — waits for the chains currently running in agy_project_next
# (run_analysis_runs.sh) to finish, then installs the code of this staging copy there and
# starts the memory study, one chain per GPU.  To cancel before it starts:
#   pkill -f "[q]ueue_memory_runs.sh"
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/data/v.pampinella/agy_project_next
echo "waiting for running chains... ($(date))"
until [ "$(pgrep -u "$USER" -f '[r]un_analysis_runs.sh' | wc -l)" -eq 0 ] && [ "$(pgrep -u "$USER" -f '[t]rain_full.py' | wc -l)" -eq 0 ]; do sleep 60; done
echo "chains finished ($(date)); installing code from $STAGE into $WORK"
rsync -a --exclude '*.json' --exclude '*.pt' --exclude '*.log' --exclude '*.parquet' --exclude '__pycache__' \
      --exclude '.pytest_cache' --exclude 'analysis' "$STAGE/" "$WORK/"
cd "$WORK"
nohup ./run_memory_runs.sh full cuda:1 > memory_full.log 2>&1 < /dev/null &
nohup ./run_memory_runs.sh lob  cuda:3 > memory_lob.log  2>&1 < /dev/null &
echo "memory study started ($(date))"
