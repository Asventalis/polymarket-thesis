#!/bin/bash
# driver of the market-map sweep: requested weights first, then the extra variants, then the metrics
cd /data/v.pampinella/agy_project_dev
G=/data/v.pampinella/data_audit/graphviz
run() { ./run_dev.sh /data/v.pampinella/data_audit/graph_map_sweep.py --variant "$1" --jobs "$2" > "$G/nohup_sweep_$1.txt" 2>&1; echo "$(date +%H:%M:%S) finished $1 rc=$?" >> "$G/sweep_progress.txt"; }
: > "$G/sweep_progress.txt"
run knn64 48
run cat025 40 & run cat075 40 & run cat100 40 & wait
run sup050 36 & run nn50 36 & run tpl050 36 & wait
run sup000 48 & run day050 48 & wait
run metrics 16 --tags cat000,cat025,cat050,cat075,cat100,sup000,sup050,nn50,tpl050,day050
echo "$(date +%H:%M:%S) ALL DONE" >> "$G/sweep_progress.txt"
