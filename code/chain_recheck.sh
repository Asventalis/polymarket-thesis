#!/bin/bash
# waits for the running recheck queue, then re-evaluates on 13 March the old runs without probabilities
cd /data/v.pampinella/agy_project_eval
while ! grep -q 'ALL DONE' recheck_queue.log; do sleep 120; done
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 11-12 --val-days 13 --suffix _recheck --out-dir recheck --labels $(cat recheck_labels.txt)
