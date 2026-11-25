#!/bin/bash
# TEST queue B2 (cuda:0): the two Chronos-feature runs, after queue B and after the Chronos cache for 21-26
cd /data/v.pampinella/agy_project_eval
while ! grep -q "ALL DONE" test_queue_tensor.log || [ ! -f /data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h_test/day=2026-03-26.parquet ] || pgrep -f tsfm_cache.py >/dev/null; do sleep 300; done
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set tsfm_dir=/data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h_test   --labels 1h_lb_hm_sg_lb5_tsfm_ablation_pred_s0 1h_lb_hm_sg_lb5_tsfm_ablation_pred_s1
echo ">>> TSFM DONE"
