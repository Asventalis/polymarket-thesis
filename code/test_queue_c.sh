#!/bin/bash
# TEST queue C (cuda:0): two of the 1h hand-feature models (queue A on cuda:2 reaches them later and skips the ones
# already evaluated), then the two Chronos-feature runs once the cache for 21-26 is complete
cd /data/v.pampinella/agy_project_eval
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set hand_dir=/data/v.pampinella/agy_project/data/derived/hand_features_test   --labels 1h_lb_hm_sg_lb5_tf64_hand_xp_ema0.998_ablation_pred_s0 1h_lb_hm_sg_lb5_tf64_hand_xp_ablation_pred_s0
while [ ! -f /data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h_test/day=2026-03-26.parquet ] || pgrep -f "tsfm_cache.py --horizon" >/dev/null; do sleep 300; done
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set tsfm_dir=/data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h_test   --labels 1h_lb_hm_sg_lb5_tsfm_ablation_pred_s0 1h_lb_hm_sg_lb5_tsfm_ablation_pred_s1
echo ">>> QUEUE C DONE"
