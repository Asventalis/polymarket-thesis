#!/bin/bash
# TEST queue A2 (cuda:2): the two residual models, after queue A and after the trees for 21-26
cd /data/v.pampinella/agy_project_eval
while ! grep -q "ALL DONE" test_queue_hand.log || [ ! -f /data/v.pampinella/agy_project/data/derived/tree_logprobs_1h_test/meta.json ] || [ ! -f /data/v.pampinella/agy_project/data/derived/tree_logprobs_1m_test/meta.json ]; do sleep 300; done
./run_dev.sh eval_checkpoint.py --device cuda:2 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set hand_dir=/data/v.pampinella/agy_project/data/derived/hand_features_test --set tree_dir=/data/v.pampinella/agy_project/data/derived/tree_logprobs_1h_test   --labels 1h_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0
./run_dev.sh eval_checkpoint.py --device cuda:2 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set hand_dir=/data/v.pampinella/agy_project/data/derived/hand_features_test --set tree_dir=/data/v.pampinella/agy_project/data/derived/tree_logprobs_1m_test   --labels 60s_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0
echo ">>> RESIDUALS DONE"
