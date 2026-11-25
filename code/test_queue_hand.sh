#!/bin/bash
# TEST queue A (cuda:2): the hand-feature models without trees; warm-up 21-23, test 24-26
cd /data/v.pampinella/agy_project_eval
./run_dev.sh eval_checkpoint.py --device cuda:2 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --set hand_dir=/data/v.pampinella/agy_project/data/derived/hand_features_test   --labels 900s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0            1h_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 1h_lb_hm_sg_lb5_tf64_hand_xp_ablation_pred_s0            1h_lb_hm_sg_lb5_tf64_hand_xp_ema0.998_ablation_pred_s0
