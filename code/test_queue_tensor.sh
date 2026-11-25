#!/bin/bash
# TEST queue B (cuda:0): the tensor-only models, after the warm-up length check; warm-up 21-23, test 24-26
cd /data/v.pampinella/agy_project_eval
while ! grep -q "ALL DONE" recheck_queue3.log; do sleep 120; done
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs   --labels 6h_lb_hm_sg_lb5_tf64_ablation_pred_s0 6h_lb_hm_sg_lb5_ablation_pred_s0 12h_lb_hm_sg_lb5_ablation_pred_s0            1h_lb_hm_sg_lb5_ablation_pred_s0 1h_lb_hm_sg_lb5_ablation_pred_s1 1h_lb_hm_sg_lb5_ablation_pred_s2            1h_lb_hm_sg_lb5_tf64_ablation_pred_s0 1h_lb_hm_sg_lb5_tf64_ablation_pred_s1 1h_lb_hm_sg_lb5_ob_pred_s0            6h_lb_hm_sg_lb5_ob_pred_s0 12h_lb_hm_sg_lb5_ablation_pred_s1 12h_lb_hm_sg_lb5_ablation_pred_s2 12h_lb_hm_sg_lb5_ob_pred_s0
