#!/bin/bash
# after the second recheck queue: does a 7-day warm-up (6-12, the whole training stream without learning)
# reproduce the original validation predictions better than 2 days for the runs with the lowest agreement?
cd /data/v.pampinella/agy_project_eval
while ! grep -q 'ALL DONE' recheck_queue2.log; do sleep 120; done
./run_dev.sh eval_checkpoint.py --device cuda:0 --warm-days 6-12 --val-days 13 --suffix _recheck_w7 --out-dir recheck \
    --labels 12h_lb_hm_sg_lb5_ablation_pred_s0 6h_lb_hm_sg_lb5_ob_pred_s0 1h_lb_hm_sg_lb5_tf64_ablation_pred_s1
