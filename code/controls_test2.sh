#!/bin/bash
# controls on the 1h and 1m test samples, after controls_test.sh and after the 1 min test run
cd /data/v.pampinella/agy_project_eval
export FA_ROW_DAYS=6-13,21-26 FA_VAL_DAYS=24-26 FA_PRED_DIR=/data/v.pampinella/agy_project_eval/test_runs FA_THREADS=24
run_ctrl() {
  [ -f /data/v.pampinella/data_audit/test_$2/proba_$2_G5.npy ] && { echo ">>> control $2 exists"; return; }
  echo ">>> control $2 start $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/feature_audit.py $1 $3 $3 10 /data/v.pampinella/data_audit/test_$2 > controls_test_$2.log 2>&1
  echo ">>> feature_audit $2 exit $? $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/control_proba.py --horizon $2 --audit-dir /data/v.pampinella/data_audit/test_$2 --threads 24 >> controls_test_$2.log 2>&1
  echo ">>> control_proba $2 exit $? $(date)"
}
while ! grep -q "CONTROLS DONE" controls_test.log; do sleep 300; done
run_ctrl 3600 1h lb_hm_sg_lb5_ablation_pred_s0_test
while [ ! -f test_runs/val_predictions_60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0_test.parquet ]; do sleep 600; done
run_ctrl 60 1m lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0_test
echo ">>> CONTROLS2 DONE"
