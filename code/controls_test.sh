#!/bin/bash
# stateless controls on the test samples of the deep runs already evaluated (CPU, 24 threads, nice), then the baselines
cd /data/v.pampinella/agy_project_eval
export FA_ROW_DAYS=6-13,21-26 FA_VAL_DAYS=24-26 FA_PRED_DIR=/data/v.pampinella/agy_project_eval/test_runs FA_THREADS=24
run_ctrl() {  # H seconds, horizon label, deep label (without horizon prefix)
  [ -f /data/v.pampinella/data_audit/test_$2/proba_$2_G5.npy ] && { echo ">>> control $2 exists"; return; }
  echo ">>> control $2 start $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/feature_audit.py $1 $3 $3 10 /data/v.pampinella/data_audit/test_$2 > controls_test_$2.log 2>&1
  echo ">>> feature_audit $2 exit $? $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/control_proba.py --horizon $2 --audit-dir /data/v.pampinella/data_audit/test_$2 --threads 24 >> controls_test_$2.log 2>&1
  echo ">>> control_proba $2 exit $? $(date)"
}
run_ctrl 900 15m lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0_test
run_ctrl 21600 6h lb_hm_sg_lb5_tf64_ablation_pred_s0_test
run_ctrl 43200 12h lb_hm_sg_lb5_ablation_pred_s0_test
for H in 60 900 3600 21600 43200; do
  [ -f baseline_results_${H}_test.json ] && continue
  echo ">>> baselines $H start $(date)"
  nice -n 10 ./run_dev.sh baseline_evaluation.py --horizon $H --val-days 24-26 --prior-days 12 --out baseline_results_${H}_test.json > baselines_test_$H.log 2>&1
  echo ">>> baselines $H exit $? $(date)"
done
echo ">>> CONTROLS DONE"
