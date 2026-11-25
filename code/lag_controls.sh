#!/bin/bash
# lag_controls.sh — does the gap between train/validation (6–13 March) and test (24–26) cost anything?
# The stateless control (LightGBM G5, feature_audit.py) refitted on the test samples of the deep runs with
#   all   : rows 6–26, fit 6–20, early stopping 21   (every day before the warm-up: no gap)
#   recent: rows 13–26, fit 14–20, early stopping 21 (a week, like the thesis control, but adjacent to the test)
# against the thesis control (fit 7–11, early stopping 12; data_audit/test_<H>/). Same samples, same hyper-parameters.
# Launched 24 Nov 2026 (supervisor review, lag question). CPU only, nice, 24 threads; sequential.
cd /data/v.pampinella/agy_project_eval
export FA_VAL_DAYS=24-26 FA_PRED_DIR=/data/v.pampinella/agy_project_eval/test_runs FA_THREADS=24
run_lag() {  # H seconds, horizon label, deep label, variant, row days, train days, es day
  OUT=/data/v.pampinella/data_audit/lag_$2_$4
  [ -f $OUT/proba_$2_G5.npy ] && { echo ">>> $2 $4 exists"; return; }
  echo ">>> $2 $4 start $(date)  rows $5 fit $6 es $7"
  FA_ROW_DAYS=$5 FA_TRAIN_DAYS=$6 FA_ES_DAY=$7 nice -n 10 ./run_dev.sh scripts/data_audit/feature_audit.py $1 $3 $3 10 $OUT > lag_$2_$4.log 2>&1
  echo ">>> feature_audit $2 $4 exit $? $(date)"
  FA_ROW_DAYS=$5 FA_TRAIN_DAYS=$6 FA_ES_DAY=$7 nice -n 10 ./run_dev.sh scripts/data_audit/control_proba.py --horizon $2 --audit-dir $OUT --threads 24 >> lag_$2_$4.log 2>&1
  echo ">>> control_proba $2 $4 exit $? $(date)"
}
for spec in "3600 1h lb_hm_sg_lb5_ablation_pred_s0_test" "43200 12h lb_hm_sg_lb5_ablation_pred_s0_test" "21600 6h lb_hm_sg_lb5_ablation_pred_s0_test" "900 15m lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0_test"; do
  set -- $spec
  run_lag $1 $2 $3 all    6-26  6-20  21
  run_lag $1 $2 $3 recent 13-26 14-20 21
done
echo ">>> LAG CONTROLS DONE $(date)"
