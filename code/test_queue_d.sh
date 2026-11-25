#!/bin/bash
# TEST queue D (cuda:3, after the Chronos cache): zero-shot Chronos-2 on the test samples of the deep runs, then the
# tabular networks on the test control tables (15m, 1h now; 1m when its control exists)
cd /data/v.pampinella/agy_project_eval
while [ ! -f /data/v.pampinella/agy_project/data/derived/tsfm/chronos2_1h_test/day=2026-03-26.parquet ] || pgrep -f "tsfm_cache.py --horizon" >/dev/null; do sleep 300; done
export PYTHONPATH=/data/v.pampinella/tsfm_pkgs HF_HOME=/data/v.pampinella/hf_cache
zs() {  # H seconds, step-min, run label (with the horizon prefix, without the _test suffix)
  [ -f tsfm_zero_shot_test/report_$1_$3_test.md ] && { echo ">>> zero-shot $1 exists"; return; }
  echo ">>> zero-shot $1 $3 start $(date)"
  (cd test_runs && nice -n 10 ../run_dev.sh ../tsfm_zero_shot.py --horizon $1 --run-label $3_test --step-min $2 --device cuda:3 --threads 16 --out-dir /data/v.pampinella/agy_project_eval/tsfm_zero_shot_test) > zero_shot_test_$1.log 2>&1
  echo ">>> zero-shot $1 exit $? $(date)"
}
zs 3600 1 1h_lb_hm_sg_lb5_ablation_pred_s0
zs 43200 5 12h_lb_hm_sg_lb5_ablation_pred_s0
zs 900 1 900s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
zs 21600 5 6h_lb_hm_sg_lb5_tf64_ablation_pred_s0
tn() {  # horizon label
  [ -d /data/v.pampinella/data_audit/test_$1/tabular_net ] && { echo ">>> tabm $1 exists"; return; }
  echo ">>> tabm $1 start $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/tabular_net.py --audit-dir /data/v.pampinella/data_audit/test_$1 --horizon $1 --device cuda:3 --threads 16 > tabm_test_$1.log 2>&1
  echo ">>> tabm $1 exit $? $(date)"
}
tn 15m
tn 1h
while [ ! -f /data/v.pampinella/data_audit/test_1m/proba_1m_G5.npy ]; do sleep 600; done
zs 60 1 60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
tn 1m
echo ">>> QUEUE D DONE"
