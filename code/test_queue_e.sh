#!/bin/bash
# TEST queue E (cuda:1): the zero-shot Chronos-2 runs that queue D could not start (wrong path), then the
# tabular networks at 1 min that went out of memory on cuda:3.
cd /data/v.pampinella/agy_project_eval
export PYTHONPATH=/data/v.pampinella/tsfm_pkgs HF_HOME=/data/v.pampinella/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
zs() {  # H seconds, step-min, run label (with the horizon prefix, without the _test suffix)
  [ -f tsfm_zero_shot_test/report_$1_$3_test.md ] && { echo ">>> zero-shot $1 exists"; return; }
  echo ">>> zero-shot $1 $3 start $(date)"
  nice -n 10 ./run_dev.sh tsfm_zero_shot.py --horizon $1 --run-label $3_test --step-min $2 --pred-dir test_runs --device cuda:1 --threads 16 --out-dir /data/v.pampinella/agy_project_eval/tsfm_zero_shot_test > zero_shot_test_$1.log 2>&1
  echo ">>> zero-shot $1 exit $? $(date)"
}
zs 3600 1 1h_lb_hm_sg_lb5_ablation_pred_s0
zs 43200 5 12h_lb_hm_sg_lb5_ablation_pred_s0
zs 900 1 900s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
zs 21600 5 6h_lb_hm_sg_lb5_tf64_ablation_pred_s0
zs 60 1 60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
if [ -f /data/v.pampinella/data_audit/test_1m/tabular_net/proba_1m_tabm_w512_d3_do0.15_lr0.001_b8192_bins16_mean.npy ]; then echo ">>> tabm 1m exists"; else
  echo ">>> tabm 1m start $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/tabular_net.py --audit-dir /data/v.pampinella/data_audit/test_1m --horizon 1m --device cuda:1 --threads 16 > tabm_test_1m.log 2>&1
  echo ">>> tabm 1m exit $? $(date)"
fi
echo ">>> QUEUE E DONE"
