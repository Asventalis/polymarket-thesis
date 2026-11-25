#!/bin/bash
# seed1_1m.sh — second seed (s1) of the two one-minute models on cuda:1: the two-pass run and the two-pass residual of the
# cross-fitted trees, each trained as its seed-0 twin (run_hand_runs_v4.sh, 6-12 March, validation 13 March) and then
# evaluated on the locked test days (warm-up 21-23, test 24-26); a 1 min test report with both seeds after each.
set -u
cd /data/v.pampinella/agy_project_eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
OR=/data/v.pampinella/agy_project_orient
DA=/data/v.pampinella/data_audit
DER=/data/v.pampinella/agy_project/data/derived
TN=w512_d3_do0.15_lr0.001_b8192_bins16
L0=60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0
L1=60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s1
R0=60s_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0
R1=60s_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s1
RUN="nice -n 10 ./run_dev.sh scripts/data_audit/test_report.py"
report () {  # $1 = out name, rest = deep labels
  local out=$1; shift
  echo ">>> report $out start $(date)"
  $RUN --horizon 1m --deep-horizon 60s --val-audit-dir $DA/deep_1m --test-audit-dir $DA/test_1m \
    --val-pred-dirs $OR: --test-pred-dir test_runs --test-suffix _test --deep "$@" \
    --tab "TabM 3 seeds=$DA/deep_1m/tabular_net/proba_1m_tabm_${TN}_mean.npy:$DA/test_1m/tabular_net/proba_1m_tabm_${TN}_mean.npy" \
    --title "Test set 24-26 March (1m), two seeds" --out reports/$out.md > reports/$out.log 2>&1
  echo ">>> report $out exit $? $(date)"
}
echo ">>> seed 1 two-pass training start $(date)"
SEED=1 EPOCHS=2 $OR/run_hand_runs_v4.sh cuda:1 60 > $OR/hand_1m_ep2_s1.log 2>&1 < /dev/null
echo ">>> seed 1 two-pass training exit $? $(date)"
if [ -f $OR/model_${L1}_final.pt ]; then
  echo ">>> seed 1 two-pass test eval start $(date)"
  ./run_dev.sh eval_checkpoint.py --device cuda:1 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs \
    --set hand_dir=$DER/hand_features_test --labels $L1 > test_eval_1m_s1.log 2>&1
  echo ">>> seed 1 two-pass test eval exit $? $(date)"
  report test_1m_s1 $L0 $L1 $R0 &
fi
echo ">>> seed 1 residual training start $(date)"
SEED=1 EPOCHS=2 TREE_DIR=$DER/tree_logprobs_1m_ctrl $OR/run_hand_runs_v4.sh cuda:1 60 > $OR/hand_1m_tree_ep2_s1.log 2>&1 < /dev/null
echo ">>> seed 1 residual training exit $? $(date)"
wait
if [ -f $OR/model_${R1}_final.pt ]; then
  echo ">>> seed 1 residual test eval start $(date)"
  ./run_dev.sh eval_checkpoint.py --device cuda:1 --warm-days 21-23 --val-days 24-26 --suffix _test --out-dir test_runs \
    --set hand_dir=$DER/hand_features_test --set tree_dir=$DER/tree_logprobs_1m_test --labels $R1 > test_eval_1m_tree_s1.log 2>&1
  echo ">>> seed 1 residual test eval exit $? $(date)"
  report test_1m_seeds $L0 $L1 $R0 $R1
fi
echo ">>> SEED1 1M DONE $(date)"
