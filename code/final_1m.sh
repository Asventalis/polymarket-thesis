#!/bin/bash
# TabM at 1 min on cuda:1 (MLP already done; the TabM variant went out of memory twice), the 1 min report with the
# residual meanwhile on CPU, then the final 1 min report with TabM.
cd /data/v.pampinella/agy_project_eval
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DA=/data/v.pampinella/data_audit
TN=w512_d3_do0.15_lr0.001_b8192_bins16
TABM=$DA/test_1m/tabular_net/proba_1m_tabm_${TN}_mean.npy
if [ ! -f $TABM ]; then
  echo ">>> tabm 1m start $(date)"
  nice -n 10 ./run_dev.sh scripts/data_audit/tabular_net.py --audit-dir $DA/test_1m --horizon 1m --device cuda:1 --threads 16 --variants tabm > tabm_test_1m_b.log 2>&1 &
  TPID=$!
fi
RUN="nice -n 10 ./run_dev.sh scripts/data_audit/test_report.py"
DEEP="60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 60s_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0"
echo ">>> 1m report (two-pass + residual) start $(date)"
$RUN --horizon 1m --deep-horizon 60s --val-audit-dir $DA/deep_1m --test-audit-dir $DA/test_1m \
  --val-pred-dirs /data/v.pampinella/agy_project_orient: --test-pred-dir test_runs --test-suffix _test \
  --deep $DEEP --title "Test set 24-26 March (1m)" --out reports/test_1m_v2.md > reports/test_1m_v2.log 2>&1
echo ">>> 1m report v2 exit $? $(date)"
if [ -n "$TPID" ]; then wait $TPID; echo ">>> tabm 1m exit $? $(date)"; fi
if [ -f $TABM ]; then
  echo ">>> 1m final report start $(date)"
  $RUN --horizon 1m --deep-horizon 60s --val-audit-dir $DA/deep_1m --test-audit-dir $DA/test_1m \
    --val-pred-dirs /data/v.pampinella/agy_project_orient: --test-pred-dir test_runs --test-suffix _test \
    --deep $DEEP --tab "TabM 3 seeds=$DA/deep_1m/tabular_net/proba_1m_tabm_${TN}_mean.npy:$TABM" \
    --title "Test set 24-26 March (1m)" --out reports/test_1m.md > reports/test_1m.log 2>&1
  echo ">>> 1m final report exit $? $(date)"
else
  echo ">>> tabm 1m still missing, final report not built"
fi
echo ">>> FINAL 1M DONE"
