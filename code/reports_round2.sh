#!/bin/bash
# Round 2 of the test reports (CPU, nice): 15 min with TabM, 1 min partial (two-pass model),
# then the full 1 h report once the residual run has written its predictions.
cd /data/v.pampinella/agy_project_eval
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
DA=/data/v.pampinella/data_audit
TN=w512_d3_do0.15_lr0.001_b8192_bins16
RUN="nice -n 10 ./run_dev.sh scripts/data_audit/test_report.py"
echo ">>> 15m report start $(date)"
$RUN --horizon 15m --deep-horizon 900s --val-audit-dir $DA/deep_15m --test-audit-dir $DA/test_15m \
  --val-pred-dirs /data/v.pampinella/agy_project_orient: --test-pred-dir test_runs --test-suffix _test \
  --deep 900s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 \
  --tab "TabM 3 seeds=$DA/deep_15m/tabular_net/proba_15m_tabm_${TN}_mean.npy:$DA/test_15m/tabular_net/proba_15m_tabm_${TN}_mean.npy" \
  --title "Test set 24-26 March (15m)" --out reports/test_15m.md > reports/test_15m.log 2>&1
echo ">>> 15m report exit $? $(date)"
echo ">>> 1m partial report start $(date)"
$RUN --horizon 1m --deep-horizon 60s --val-audit-dir $DA/deep_1m --test-audit-dir $DA/test_1m \
  --val-pred-dirs /data/v.pampinella/agy_project_orient: --test-pred-dir test_runs --test-suffix _test \
  --deep 60s_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 \
  --title "Test set 24-26 March (1m)" --out reports/test_1m_partial.md > reports/test_1m_partial.log 2>&1
echo ">>> 1m partial report exit $? $(date)"
RES=1h_lb_hm_sg_lb5_tf64_hand_xp_tree_tro_ep2_ablation_pred_s0
while [ ! -s test_runs/val_predictions_${RES}_test.parquet ] || pgrep -f "run-label ${RES}_test" > /dev/null; do sleep 300; done
echo ">>> 1h full report start $(date)"
$RUN --horizon 1h --val-audit-dir $DA --test-audit-dir $DA/test_1h \
  --val-pred-dirs /data/v.pampinella/agy_project_orient: /data/v.pampinella/agy_project_eval/recheck:_recheck \
  --test-pred-dir test_runs --test-suffix _test \
  --deep 1h_lb_hm_sg_lb5_ablation_pred_s0 1h_lb_hm_sg_lb5_ablation_pred_s1 1h_lb_hm_sg_lb5_ablation_pred_s2 \
         1h_lb_hm_sg_lb5_tf64_ablation_pred_s0 1h_lb_hm_sg_lb5_tf64_ablation_pred_s1 1h_lb_hm_sg_lb5_ob_pred_s0 \
         1h_lb_hm_sg_lb5_tf64_hand_xp_ablation_pred_s0 1h_lb_hm_sg_lb5_tf64_hand_xp_ep2_ablation_pred_s0 \
         1h_lb_hm_sg_lb5_tf64_hand_xp_ema0.998_ablation_pred_s0 ${RES} \
         1h_lb_hm_sg_lb5_tsfm_ablation_pred_s0 1h_lb_hm_sg_lb5_tsfm_ablation_pred_s1 \
  --tab "TabM 3 seeds=$DA/tabular_net/proba_1h_tabm_${TN}_mean.npy:$DA/test_1h/tabular_net/proba_1h_tabm_${TN}_mean.npy" \
        "MLP 3 seeds=$DA/tabular_net/proba_1h_mlp_${TN}_mean.npy:$DA/test_1h/tabular_net/proba_1h_mlp_${TN}_mean.npy" \
  --title "Test set 24-26 March (1h)" --out reports/test_1h.md > reports/test_1h.log 2>&1
echo ">>> 1h full report exit $? $(date)"
echo ">>> REPORTS ROUND 2 DONE"
