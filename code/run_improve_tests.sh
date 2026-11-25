#!/bin/bash
# run_improve_tests.sh — short paired tests on top of the LOB-only model with skip path and global
# state (t_<H>_lob_skip_glob of run_navis_ctan_tests.sh is the reference).  Same setting: train
# 11–12 March, validate on 13 March, seed 0, CPU.  For each horizon:
#   sg_lo     + log-odds price features (level and change per lookback), also in the direct path
#   sg_mr     direct path initialised as Mean Reversion instead of zero
#   sg_hm     + hierarchical market memory (60 s, 3600 s)
#   sg_lb5    lookbacks 0.25, 1, 3, 6, 12 h instead of 1, 6, 12 h
#   sg_all    log-odds + Mean-Reversion init + hierarchical memory
#   nohup ./run_improve_tests.sh > improve_tests.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON="--device cpu --seed 0 --train-days 11-12 --val-days 13 --save-val-predictions --ablation --skip-lookback --global-state"
LB="--lookback-hours 1,6,12"
for H in 3600 43200; do
  L="$((H / 3600))h"
  r() { label=$1; shift; ./run_dev.sh train_full.py $COMMON --horizon $H --run-label t_${L}_$label "$@" > t_${L}_$label.log 2>&1 & }
  r sg_lo  $LB --lookback-logodds
  r sg_mr  $LB --skip-init meanrev
  r sg_hm  $LB --memory-periods 60,3600
  r sg_lb5 --lookback-hours 0.25,1,3,6,12
  r sg_all $LB --lookback-logodds --skip-init meanrev --memory-periods 60,3600
done
echo "started 10 runs ($(date))"
wait
echo "ALL DONE ($(date))"
