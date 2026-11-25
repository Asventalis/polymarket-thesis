#!/bin/bash
# run_orientation_tests.sh — short paired test of the edge-orientation fix, on CPU (the GPUs
# are running the memory study, one run per GPU).  Train 11–12 March, validate on 13 March,
# seed 0, lookback features; for each horizon: full model with the raw direction, full model
# with the book-oriented direction, LOB-only reference.  All runs in parallel, 8 threads each.
#   nohup ./run_orientation_tests.sh > orient_tests.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON="--device cpu --seed 0 --lookback-hours 1,6,12 --train-days 11-12 --val-days 13 --save-val-predictions"
for H in 3600 43200; do
  L="$((H / 3600))h"
  ./run_dev.sh train_full.py $COMMON --horizon $H --run-label t_${L}_full                               > t_${L}_full.log 2>&1 &
  ./run_dev.sh train_full.py $COMMON --horizon $H --run-label t_${L}_full_ob --edge-orientation book    > t_${L}_full_ob.log 2>&1 &
  ./run_dev.sh train_full.py $COMMON --horizon $H --run-label t_${L}_lob --ablation                     > t_${L}_lob.log 2>&1 &
done
echo "started 6 runs ($(date))"
wait
echo "ALL DONE ($(date))"
