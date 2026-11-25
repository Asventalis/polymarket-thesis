#!/bin/bash
# run_skill_tests.sh — short paired test of the wallet track-record edge features, on CPU.
# Same setting as run_orientation_tests.sh (train 11–12 March, validate on 13 March, seed 0,
# lookback features, book-oriented edges), whose runs t_<H>_full_ob and t_<H>_lob are the
# references.  For each horizon: real track record (_sk) and placebo (_skp, the record of
# another wallet), in parallel, 8 threads each.
#   nohup ./run_skill_tests.sh > skill_tests.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"
for H in 3600 43200; do
  L="$((H / 3600))h"
  [ -d wallet_skill/h$L ] || ./run_dev.sh build_wallet_skill.py --horizon $H --history-days 6-13 --target-days 11-13 > skill_build_$L.log 2>&1 \
    || { echo "skill build $L FAILED"; exit 1; }
  echo "skill $L ready ($(date))"
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON="--device cpu --seed 0 --lookback-hours 1,6,12 --train-days 11-12 --val-days 13 --save-val-predictions --edge-orientation book"
# (the placebo runs were restarted by hand with --skill-dir wallet_skill_v2 after the placebo became activity-matched)
for H in 3600 43200; do
  L="$((H / 3600))h"
  ./run_dev.sh train_full.py $COMMON --horizon $H --edge-skill real    --run-label t_${L}_full_ob_sk  > t_${L}_full_ob_sk.log 2>&1 &
  ./run_dev.sh train_full.py $COMMON --horizon $H --edge-skill placebo --run-label t_${L}_full_ob_skp > t_${L}_full_ob_skp.log 2>&1 &
done
echo "started 4 runs ($(date))"
wait
echo "ALL DONE ($(date))"
