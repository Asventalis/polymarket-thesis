#!/bin/bash
# run_navis_ctan_tests.sh — short paired tests, on CPU, same setting as run_orientation_tests.sh
# (train 11–12 March, validate on 13 March, seed 0, lookback features), whose runs t_<H>_lob and
# t_<H>_full_ob are the references.  For each horizon:
#   lob_skip        LOB-only + direct path from the lookback features to the output
#   lob_skip_glob   the same + global state
#   lob_ctan        LOB-only with the CTAN state update (learned decay kept)
#   lob_ctan_dnone  the same without decay (CTAN as in the paper: dissipation only through gamma)
#   full_ob_ctan    graph model (book-oriented edges) with the CTAN update for markets and wallets
#   nohup ./run_navis_ctan_tests.sh > navis_ctan_tests.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
COMMON="--device cpu --seed 0 --lookback-hours 1,6,12 --train-days 11-12 --val-days 13 --save-val-predictions"
for H in 3600 43200; do
  L="$((H / 3600))h"
  r() { label=$1; shift; ./run_dev.sh train_full.py $COMMON --horizon $H --run-label t_${L}_$label "$@" > t_${L}_$label.log 2>&1 & }
  r lob_skip       --ablation --skip-lookback
  r lob_skip_glob  --ablation --skip-lookback --global-state
  r lob_ctan       --ablation --state-update ctan
  r lob_ctan_dnone --ablation --state-update ctan --decay-mode none
  r full_ob_ctan   --edge-orientation book --state-update ctan
done
echo "started 10 runs ($(date))"
wait
echo "ALL DONE ($(date))"
