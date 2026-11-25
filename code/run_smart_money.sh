#!/bin/bash
# run_smart_money.sh — smart-money test at the three horizons (CPU only, ~no GPU use)
#   nohup ./run_smart_money.sh > smart_money/chain.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p smart_money
for H in 3600 21600 43200; do
  L="$((H / 3600))h"
  echo ">>> smart money $L ($(date))"
  ./run_dev.sh smart_money_test.py --horizon $H --out smart_money/smart_money_$L.json > smart_money/smart_money_$L.log 2>&1
  echo ">>> smart money $L DONE rc=$? ($(date))"
done
echo ">>> ALL DONE ($(date))"
