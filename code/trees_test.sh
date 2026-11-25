#!/bin/bash
# tree log-probabilities for 21-26 March (warm-up 21-23 + test 24-26), model fitted on 6-12 as in validation
cd /data/v.pampinella/agy_project_eval
nice -n 10 ./run_dev.sh tree_features.py /data/v.pampinella/agy_project/data/derived/tree_logprobs_1h_test --horizon 3600 --rounds 536 --every 10 --threads 24   --days 21,22,23,24,25,26 --train-days 6,7,8,9,10,11,12 --end-day 26 --hand-dir /data/v.pampinella/agy_project/data/derived/hand_features --hand-dirs /data/v.pampinella/agy_project/data/derived/hand_features_test
echo ">>> trees 1h exit $?"
nice -n 10 ./run_dev.sh tree_features.py /data/v.pampinella/agy_project/data/derived/tree_logprobs_1m_test --horizon 60 --rounds 324 --every 10 --threads 24   --days 21,22,23,24,25,26 --train-days 6,7,8,9,10,11,12 --end-day 26 --hand-dir /data/v.pampinella/agy_project/data/derived/hand_features --hand-dirs /data/v.pampinella/agy_project/data/derived/hand_features_test
echo ">>> trees 1m exit $?"
echo ">>> TREES DONE"
