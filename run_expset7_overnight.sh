#!/bin/bash
# Experiment Set 7 driver: functional-similarity edges (option 2), then
# Zheng-style G1/G2/G3 fusion architecture (option 3). Sequential to avoid
# CPU contention. Continues past a failed step.
set -u
cd "c:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"

echo "=== [1/2] Functional-similarity edges (--func-sim) ($(date)) ==="
python -u step4_model.py --with-sc --func-sim
if [ $? -eq 0 ]; then
  cp results_cv_multigraph.csv results_cv_func_sim.csv
  cp results_summary_multigraph.csv results_summary_func_sim.csv
  echo "=== [1/2] DONE -> results_*_func_sim.csv ==="
else
  echo "=== [1/2] FAILED — see output above ==="
fi

echo "=== [2/2] Zheng G1/G2/G3 fusion architecture ($(date)) ==="
python -u step4e_zheng_fusion.py
if [ $? -eq 0 ]; then
  echo "=== [2/2] DONE -> results_*_zheng_fusion.csv ==="
else
  echo "=== [2/2] FAILED — see output above ==="
fi

echo "=== EXPERIMENT SET 7 FINISHED ($(date)) ==="
