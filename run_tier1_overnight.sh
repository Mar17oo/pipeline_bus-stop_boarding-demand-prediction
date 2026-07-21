#!/bin/bash
# Tier 1 overnight driver — runs the 3 required GATv2 experiments sequentially,
# then the optional V/C experiment. Continues past a failed step (does not
# abort the whole queue) so a crash in one experiment doesn't block the rest.
set -u
cd "c:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"

echo "=== [1/4] OSM + service_coverage ($(date)) ==="
python -u step4_model.py --osm-only --with-sc
if [ $? -eq 0 ]; then
  cp results_cv_multigraph.csv results_cv_osm_sc.csv
  cp results_summary_multigraph.csv results_summary_osm_sc.csv
  echo "=== [1/4] DONE -> results_*_osm_sc.csv ==="
else
  echo "=== [1/4] FAILED — see output above ==="
fi

echo "=== [2/4] AI23 + OSM + service_coverage (headline) ($(date)) ==="
python -u step4_model.py --with-sc
if [ $? -eq 0 ]; then
  cp results_cv_multigraph.csv results_cv_ai23_osm_sc.csv
  cp results_summary_multigraph.csv results_summary_ai23_osm_sc.csv
  echo "=== [2/4] DONE -> results_*_ai23_osm_sc.csv ==="
else
  echo "=== [2/4] FAILED — see output above ==="
fi

echo "=== [3/4] GNN-fairness check: K=10, AI23+OSM+SC ($(date)) ==="
python -u step4_model.py --with-sc --k10
if [ $? -eq 0 ]; then
  cp results_cv_multigraph.csv results_cv_gnn_fairness.csv
  cp results_summary_multigraph.csv results_summary_gnn_fairness.csv
  echo "=== [3/4] DONE -> results_*_gnn_fairness.csv ==="
else
  echo "=== [3/4] FAILED — see output above ==="
fi

echo "=== [4/4] OPTIONAL: V/C overcrowding experiment ($(date)) ==="
python -u step5b_vc_experiment.py
if [ $? -eq 0 ]; then
  echo "=== [4/4] DONE -> results_*_vc_ai23_osm.csv ==="
else
  echo "=== [4/4] FAILED — see output above ==="
fi

echo "=== ALL TIER1 RUNS FINISHED ($(date)) ==="
