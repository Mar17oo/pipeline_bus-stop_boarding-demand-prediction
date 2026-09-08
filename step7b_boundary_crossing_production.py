"""
STEP 7B — Boundary crossing on the production multigraph
==========================================================
step5a's porosity-graph check (K=5 KNN only, one-directional, not
deduplicated, no route edges -- 89,715 edges) is a diagnostic graph, not
what GATv2 actually trains on. This rebuilds the real production
multigraph the same way step4_model.py does -- K=5 KNN + route edges,
deduplicated, 124,234 directed edges, using step4_model's own
build_knn_edge_index / build_route_edges so the graph checked here is
edge-for-edge identical to training -- and computes:

  1. Per-borough boundary-crossing share on that graph (touched stops,
     not edges): boundary_diagnostic_by_borough_production.csv
  2. Its Pearson correlation with the GATv2-MLP WMAPE gap
     (results_cv_ai23_osm_sc.csv), the same mechanism check as the
     porosity-graph version, on the graph the model actually sees.

Needs data/ (raw BUSTO quarter-hour CSVs) for build_route_edges, same
requirement as step4_model.py itself -- this is a core-pipeline script,
not one of the fast no-retrain checks in reproduce_tables_and_stats.ipynb.

Output:
  boundary_diagnostic_by_borough_production.csv
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats

import step4_model as m

OUTPUT_CSV = "boundary_diagnostic_by_borough_production.csv"
RESULTS_CV = "results_cv_ai23_osm_sc.csv"

print("Loading features and rebuilding the production multigraph...")
feat = pd.read_csv(m.DATA_FILE)
lad = feat[m.BOROUGH_COL].values

knn_ei = m.build_knn_edge_index(feat["lat"].values, feat["lon"].values)
route_ei = m.build_route_edges(feat)
full_ei = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
src, dst = full_ei[0].numpy(), full_ei[1].numpy()
cross = lad[src] != lad[dst]

n_edges = len(src)
n_cross_edges = int(cross.sum())
print(f"  {n_edges:,} directed edges, {n_cross_edges:,} "
      f"({100 * n_cross_edges / n_edges:.2f}%) cross a borough boundary")

touched = np.zeros(len(feat), dtype=bool)
touched[src[cross]] = True
touched[dst[cross]] = True

per_borough = (
    pd.DataFrame({"lad_name": lad, "touched": touched})
    .groupby("lad_name")
    .agg(n_stops=("touched", "size"), pct_boundary=("touched", lambda s: 100 * s.mean()))
    .sort_values("pct_boundary", ascending=False)
)
per_borough.to_csv(OUTPUT_CSV)
print(f"  Saved -> {OUTPUT_CSV}")
print(per_borough.head(5))

kensington_pct = per_borough.loc["Kensington and Chelsea", "pct_boundary"]
print(f"\nKensington and Chelsea crossing share (production graph): {kensington_pct:.1f}%")

# ── Correlation with the GATv2-MLP WMAPE gap ─────────────────────────────
print(f"\nLoading {RESULTS_CV} for the GATv2-MLP gap...")
cv = pd.read_csv(RESULTS_CV)
piv = cv.pivot(index="borough", columns="model", values="WMAPE")
delta = piv["GATv2"] - piv["MLP"]

merged = per_borough[["pct_boundary"]].join(delta.rename("delta"), how="inner")
assert len(merged) == 33, f"expected 33 boroughs, got {len(merged)}"

r, p = stats.pearsonr(merged["pct_boundary"], merged["delta"])
print(f"Pearson r = {r:.4f}, p = {p:.4f}, n = {len(merged)}  "
      f"(pct_boundary vs GATv2-MLP WMAPE delta, production graph)")
