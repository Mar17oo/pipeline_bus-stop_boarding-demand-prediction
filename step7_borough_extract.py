"""
STEP 7 — Per-borough extract for the results-chapter discussion
===================================================================
From the headline AI23+OSM+SC per-fold results, prints:
  - Camden and Hillingdon: GATv2 vs RF WMAPE (named in the dissertation's
    cross-borough-aggregation-contamination argument)
  - The 3 boroughs where GATv2 loses worst vs RF (GATv2 - RF largest positive gap)
  - The 3 boroughs where GATv2 is closest to RF (smallest |GATv2 - RF| gap)

Usage: python step7_borough_extract.py [results_cv_file.csv] [model_col]
Default file: results_cv_ai23_osm_sc.csv (headline experiment)
Default model_col: GATv2 — pass explicitly for variant CSVs that carry a
differently-named graph model alongside the reused "GATv2" baseline column
(e.g. results_cv_zheng_fusion.csv has BOTH "GATv2" (reused headline) and
"GATv2-Fusion" (the new variant) — passing no model_col there would silently
extract the WRONG (reused, not new) column).
"""

import sys
import pandas as pd

CV_FILE   = sys.argv[1] if len(sys.argv) > 1 else "results_cv_ai23_osm_sc.csv"
MODEL_COL = sys.argv[2] if len(sys.argv) > 2 else "GATv2"
NAMED_BOROUGHS = ["Camden", "Hillingdon"]


def main():
    df = pd.read_csv(CV_FILE)
    pivot = df.pivot(index="borough", columns="model", values="WMAPE")
    if MODEL_COL not in pivot.columns:
        print(f"ERROR: model column '{MODEL_COL}' not found. Available: {list(pivot.columns)}")
        return
    pivot["gap"] = pivot[MODEL_COL] - pivot["RF"]

    print(f"Source: {CV_FILE}  (graph model column: {MODEL_COL})\n")

    print("Named boroughs (Camden / Hillingdon):")
    for b in NAMED_BOROUGHS:
        matches = [idx for idx in pivot.index if b.lower() in idx.lower()]
        if not matches:
            print(f"  {b}: NOT FOUND in {CV_FILE}")
            continue
        for m in matches:
            r = pivot.loc[m]
            print(f"  {m:<30s} {MODEL_COL}={r[MODEL_COL]:.4f}  RF={r['RF']:.4f}  "
                  f"gap({MODEL_COL}-RF)={r['gap']:+.4f}")

    print(f"\n3 boroughs where {MODEL_COL} loses WORST vs RF (largest gap):")
    worst = pivot.sort_values("gap", ascending=False).head(3)
    for idx, r in worst.iterrows():
        print(f"  {idx:<30s} {MODEL_COL}={r[MODEL_COL]:.4f}  RF={r['RF']:.4f}  gap={r['gap']:+.4f}")

    print(f"\n3 boroughs where {MODEL_COL} is CLOSEST to RF (smallest |gap|):")
    closest = pivot.reindex(pivot["gap"].abs().sort_values().index).head(3)
    for idx, r in closest.iterrows():
        print(f"  {idx:<30s} {MODEL_COL}={r[MODEL_COL]:.4f}  RF={r['RF']:.4f}  gap={r['gap']:+.4f}")

    keep_cols = ["borough"] + [c for c in ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", MODEL_COL] if c in pivot.columns] + ["gap"]
    out = pivot.reset_index()[keep_cols]
    out.to_csv("borough_extract.csv", index=False)
    print(f"\nSaved full per-borough pivot -> borough_extract.csv")


if __name__ == "__main__":
    main()
