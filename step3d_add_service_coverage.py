"""
STEP 3D — Add service_coverage to stops_features_osm.csv
=========================================================
service_coverage = n_route_dir_qhr_rows from BUSTO survey aggregation.

Meaning: number of distinct (route × direction × quarter-hour) combinations
that serve each stop. Measures SERVICE SUPPLY DIVERSITY — how many different
bus services and time slots cover this stop.

Cold-start validity: for a NEW stop, derive this from the PLANNED timetable
(GTFS / Bus Open Data Service). For existing stops in BUSTO, it comes from
the survey rows directly. This is supply information, not demand — a planner
knows it before the stop opens.

Correlation with boardings is expected (more services → more passengers) but
correlation ≠ leakage: the causal direction is supply → demand, which is
exactly the relationship the model should learn.

Run time: < 5 seconds (pure pandas merge).
"""

import pandas as pd

BUSTO_CSV    = "busto_stop_level_boardings.csv"
FEATURES_CSV = "stops_features_osm.csv"
OUTPUT_CSV   = "stops_features_osm.csv"   # overwrite in place

print("Loading data...")
busto    = pd.read_csv(BUSTO_CSV,    dtype={"STOPCODE": "string"})
features = pd.read_csv(FEATURES_CSV, dtype={"STOPCODE": "string"})
print(f"  BUSTO:    {len(busto):,} stops")
print(f"  Features: {len(features):,} stops")

sc = (busto[["STOPCODE", "n_route_dir_qhr_rows"]]
      .rename(columns={"n_route_dir_qhr_rows": "service_coverage"}))

merged = features.merge(sc, on="STOPCODE", how="left")
merged["service_coverage"] = merged["service_coverage"].fillna(0).astype(int)

matched = merged["service_coverage"].gt(0).sum()
print(f"\nservice_coverage stats:")
print(f"  matched  = {matched:,} / {len(merged):,} stops  "
      f"({100*matched/len(merged):.1f}%)")
print(f"  mean     = {merged['service_coverage'].mean():.1f}")
print(f"  median   = {merged['service_coverage'].median():.0f}")
print(f"  max      = {merged['service_coverage'].max()}")
print(f"  pct_zero = {100*(merged['service_coverage']==0).mean():.1f}%")

merged.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved -> {OUTPUT_CSV}")
print(f"Columns now: {list(merged.columns)}")
