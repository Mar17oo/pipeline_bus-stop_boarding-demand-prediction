"""
STEP 5A — Borough WMAPE choropleth map
=======================================
Creates a 4-panel London map showing per-borough WMAPE for each model
(HistAvg, RF, MLP, GATv2) using the best feature set (AI23+OSM).

Output: borough_wmape_map.png  (dissertation-quality figure, 300 dpi)

Requirements: geopandas, matplotlib (both already in environment)
Borough boundaries: ONS Open Geography Portal API (no login required)
Run time: ~60 seconds (one HTTP request for boundaries)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RESULTS_FILE = "results_cv_ai23_osm.csv"   # best feature set
OUTPUT_PNG   = "borough_wmape_map.png"

# ── 1. Load per-borough results ─────────────────────────────────────────────
print("Loading results...")
df = pd.read_csv(RESULTS_FILE)
print(f"  {len(df)} rows, {df['borough'].nunique()} boroughs, "
      f"{df['model'].nunique()} models")

# Pivot: one row per borough, columns = model WMAPE
pivot = df.pivot(index="borough", columns="model", values="WMAPE").reset_index()

# ── 2. Load London borough boundaries from ONS ───────────────────────────────
print("Downloading London borough boundaries from ONS...")
ONS_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_May_2023_UK_BFC_V2/FeatureServer/0/query"
    # LAD23CD prefix E09 = the 32 London boroughs + City of London (33 LADs).
    # Matching on LAD23NM LIKE '%London%' (previous query) only matched "City
    # of London" itself, since borough names like "Camden" don't contain the
    # word "London" — that bug meant only 1/33 boroughs ever rendered.
    "?where=LAD23CD+LIKE+'E09%25'"
    "&outFields=LAD23NM&returnGeometry=true"
    "&geometryPrecision=5&outSR=4326&f=geojson"
)
try:
    gdf = gpd.read_file(ONS_URL)
    gdf = gdf.rename(columns={"LAD23NM": "borough"})
    print(f"  Downloaded {len(gdf)} borough boundaries")
except Exception as e:
    print(f"  ONS API failed ({e}), trying backup URL...")
    BACKUP_URL = (
        "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/"
        "json/administrative/eng/lad.json"
    )
    gdf_all = gpd.read_file(BACKUP_URL)
    # Filter to London boroughs present in results
    london_names = set(pivot["borough"].str.strip())
    gdf = gdf_all[gdf_all["LAD13NM"].isin(london_names)].copy()
    gdf = gdf.rename(columns={"LAD13NM": "borough"})
    print(f"  Backup: {len(gdf)} boroughs matched")

# ── 3. Name normalisation ─────────────────────────────────────────────────────
# Minor spelling differences between ONS names and BUSTO names
NAME_MAP = {
    "Kingston upon Thames":     "Kingston upon Thames",
    "Richmond upon Thames":     "Richmond upon Thames",
    "Barking and Dagenham":     "Barking and Dagenham",
    "Hammersmith and Fulham":   "Hammersmith and Fulham",
    "Kensington and Chelsea":   "Kensington and Chelsea",
}
# Strip whitespace
gdf["borough"]   = gdf["borough"].str.strip()
pivot["borough"] = pivot["borough"].str.strip()

merged = gdf.merge(pivot, on="borough", how="inner")
print(f"  Merged: {len(merged)} boroughs matched to results")
if len(merged) < 30:
    unmatched = set(pivot["borough"]) - set(merged["borough"])
    print(f"  WARNING — unmatched boroughs: {unmatched}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
MODELS  = ["HistAvg", "RF", "MLP", "GATv2"]
TITLES  = ["(a) Historical Average", "(b) Random Forest",
           "(c) MLP", "(d) GATv2 (proposed)"]
COLOURS = ["#d73027","#fc8d59","#fee090","#e0f3f8","#91bfdb","#4575b4"]
BOUNDS  = [0.70, 0.77, 0.80, 0.83, 0.86, 0.92, 1.80]
CMAP    = plt.cm.RdYlBu_r
NORM    = BoundaryNorm(BOUNDS, CMAP.N)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "WMAPE by London Borough — Leave-Borough-Out CV (AI23 + OSM features)\n"
    "Lower (blue) = better prediction   |   Higher (red) = harder to predict",
    fontsize=13, y=0.98
)

for ax, model, title in zip(axes.flat, MODELS, TITLES):
    merged.plot(
        column=model,
        ax=ax,
        cmap=CMAP,
        norm=NORM,
        edgecolor="white",
        linewidth=0.4,
    )
    # Borough labels for the worst and best performers
    for _, row in merged.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        val = row[model]
        if val >= merged[model].quantile(0.85) or val <= merged[model].quantile(0.15):
            ax.annotate(
                f"{row['borough'].split()[0]}\n{val:.2f}",
                xy=(cx, cy), ha="center", va="center",
                fontsize=5.5, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none")
            )
    ax.set_title(f"{title}\nMean WMAPE = {merged[model].mean():.3f}", fontsize=11)
    ax.axis("off")

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.02, pad=0.02, shrink=0.75)
cbar.set_label("WMAPE (lower = better)", fontsize=11)
cbar.set_ticks(BOUNDS)
cbar.set_ticklabels([f"{b:.2f}" for b in BOUNDS])

plt.tight_layout(rect=[0, 0, 0.92, 0.96])
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"\nSaved -> {OUTPUT_PNG}")

# ── 5. Print ranked borough table ────────────────────────────────────────────
print("\nPer-borough WMAPE (AI23+OSM, sorted by GATv2):")
print(f"{'Borough':<32} {'HistAvg':>8} {'RF':>8} {'MLP':>8} {'GATv2':>8}")
print("-" * 68)
for _, row in merged.sort_values("GATv2").iterrows():
    print(f"{row['borough']:<32} {row['HistAvg']:>8.3f} {row['RF']:>8.3f} "
          f"{row['MLP']:>8.3f} {row['GATv2']:>8.3f}")
