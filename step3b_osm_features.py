"""
STEP 3b (OPTIONAL) — Add OpenStreetMap POI features to stops_features.csv
=========================================================================
Matches Zheng et al. (2025) POI categories:
  residences, shopping, companies, education, entertainment

Uses osmnx to download all London POIs ONCE (not per-stop API calls).
Then does fast spatial join: count POIs within 500m of each bus stop.

Run AFTER step3_lsoa_features.py (requires stops_features.csv).
Produces: stops_features_osm.csv  (same as stops_features.csv + 5 OSM columns)

Install: pip install osmnx geopandas shapely
Runtime: ~20 min download + ~10 min spatial join
=========================================================================
"""

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

INPUT_CSV   = "stops_features.csv"
OUTPUT_CSV  = "stops_features_osm.csv"
BUFFER_M    = 500   # metres — matches Zheng et al. (2025)
CRS_METRIC  = "EPSG:27700"  # British National Grid (metres)
CRS_WGS84   = "EPSG:4326"

# ---------------------------------------------------------------------------
# POI tag categories — mapped to Zheng et al. (2025) five functional types
# ---------------------------------------------------------------------------
POI_CATEGORIES = {
    "poi_residential": {
        "landuse": ["residential"],
    },
    "poi_shopping": {
        "shop": True,               # all shop types
        "amenity": ["marketplace"],
    },
    "poi_company": {
        "office": True,
        "landuse": ["commercial", "industrial", "retail"],
    },
    "poi_education": {
        "amenity": ["school", "university", "college", "kindergarten", "library"],
        "building": ["school", "university"],
    },
    "poi_entertainment": {
        "amenity": ["restaurant", "cafe", "bar", "pub", "fast_food",
                    "cinema", "theatre", "nightclub", "gym"],
        "leisure": ["park", "sports_centre", "fitness_centre", "swimming_pool"],
    },
}


def download_pois(place="Greater London, UK"):
    """Download POIs for all categories from OSM for the given place."""
    print(f"Downloading OSM data for '{place}'...")
    print("(This queries OpenStreetMap once — takes ~10-20 min on first run)")

    all_pois = {}
    for cat_name, tags in POI_CATEGORIES.items():
        print(f"  Fetching: {cat_name} ...", end=" ", flush=True)
        try:
            gdf = ox.features_from_place(place, tags=tags)
            # Keep only point and polygon centroids
            gdf = gdf[["geometry"]].copy()
            gdf["geometry"] = gdf.geometry.centroid
            gdf = gdf.to_crs(CRS_METRIC)
            all_pois[cat_name] = gdf
            print(f"{len(gdf):,} features")
        except Exception as e:
            print(f"FAILED ({e}) — setting to 0 for all stops")
            all_pois[cat_name] = gpd.GeoDataFrame(geometry=[], crs=CRS_METRIC)

    return all_pois


def count_pois_within_buffer(stops_gdf, pois_gdf, buffer_m):
    """Count POI points within buffer_m metres of each stop."""
    buffered = stops_gdf.geometry.buffer(buffer_m)
    counts = []
    for buf in buffered:
        mask = pois_gdf.geometry.within(buf)
        counts.append(int(mask.sum()))
    return counts


def main():
    # 1. Load stops
    print(f"Loading {INPUT_CSV} ...")
    stops = pd.read_csv(INPUT_CSV)
    print(f"  {len(stops):,} stops loaded")

    # 2. Convert to GeoDataFrame (BNG metres for accurate buffering)
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=[Point(lon, lat) for lon, lat in zip(stops["lon"], stops["lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    # 3. Download all London POIs
    all_pois = download_pois("Greater London, UK")

    # 4. Count POIs within 500m for each category
    print("\nCounting POIs within 500m of each stop...")
    for cat_name, pois_gdf in all_pois.items():
        print(f"  {cat_name} ...", end=" ", flush=True)
        if len(pois_gdf) == 0:
            stops[cat_name] = 0
        else:
            stops[cat_name] = count_pois_within_buffer(stops_gdf, pois_gdf, BUFFER_M)
        print(f"done  (mean={stops[cat_name].mean():.1f}, max={stops[cat_name].max()})")

    # 5. Save
    stops.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")
    print(f"New columns: {list(POI_CATEGORIES.keys())}")
    print("\nNEXT: in step4_model.py, add these 5 columns to FEAT_COLS and re-run.")
    print("      Or point STOPS_CSV = 'stops_features_osm.csv' at the top of step4.")

    # 6. Quick stats
    print("\n--- POI summary (median per stop) ---")
    for col in POI_CATEGORIES:
        print(f"  {col:25s}: median={stops[col].median():.0f}  "
              f"pct_zero={100*(stops[col]==0).mean():.1f}%")


if __name__ == "__main__":
    main()
