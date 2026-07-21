"""
Add poi_scenic (scenic spots) to stops_features_osm.csv.
Matches Zheng et al. (2025) 7th POI category.
Runs in ~10 min (one OSM query + spatial join).
"""
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

INPUT_CSV  = "stops_features_osm.csv"
OUTPUT_CSV = "stops_features_osm.csv"   # overwrite in place
BUFFER_M   = 500
CRS_METRIC = "EPSG:27700"
CRS_WGS84  = "EPSG:4326"

SCENIC_TAGS = {
    "tourism": ["attraction", "viewpoint", "museum", "gallery", "zoo",
                "theme_park", "artwork"],
    "historic": True,       # all historic tags (monuments, ruins, memorials)
    "amenity": ["place_of_worship"],
}

print("Loading stops_features_osm.csv ...")
stops = pd.read_csv(INPUT_CSV)
print(f"  {len(stops):,} stops")

stops_gdf = gpd.GeoDataFrame(
    stops,
    geometry=[Point(lon, lat) for lon, lat in zip(stops["lon"], stops["lat"])],
    crs=CRS_WGS84,
).to_crs(CRS_METRIC)

print("Downloading scenic spots from OSM for Greater London ...")
try:
    scenic = ox.features_from_place("Greater London, UK", tags=SCENIC_TAGS)
    scenic = scenic[["geometry"]].copy()
    scenic["geometry"] = scenic.geometry.centroid
    scenic = scenic.to_crs(CRS_METRIC)
    print(f"  {len(scenic):,} scenic features found")
except Exception as e:
    print(f"  Download failed: {e} — setting poi_scenic = 0")
    stops["poi_scenic"] = 0
    stops.to_csv(OUTPUT_CSV, index=False)
    raise SystemExit

print("Counting scenic spots within 500m of each stop ...")
buffered = stops_gdf.geometry.buffer(BUFFER_M)
counts = [int(scenic.geometry.within(buf).sum()) for buf in buffered]
stops["poi_scenic"] = counts

print(f"  mean={stops['poi_scenic'].mean():.1f}  "
      f"max={stops['poi_scenic'].max()}  "
      f"pct_zero={100*(stops['poi_scenic']==0).mean():.1f}%")

stops.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved -> {OUTPUT_CSV}")
print(f"Columns now: {[c for c in stops.columns if c.startswith('poi_')]}")
