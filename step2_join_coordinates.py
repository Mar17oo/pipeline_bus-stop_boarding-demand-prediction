"""
============================================================================
STEP 2 (REAL DATA) — Join BUSTO stops to coordinates
============================================================================
Takes:
  - busto_stop_level_boardings.csv   (output of step1_aggregate_busto.py)
  - bus stop locations file          (STOP_CODE, OS_EASTING, OS_NORTHING, ...)
Produces:
  - stops_with_coords.csv            (STOPCODE | boardings | lat | lon | borough-ready)

CRITICAL: this script first VERIFIES if BUSTO STOPCODE actually
matches STOP_CODE in Bus_Stops.csv before joining.

============================================================================
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONFIG — point these at your real files.
#    NOTE: this version matches the TfL GIS Open Data Hub bus stops file, whose
#    columns are STOP_CODE, NAPTAN_ATCO, OS_EASTING, OS_NORTHING, STOP_NAME, etc.
# ---------------------------------------------------------------------------
BUSTO_STOPS = "busto_stop_level_boardings.csv"     # from step 1
LOCATIONS   = "data/Bus_Stops.csv"                  # the FULL GIS locations file
# Which column in the locations file matches BUSTO STOPCODE?
# For the GIS file this is 'STOP_CODE' (same BP#### / plain-number format as BUSTO).
LOC_CODE_COL = "STOP_CODE"
# Coordinate column names in the GIS file:
EASTING_COL = "OS_EASTING"
NORTHING_COL = "OS_NORTHING"


def load_inputs():
    busto = pd.read_csv(BUSTO_STOPS, dtype={"STOPCODE": "string"})
    loc = pd.read_csv(LOCATIONS, dtype={LOC_CODE_COL: "string"})
    # normalise codes: strip spaces, uppercase, remove accidental '.0' from floats
    busto["STOPCODE"] = busto["STOPCODE"].str.strip().str.upper()
    loc[LOC_CODE_COL] = (loc[LOC_CODE_COL].astype("string")
                         .str.replace(r"\.0$", "", regex=True).str.strip().str.upper())
    return busto, loc


# ---------------------------------------------------------------------------
# 1. VERIFY the code match BEFORE joining.
# ---------------------------------------------------------------------------
def verify_match(busto, loc, code_col):
    busto_codes = set(busto["STOPCODE"].dropna())
    loc_codes = set(loc[code_col].dropna())
    overlap = busto_codes & loc_codes
    pct = 100 * len(overlap) / max(1, len(busto_codes))
    print("=" * 60)
    print(f"VERIFICATION — does BUSTO STOPCODE match '{code_col}'?")
    print(f"  BUSTO unique stops      : {len(busto_codes):,}")
    print(f"  Locations unique codes  : {len(loc_codes):,}")
    print(f"  Matched                 : {len(overlap):,}  ({pct:.1f}% of BUSTO stops)")
    if pct > 80:
        print("  VERDICT: STRONG MATCH -> this is the right join key. Proceed.")
    elif pct > 30:
        print("  VERDICT: PARTIAL match -> check code formatting, or try another column.")
    else:
        print("  VERDICT: POOR match -> WRONG column. Try Bus_Stop_Code or Naptan_Atco,")
        print("           or you need a code-translation lookup. Raise with supervisor.")
    print("  Example unmatched BUSTO codes:",
          list(busto_codes - loc_codes)[:5])
    print("=" * 60)
    return pct


# ---------------------------------------------------------------------------
# 2. Convert Easting/Northing (EPSG:27700) -> Lat/Lon (EPSG:4326).
# ---------------------------------------------------------------------------
def add_latlon(df, easting=None, northing=None):
    easting = easting or EASTING_COL
    northing = northing or NORTHING_COL
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(df[easting].to_numpy(), df[northing].to_numpy())
    df = df.copy()
    df["lat"], df["lon"] = lat, lon
    return df


# ---------------------------------------------------------------------------
# 3. Main join.
# ---------------------------------------------------------------------------
def main():
    busto, loc = load_inputs()
    pct = verify_match(busto, loc, LOC_CODE_COL)
    if pct < 30:
        print("\nStopping: fix the join key before continuing.")
        return

    loc = add_latlon(loc)

    merged = busto.merge(
        loc[[LOC_CODE_COL, "lat", "lon", "NAPTAN_ATCO", "STOP_NAME"]],
        left_on="STOPCODE", right_on=LOC_CODE_COL, how="left",
    )

    n_missing = merged["lat"].isna().sum()
    print(f"\nJoined {len(merged):,} stops; {n_missing:,} have no coordinates "
          f"({100*n_missing/len(merged):.1f}%).")
    print("Stops missing coords are usually virtual/notional stops -> consider dropping.")

    merged = merged.dropna(subset=["lat", "lon"])
    merged.to_csv("stops_with_coords.csv", index=False)
    print(f"\nSaved -> stops_with_coords.csv  ({len(merged):,} stops with coordinates)")
    print("\nNEXT (step 3): point-in-polygon lat/lon -> LSOA, then merge AI23 features "
          "+ ONS borough lookup, then filter to London.")


if __name__ == "__main__":
    main()
