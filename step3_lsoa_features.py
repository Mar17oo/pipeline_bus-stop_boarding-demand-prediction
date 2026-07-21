"""
STEP 3 — Assign LSOA, filter to London, join AI23 accessibility features
=========================================================================
Takes:
  stops_with_coords.csv              (step 2 output)
  data/Bus_Stops.csv                 (TfL GIS file — for postcodes)
  data/access_*.csv                  (8 AI23 feature files)
  data/lsoa21_to_lsoa11_lookup.csv   (auto-downloaded from ONS if missing)

Produces:
  stops_features.csv  —  London stops: demand + coords + LSOA + borough + 8 AI23 features

Pipeline:
  1. Attach postcodes from TfL GIS file
  2. Batch-query postcodes.io -> LSOA code + borough (free API, no key)
  3. Apply ONS LSOA 2021->2011 crosswalk to recover stops in reorganised areas
  4. Filter to London (LAD codes E09*)
  5. Join 8 AI23 accessibility features (pt, am, 30-min threshold)
=========================================================================
"""

import os
import io
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
STOPS_WITH_COORDS     = "stops_with_coords.csv"
BUS_STOPS_GIS         = "data/Bus_Stops.csv"
LSOA_CROSSWALK_CACHE  = "data/lsoa21_to_lsoa11_lookup.csv"
# ONS "LSOA 2011 to LSOA 2021 to LAD 2022 Best Fit Lookup for EW (V2)"
LSOA_CROSSWALK_URL    = ("https://open-geography-portalx-ons.hub.arcgis.com"
                         "/api/download/v1/items/b684a0dbf786473f9563ec0616da2f8b"
                         "/csv?layers=0")

AI23_FEATURES = [
    # (file, raw column, output name)
    ("data/access_employment_all_pt.csv",   "access_employment_all_30",   "employment_all_30min"),
    ("data/access_hospitals_pt.csv",        "access_hospitals_30",        "hospitals_30min"),
    ("data/access_gp_practices_pt.csv",     "access_gp_practices_30",     "gp_30min"),
    ("data/access_supermarkets_pt.csv",     "access_supermarkets_30",     "supermarkets_30min"),
    ("data/access_pharmacies_pt.csv",       "access_pharmacies_30",       "pharmacies_30min"),
    ("data/access_primary_schools_pt.csv",  "access_primary_schools_30",  "primary_schools_30min"),
    ("data/access_secondary_schools_pt.csv","access_secondary_schools_30","secondary_schools_30min"),
    ("data/access_main_bua_pt.csv",         "access_main_bua_30",         "main_bua_30min"),
]
AI23_TIME_BAND = "am"      # morning peak aligns with BUSTO typical-weekday demand

POSTCODES_IO_URL = "https://api.postcodes.io/postcodes"
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# 1. Load stops + attach postcodes from TfL GIS file
# ---------------------------------------------------------------------------
def load_stops_with_postcodes():
    stops = pd.read_csv(STOPS_WITH_COORDS, dtype=str)
    gis   = pd.read_csv(BUS_STOPS_GIS, dtype=str, usecols=["STOP_CODE", "POSTCODE"])
    gis["STOP_CODE"]   = gis["STOP_CODE"].str.strip().str.upper()
    stops["STOP_CODE"] = stops["STOP_CODE"].str.strip().str.upper()
    stops = stops.merge(gis, on="STOP_CODE", how="left")
    n_miss = stops["POSTCODE"].isna().sum()
    print(f"  Stops with postcode : {stops['POSTCODE'].notna().sum():,} / {len(stops):,}  "
          f"({n_miss} missing)")
    return stops.dropna(subset=["POSTCODE"]).copy()


# ---------------------------------------------------------------------------
# 2. Batch-query postcodes.io -> LSOA code + borough
# ---------------------------------------------------------------------------
def lookup_postcodes(postcodes: list) -> pd.DataFrame:
    rows, total = [], len(postcodes)
    for i in range(0, total, BATCH_SIZE):
        batch = postcodes[i: i + BATCH_SIZE]
        resp  = requests.post(POSTCODES_IO_URL, json={"postcodes": batch}, timeout=30)
        resp.raise_for_status()
        for item in resp.json()["result"]:
            pc = item["query"]
            r  = item["result"]
            if r is None:
                rows.append({"POSTCODE": pc, "pc_lsoa": None,
                             "lad_code": None, "lad_name": None})
            else:
                rows.append({
                    "POSTCODE": pc,
                    "pc_lsoa": r.get("codes", {}).get("lsoa"),   # may be 2011 or 2021
                    "lad_code": r.get("codes", {}).get("admin_district"),
                    "lad_name": r.get("admin_district"),
                })
        done = min(i + BATCH_SIZE, total)
        if done % 1000 == 0 or done == total:
            print(f"    postcodes.io: {done:,}/{total:,}")
        time.sleep(0.05)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Download (once) and apply ONS LSOA 2021->2011 best-fit crosswalk
#    Reason: postcodes.io returns 2021 LSOA codes for areas reorganised after
#    the 2021 census boundary revision. AI23 uses 2011 geography, so those codes
#    don't match. The crosswalk maps each 2021 LSOA back to its 2011 parent.
# ---------------------------------------------------------------------------
def load_lsoa_crosswalk() -> dict:
    if not os.path.exists(LSOA_CROSSWALK_CACHE):
        print(f"  Downloading ONS LSOA 2021->2011 best-fit lookup (~9 MB)...")
        r = requests.get(LSOA_CROSSWALK_URL, timeout=120)
        r.raise_for_status()
        with open(LSOA_CROSSWALK_CACHE, "wb") as f:
            f.write(r.content)
        print(f"  Saved -> {LSOA_CROSSWALK_CACHE}")
    df = pd.read_csv(LSOA_CROSSWALK_CACHE, dtype=str)
    # Column detection: file has LSOA11CD and LSOA21CD
    col21 = next(c for c in df.columns if "21" in c and "CD" in c.upper() and "LSOA" in c.upper())
    col11 = next(c for c in df.columns if "11" in c and "CD" in c.upper() and "LSOA" in c.upper())
    xwalk = dict(zip(df[col21].str.strip(), df[col11].str.strip()))
    print(f"  Crosswalk loaded: {len(xwalk):,} LSOA21->LSOA11 mappings  "
          f"(cols: {col21} -> {col11})")
    return xwalk


def apply_crosswalk(codes: pd.Series, ai23_codes: set, xwalk: dict) -> pd.Series:
    """Return LSOA11CD to use for AI23 join: keep as-is if already in AI23, else remap."""
    def remap(c):
        if pd.isna(c):
            return c
        if c in ai23_codes:
            return c          # already a valid 2011 code
        return xwalk.get(c, c)  # try crosswalk; leave unchanged if not found
    return codes.map(remap)


# ---------------------------------------------------------------------------
# 4. Load one AI23 file filtered to pt + am
# ---------------------------------------------------------------------------
def load_ai23_feature(path, col, out_name):
    df = pd.read_csv(path, dtype={"geo_code": "string"})
    df = df[(df["mode"] == "pt") & (df["time_of_day"] == AI23_TIME_BAND)]
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in {path}. Have: {df.columns.tolist()}")
    return df[["geo_code", col]].rename(columns={col: out_name})


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    # — 5a. Stops + postcodes
    print("\n[1] Loading stops + postcodes...")
    stops = load_stops_with_postcodes()

    # — 5b. Postcode -> LSOA + borough
    unique_pcs = stops["POSTCODE"].str.strip().str.upper().unique().tolist()
    print(f"\n[2] Querying postcodes.io ({len(unique_pcs):,} unique postcodes, "
          f"{len(unique_pcs)//BATCH_SIZE+1} batches)...")
    pc_info = lookup_postcodes(unique_pcs)
    pc_info["POSTCODE"] = pc_info["POSTCODE"].str.strip().str.upper()
    stops["POSTCODE"]   = stops["POSTCODE"].str.strip().str.upper()
    stops = stops.merge(pc_info, on="POSTCODE", how="left")

    n_no_pc = stops["pc_lsoa"].isna().sum()
    print(f"  {n_no_pc} stops with no LSOA from postcodes.io (invalid/terminated) — dropped.")
    stops = stops.dropna(subset=["pc_lsoa"])

    # — 5c. Filter to London
    london = stops[stops["lad_code"].str.startswith("E09", na=False)].copy()
    print(f"\n[3] London stops : {len(london):,}  |  outside London : {len(stops)-len(london):,}")
    print(f"    Boroughs      : {london['lad_name'].nunique()}")

    # — 5d. Load AI23
    print("\n[4] Loading AI23 features (pt, am, 30-min)...")
    ai23 = None
    for path, col, out_name in AI23_FEATURES:
        feat = load_ai23_feature(path, col, out_name)
        print(f"    {out_name}: {len(feat):,} LSOAs")
        ai23 = feat if ai23 is None else ai23.merge(feat, on="geo_code", how="outer")
    ai23_codes = set(ai23["geo_code"].dropna())

    # — 5e. First-pass join on raw LSOA code
    london = london.merge(ai23, left_on="pc_lsoa", right_on="geo_code", how="left")
    first_feat = AI23_FEATURES[0][2]
    n_miss = london[first_feat].isna().sum()
    print(f"\n[5] First-pass join: {len(london)-n_miss:,} matched, {n_miss:,} unmatched.")

    # — 5f. Crosswalk fix for unmatched (2021 LSOA codes -> 2011 parent)
    if n_miss > 0:
        print(f"\n[6] Applying ONS LSOA 2021->2011 crosswalk to recover {n_miss:,} stops...")
        xwalk = load_lsoa_crosswalk()

        # remap the raw LSOA code; only touch unmatched rows to leave matched ones alone
        mask = london[first_feat].isna()
        remapped = apply_crosswalk(london.loc[mask, "pc_lsoa"], ai23_codes, xwalk)
        london.loc[mask, "pc_lsoa"] = remapped

        # re-merge only the previously unmatched rows
        unmatched_df = london[mask].drop(
            columns=[f[2] for f in AI23_FEATURES] + ["geo_code"], errors="ignore"
        )
        fixed = unmatched_df.merge(ai23, left_on="pc_lsoa", right_on="geo_code", how="left")
        london = pd.concat([london[~mask], fixed], ignore_index=True)

        n_still = london[first_feat].isna().sum()
        recovered = n_miss - n_still
        print(f"    Recovered : {recovered:,}  |  still unmatched : {n_still:,}")
        if n_still > 0:
            sample = london.loc[london[first_feat].isna(), "pc_lsoa"].dropna().unique()[:5]
            print(f"    Remaining unmatched LSOA codes (sample): {list(sample)}")
            print(f"    -> These will be dropped and noted as a data limitation.")

    # — 5g. Final drop of unresolvable stops
    n_drop = london[first_feat].isna().sum()
    london = london.dropna(subset=[first_feat]).copy()
    london = london.rename(columns={"pc_lsoa": "lsoa11cd"})

    # — 5h. Summary statistics
    feat_cols = [f[2] for f in AI23_FEATURES]
    print(f"\n[7] Feature summary ({len(london):,} London stops with full data):")
    print(london[feat_cols].describe().round(1).to_string())

    # — 5i. Save
    out_cols = (
        ["STOPCODE", "stop_name", "total_boardings", "lat", "lon",
         "POSTCODE", "lsoa11cd", "lad_code", "lad_name"]
        + feat_cols
    )
    out_cols = [c for c in out_cols if c in london.columns]
    london[out_cols].to_csv("stops_features.csv", index=False)

    print(f"\n{'='*60}")
    print(f"SAVED: stops_features.csv")
    print(f"  Rows    : {len(london):,} London bus stops")
    print(f"  Columns : {len(out_cols)}  ({', '.join(feat_cols)})")
    print(f"  Dropped : {n_drop:,} stops with unresolvable LSOA (data limitation)")
    print(f"{'='*60}")
    print("\nNEXT (step 4): build multigraph, leave-borough-out spatial CV, "
          "train Historical Average / RF / GATv2.")


if __name__ == "__main__":
    main()
