"""
============================================================================
STEP 1 (REAL DATA) — Aggregate BUSTO to stop-level boardings 
============================================================================
Run this on 4 downloaded BUSTO CSVs. No coordinates needed yet.
This produces your TARGET table: one row per stop, total weekday boardings.

After this, we are getting coordinates per STOPCODE
(from the "Bus stop locations and routes" file) to join accessibility features.
============================================================================
"""

import glob
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Point this at the folder containing your 4 BUSTO weekday CSVs.
#    The filenames contain route ranges (1-149, 150-299, 300-549, letter/N).
# ---------------------------------------------------------------------------
CSV_FOLDER = "data"   # <-- change to your data folder, e.g. "data/busto/"
PATTERN = "*Weekday*QUARTER HOUR*.csv"   

files = glob.glob(f"{CSV_FOLDER}/{PATTERN}")
print(f"Found {len(files)} files:")
for f in files:
    print("  ", f)

# ---------------------------------------------------------------------------
# 2. Load and concatenate all 4 files into one big table.
# ---------------------------------------------------------------------------
frames = []
for f in files:
    # STOPCODE is alphanumeric (e.g. BP2636) -> read as string, not number
    df = pd.read_csv(f, dtype={"STOPCODE": "string"})
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows loaded: {len(raw):,}")
print("Columns:", list(raw.columns))

# ---------------------------------------------------------------------------
# 3. AGGREGATE to stop level.
#    Each STOPCODE appears many times (every route x direction x quarter-hour).
#    We SUM Boardings across all of them -> total typical-weekday boardings.
#    (We keep one STOPNAME per stop for readability.)
# ---------------------------------------------------------------------------
stops = (
    raw.groupby("STOPCODE")
       .agg(
           stop_name=("STOPNAME", "first"),
           total_boardings=("Boardings", "sum"),
           n_route_dir_qhr_rows=("Boardings", "size"),  # how many rows contributed
       )
       .reset_index()
)

# ---------------------------------------------------------------------------
# 4. Quick sanity checks (always inspect real data!)
# ---------------------------------------------------------------------------
print(f"\nUnique stops: {len(stops):,}")
print(f"Boardings — min: {stops['total_boardings'].min():.2f}  "
      f"median: {stops['total_boardings'].median():.2f}  "
      f"max: {stops['total_boardings'].max():.2f}")
print(f"Stops with zero total boardings: {(stops['total_boardings'] == 0).sum()}")
print("\nBusiest 5 stops:")
print(stops.nlargest(5, "total_boardings")[["STOPCODE", "stop_name", "total_boardings"]]
      .to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Save the stop-level target table.
# ---------------------------------------------------------------------------
stops.to_csv("busto_stop_level_boardings.csv", index=False)
print("\nSaved -> busto_stop_level_boardings.csv")
print("\nNEXT: get coordinates per STOPCODE from the 'Bus stop locations and "
      "routes' file, then point-in-polygon into LSOA to join AI23 features.")
