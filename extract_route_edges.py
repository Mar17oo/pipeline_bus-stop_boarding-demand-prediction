"""
Extract route edges from BUSTO CSVs -> route_edges.csv
Run once locally. Upload route_edges.csv to Colab alongside stops_features.csv.
"""
import glob
import pandas as pd

STOPS_FEATURES = "stops_features.csv"
DATA_FOLDER    = "data"
OUTPUT         = "route_edges.csv"

df_stops = pd.read_csv(STOPS_FEATURES, dtype={"STOPCODE": "string"})
code_to_idx = {
    c: i for i, c in enumerate(
        df_stops["STOPCODE"].astype(str).str.strip().str.upper()
    )
}

files = glob.glob(f"{DATA_FOLDER}/*Weekday*QUARTER HOUR*.csv")
print(f"Reading {len(files)} BUSTO files...")

frames = []
for f in files:
    frames.append(
        pd.read_csv(f, dtype={"STOPCODE": "string"},
                    usecols=["ROUTE", "DIRECTION", "STOPSEQUENCE", "STOPCODE"])
    )
raw = pd.concat(frames, ignore_index=True)
raw["STOPCODE"] = raw["STOPCODE"].str.strip().str.upper()
raw = raw[raw["STOPCODE"].isin(code_to_idx)].copy()
raw["node_idx"] = raw["STOPCODE"].map(code_to_idx)

src_list, dst_list = [], []
for _, grp in raw.groupby(["ROUTE", "DIRECTION"]):
    idxs = (
        grp.drop_duplicates("STOPCODE")
           .sort_values("STOPSEQUENCE")["node_idx"]
           .tolist()
    )
    for i in range(len(idxs) - 1):
        src_list += [idxs[i], idxs[i + 1]]
        dst_list += [idxs[i + 1], idxs[i]]

edges = pd.DataFrame({"src": src_list, "dst": dst_list}).drop_duplicates()
edges.to_csv(OUTPUT, index=False)
print(f"Saved {len(edges):,} directed route edges -> {OUTPUT}")
