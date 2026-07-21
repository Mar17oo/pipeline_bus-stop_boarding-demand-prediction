"""
STEP 5B — V/C (Volume/Capacity) target experiment
==================================================
Reframes the cold-start problem as OVERCROWDING PREDICTION.

Instead of predicting raw boarding counts, predict the PEAK V/C RATIO at each
stop — the worst-case occupancy across all services. This is more policy-relevant:
TfL uses V/C to allocate vehicle capacity for new services.

Target: peak_vc = 90th-percentile V/C across all (route × direction × QH)
        combinations at the stop. Captures the "busy moment" rather than
        the noisiest single peak, which could be a data artefact.

        For reference: TfL crowding threshold ≈ 0.85 V/C (seated + standing).

Output:
  stops_features_vc.csv        — features + peak_vc target
  results_cv_vc_ai23_osm.csv   — per-fold results (WMAPE on V/C)
  results_summary_vc_ai23_osm.csv

Run time: ~2–3 hours (33-fold CV, same as main experiment)
"""

import sys, time, warnings, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import BallTree
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import subgraph as pyg_subgraph

QUICK = "--quick" in sys.argv

def log(msg): print(msg, flush=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURES_CSV = "stops_features_osm.csv"
BUSTO_FOLDER = "data"
BOROUGH_COL  = "lad_name"
TARGET_COL   = "peak_vc"        # new target — 90th pct V/C per stop

FEAT_COLS  = [
    "employment_all_30min", "hospitals_30min", "gp_30min",
    "supermarkets_30min",   "pharmacies_30min",
    "primary_schools_30min","secondary_schools_30min","main_bua_30min",
]
OSM_COLS   = ["poi_residential","poi_shopping","poi_company",
              "poi_education","poi_entertainment","poi_scenic"]
COORD_COLS = ["lat","lon"]

K_NEIGHBORS = 5
HIDDEN_DIM  = 64
HEADS       = 4
DROPOUT     = 0.15
LR          = 5e-4
EPOCHS      = 1000
PATIENCE    = 40
VAL_FRAC    = 0.1
RF_TREES    = 300
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── 1. BUILD PEAK V/C TARGET ──────────────────────────────────────────────────
log("Building peak V/C target from BUSTO CSVs...")
files = glob.glob(f"{BUSTO_FOLDER}/*Weekday*QUARTER HOUR*.csv")
frames = [pd.read_csv(f, dtype={"STOPCODE":"string"},
                      usecols=["STOPCODE","V/C"])
          for f in files]
raw = pd.concat(frames, ignore_index=True)
raw["STOPCODE"] = raw["STOPCODE"].str.strip().str.upper()
raw["V/C"] = pd.to_numeric(raw["V/C"], errors="coerce").fillna(0)

# 90th-percentile V/C per stop — robust peak that avoids single-observation spikes
vc_agg = (raw.groupby("STOPCODE")["V/C"]
             .quantile(0.90)
             .reset_index()
             .rename(columns={"V/C": TARGET_COL}))
log(f"  {len(vc_agg):,} stops with V/C data")
log(f"  peak_vc: mean={vc_agg[TARGET_COL].mean():.3f}  "
    f"max={vc_agg[TARGET_COL].max():.3f}  "
    f"pct>0.85={100*(vc_agg[TARGET_COL]>0.85).mean():.1f}%")

# ── 2. LOAD FEATURES + MERGE TARGET ──────────────────────────────────────────
log("\nLoading features...")
df = pd.read_csv(FEATURES_CSV, dtype={"STOPCODE":"string"})
df["STOPCODE"] = df["STOPCODE"].str.strip().str.upper()
df = df.merge(vc_agg, on="STOPCODE", how="inner")
log(f"  {len(df):,} stops after merge (dropped stops with no V/C data)")

df.to_csv("stops_features_vc.csv", index=False)
log("  Saved -> stops_features_vc.csv")

# ── 3. FEATURES ───────────────────────────────────────────────────────────────
osm_present = [c for c in OSM_COLS if c in df.columns]
ACTIVE_FEAT = FEAT_COLS + osm_present    # AI23 + OSM
ALL_COLS    = ACTIVE_FEAT + COORD_COLS

def prep_features(df_):
    X = np.zeros((len(df_), len(ALL_COLS)), dtype=np.float32)
    for i, c in enumerate(ACTIVE_FEAT):
        X[:, i] = np.log1p(df_[c].values)
    X[:, len(ACTIVE_FEAT)]     = df_["lat"].values
    X[:, len(ACTIVE_FEAT) + 1] = df_["lon"].values
    return X

# ── 4. METRICS ────────────────────────────────────────────────────────────────
def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true-y_pred)) / (np.sum(np.abs(y_true))+1e-8))

def score(y_true, y_pred, name):
    y_pred = np.clip(y_pred, 0, None)
    return {
        "model": name,
        "WMAPE": round(wmape(y_true, y_pred), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 4),
    }

# ── 5. GRAPH BUILDERS ─────────────────────────────────────────────────────────
def build_knn(lats, lons, k=K_NEIGHBORS):
    coords = np.radians(np.column_stack([lats, lons]))
    _, nbrs = BallTree(coords, metric="haversine").query(coords, k=k+1)
    src, dst = [], []
    for i, row in enumerate(nbrs):
        for j in row[1:]: src+=[i,j]; dst+=[j,i]
    return torch.unique(torch.tensor([src,dst],dtype=torch.long), dim=1)

def build_route_edges(df_stops):
    files2 = glob.glob(f"{BUSTO_FOLDER}/*Weekday*QUARTER HOUR*.csv")
    fr = [pd.read_csv(f, dtype={"STOPCODE":"string"},
                      usecols=["ROUTE","DIRECTION","STOPSEQUENCE","STOPCODE"])
          for f in files2]
    raw2 = pd.concat(fr, ignore_index=True)
    code2idx = {c:i for i,c in enumerate(
        df_stops["STOPCODE"].astype(str).str.strip().str.upper())}
    raw2["STOPCODE"] = raw2["STOPCODE"].str.strip().str.upper()
    raw2 = raw2[raw2["STOPCODE"].isin(code2idx)].copy()
    raw2["node_idx"] = raw2["STOPCODE"].map(code2idx)
    src,dst = [],[]
    for _,grp in raw2.groupby(["ROUTE","DIRECTION"]):
        idxs = (grp.drop_duplicates("STOPCODE")
                   .sort_values("STOPSEQUENCE")["node_idx"].tolist())
        for i in range(len(idxs)-1):
            src+=[idxs[i],idxs[i+1]]; dst+=[idxs[i+1],idxs[i]]
    return torch.unique(torch.tensor([src,dst],dtype=torch.long), dim=1)

# ── 6. MODELS ─────────────────────────────────────────────────────────────────
class GATv2Model(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1   = GATv2Conv(in_ch, HIDDEN_DIM, heads=HEADS,
                              dropout=DROPOUT, concat=True)
        self.c2   = GATv2Conv(HIDDEN_DIM*HEADS, 1, heads=1,
                              dropout=DROPOUT, concat=False)
        self.skip = torch.nn.Linear(in_ch, HIDDEN_DIM*HEADS, bias=False)
    def forward(self, x, ei):
        h = F.elu(self.c1(x, ei)) + self.skip(x)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.c2(h, ei).squeeze(-1)

class MLPModel(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        h = HIDDEN_DIM*HEADS
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_ch,h), torch.nn.ELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(h,h//2), torch.nn.ELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(h//2,1),
        )
    def forward(self, x, ei=None): return self.net(x).squeeze(-1)

def train_nn(model, x, y, ei, tr_pos, val_pos):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, best_w, no_imp = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out  = model(x, ei)
        loss = F.mse_loss(out[tr_pos], y[tr_pos])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(x, ei)[val_pos], y[val_pos]).item()
        if vl < best_val - 1e-5:
            best_val = vl
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    model.load_state_dict(best_w)
    return model

# ── 7. CROSS-VALIDATION ───────────────────────────────────────────────────────
log("\nBuilding multigraph...")
X_raw = prep_features(df)
knn_ei   = build_knn(df["lat"].values, df["lon"].values)
route_ei = build_route_edges(df)
ei_all   = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
log(f"  {ei_all.shape[1]:,} directed edges, {len(df):,} nodes")

log(f"\nFeatures: {len(ALL_COLS)} ({len(ACTIVE_FEAT)} AI23+OSM + 2 coords) [V/C target]")

# Pre-warm
_dummy = GATv2Model(len(ALL_COLS))
with torch.no_grad(): _dummy(torch.zeros(5, len(ALL_COLS)),
                             torch.zeros(2,4,dtype=torch.long))
log("Pre-warm done.\n")

boroughs = sorted(df[BOROUGH_COL].unique())
if QUICK: boroughs = boroughs[:5]; log("[QUICK] 5 folds only\n")

records, t0 = [], time.time()
for bi, boro in enumerate(boroughs, 1):
    mask_te = df[BOROUGH_COL] == boro
    mask_tr = ~mask_te
    idx_all = np.arange(len(df))
    tr_idx  = idx_all[mask_tr.values]
    te_idx  = idx_all[mask_te.values]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)
    X_sc_tr = X_sc[tr_idx]
    X_sc_te = X_sc[te_idx]

    y_all = df[TARGET_COL].values.astype(np.float32)
    y_tr  = y_all[tr_idx]
    y_te  = y_all[te_idx]

    val_n   = max(1, int(len(tr_idx)*VAL_FRAC))
    rng     = np.random.default_rng(SEED)
    val_pos = rng.choice(len(tr_idx), val_n, replace=False)
    tr_pos  = np.setdiff1d(np.arange(len(tr_idx)), val_pos)

    # HistAvg
    ha_pred = np.full(len(te_idx), y_tr.mean())

    # RF
    rf = RandomForestRegressor(RF_TREES, random_state=SEED, n_jobs=-1)
    rf.fit(X_sc_tr[tr_pos], y_tr[tr_pos])
    rf_pred = rf.predict(X_sc_te)

    # GATv2 + MLP
    ei_sub, nmap = pyg_subgraph(
        torch.tensor(tr_idx, dtype=torch.long), ei_all,
        relabel_nodes=True, num_nodes=len(df))
    ei_full = ei_all

    x_t  = torch.tensor(X_sc, dtype=torch.float32)
    y_t  = torch.tensor(y_all, dtype=torch.float32)
    tr_t = torch.tensor(tr_idx, dtype=torch.long)
    te_t = torch.tensor(te_idx, dtype=torch.long)

    tr_pos_t  = torch.tensor(tr_idx[tr_pos],  dtype=torch.long)
    val_pos_t = torch.tensor(tr_idx[val_pos], dtype=torch.long)

    gatv2 = train_nn(GATv2Model(len(ALL_COLS)), x_t, y_t, ei_full,
                     tr_pos_t, val_pos_t)
    mlp   = train_nn(MLPModel(len(ALL_COLS)),   x_t, y_t, ei_full,
                     tr_pos_t, val_pos_t)

    gatv2.eval(); mlp.eval()
    with torch.no_grad():
        gat_pred = gatv2(x_t, ei_full)[te_t].numpy()
        mlp_pred = mlp(x_t, ei_full)[te_t].numpy()

    fold_records = []
    for name, pred in [("HistAvg",ha_pred),("RF",rf_pred),
                        ("MLP",mlp_pred),("GATv2",gat_pred)]:
        r = score(y_te, pred, name)
        r["borough"] = boro; r["n_test"] = len(te_idx)
        fold_records.append(r)
        records.append(r)

    t_fold = time.time() - t0
    row = {r["model"]: r["WMAPE"] for r in fold_records}
    log(f"  [{bi:2d}/{len(boroughs)}] {boro:<35} n={len(te_idx):4d}  "
        f"HA={row['HistAvg']:.3f}  RF={row['RF']:.3f}  "
        f"MLP={row['MLP']:.3f}  GATv2={row['GATv2']:.3f}  "
        f"({t_fold/60:.0f}m)")

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
cv_df = pd.DataFrame(records)[["borough","n_test","model","WMAPE","RMSE","MAE"]]
cv_df.to_csv("results_cv_vc_ai23_osm.csv", index=False)

summary = (cv_df.groupby("model")[["WMAPE","RMSE","MAE"]]
           .agg(["mean","std","median"])
           .round(4))
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv("results_summary_vc_ai23_osm.csv", index=False)

total = (time.time()-t0)/60
log(f"\nTotal time: {total:.1f} min")
log("Saved -> results_cv_vc_ai23_osm.csv  |  results_summary_vc_ai23_osm.csv")

log("\n" + "="*70)
log(f"RESULTS — V/C TARGET (90th-pct peak occupancy)  [{len(boroughs)} folds]")
log("="*70)
for _, row in summary.iterrows():
    log(f"  {row['model']:<8} WMAPE={row['WMAPE_mean']:.4f}"
        f"(±{row['WMAPE_std']:.4f})  med={row['WMAPE_median']:.4f}"
        f"  RMSE={row['RMSE_mean']:.4f}  MAE={row['MAE_mean']:.4f}")
log("="*70)
