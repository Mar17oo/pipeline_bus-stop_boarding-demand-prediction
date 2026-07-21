"""
STEP 4 — Leave-borough-out spatial CV: HistAvg, RF, MLP (ablation), GATv2
==========================================================================
Run with:  python -u step4_model.py               (full 33-fold, AI23+OSM if available)
           python -u step4_model.py --quick        (5 folds only, for testing)
           python -u step4_model.py --osm-only     (6 OSM + lat/lon only, no AI23)
           python -u step4_model.py --quick --osm-only

TRUE INDUCTIVE SETTING:
  Train fold: GATv2 trained on subgraph of training stops only.
  Test  fold: test stops added to graph; they aggregate from training
              neighbours. Model never sees test labels. (Hamilton et al. 2017)

MULTIGRAPH (two edge types combined):
  (1) Geographic KNN  — K=5 nearest stops by Haversine distance
  (2) Route connectivity — consecutive stops sharing the same bus route/direction
      from BUSTO CSVs. This is the functionally meaningful edge: stops on the
      same route have correlated demand driven by the same passenger flow.
Loss:  MSE on log1p(boardings). Metrics reported on original scale.
==========================================================================
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import BallTree
from xgboost import XGBRegressor
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import subgraph as pyg_subgraph

QUICK     = "--quick"     in sys.argv
OSM_ONLY  = "--osm-only"  in sys.argv   # 6 OSM + lat/lon only (no AI23)
AI23_ONLY = "--ai23-only" in sys.argv   # force AI23 + lat/lon (ignore OSM file)
WITH_SC   = "--with-sc"   in sys.argv   # add service_coverage (step3d) to any config
K10       = "--k10"       in sys.argv   # GNN-fairness check: K=10 instead of K=5
PCA_AI23  = "--pca-ai23"  in sys.argv   # collapse the 8 AI23 features to 3 PCA
                                         # components (VIF up to 37.7 pairwise —
                                         # see experiment_log.md Experiment Set 6)
FUNC_SIM  = "--func-sim"  in sys.argv   # add functional-similarity edges (Zheng
                                         # et al. 2025-style: Pearson corr. of OSM
                                         # POI vectors) as a 3rd edge type in the
                                         # existing single multigraph — Experiment Set 7

def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
import os as _os
# Use OSM-enriched features if step3b has been run; fall back to base features.
# Override with --ai23-only to force the base AI23 feature set.
if AI23_ONLY:
    DATA_FILE = "stops_features.csv"
elif _os.path.exists("stops_features_osm.csv"):
    DATA_FILE = "stops_features_osm.csv"
else:
    DATA_FILE = "stops_features.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"

# Accessibility features (LSOA-level from AI23).
# All have skew > 1.7 so we apply log1p before scaling (see prep_features()).
FEAT_COLS   = [
    "employment_all_30min", "hospitals_30min", "gp_30min",
    "supermarkets_30min",   "pharmacies_30min",
    "primary_schools_30min","secondary_schools_30min",
    "main_bua_30min",
]
# OSM POI features added by step3b (matches Zheng et al. 2025 POI categories).
OSM_COLS = ["poi_residential", "poi_shopping", "poi_company",
            "poi_education", "poi_entertainment", "poi_scenic"]
# Service coverage added by step3d: n distinct (route×direction×QH) combinations
# serving each stop. Valid cold-start feature — derived from planned timetable.
SC_COL = "service_coverage"
# lat/lon added as stop-level features so each node is spatially unique.
# Without these, 96% of stops share their LSOA with ≥1 neighbour, giving
# identical feature vectors — message passing then adds zero information.
COORD_COLS  = ["lat", "lon"]
ALL_FEAT_COLS = FEAT_COLS + COORD_COLS   # extended dynamically in prep_features

K_NEIGHBORS = 10 if K10 else 5   # --k10: GNN-fairness check (more aggregation context)
HIDDEN_DIM  = 64    # matches Zheng et al. (2025) spatial attention capacity
HEADS       = 4     # matches standard GAT literature (Velickovic 2018, Brody 2022)
DROPOUT     = 0.15
LR          = 5e-4
EPOCHS      = 500   # safe ceiling — early stopping triggers well before this
PATIENCE    = 20    # checks, evaluated every VAL_EVERY epochs
VAL_EVERY   = 5     # evaluate val loss every N epochs (saves ~40% NN compute)
VAL_FRAC    = 0.1

RF_TREES    = 150   # diminishing returns past ~100 trees for this dataset size
N_PCA_COMPONENTS = 3   # captures 94.4% of variance across the 8 AI23 features
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------------------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------------------------
def prep_features(df):
    """
    Returns raw feature matrix before any scaling.
    --osm-only : 6 OSM POI counts + lat/lon  (8 features, matches Zheng feature type)
    default    : 8 AI23 + OSM if present + lat/lon
    All non-coord features are log1p-transformed (heavy right tail).
    """
    osm_present = [c for c in OSM_COLS if c in df.columns]
    sc_present  = [SC_COL] if (WITH_SC and SC_COL in df.columns) else []

    if OSM_ONLY and osm_present:
        active_feat = osm_present + sc_present          # OSM [+ SC], no AI23
    elif AI23_ONLY:
        active_feat = FEAT_COLS + sc_present            # AI23 [+ SC], no OSM
    else:
        active_feat = FEAT_COLS + osm_present + sc_present  # AI23 + OSM [+ SC]

    active_all = active_feat + COORD_COLS
    global ALL_FEAT_COLS, AI23_IDX
    ALL_FEAT_COLS = active_all
    # Indices of the 8 raw AI23 columns within X, if present — used by
    # run_fold() to apply per-fold PCA reduction when --pca-ai23 is set.
    AI23_IDX = [active_feat.index(c) for c in FEAT_COLS if c in active_feat]

    X = np.zeros((len(df), len(active_all)), dtype=np.float32)
    for i, c in enumerate(active_feat):
        X[:, i] = np.log1p(df[c].values)
    X[:, len(active_feat)]     = df["lat"].values
    X[:, len(active_feat) + 1] = df["lon"].values
    return X


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))

def score(y_true, y_pred, name):
    y_pred = np.clip(y_pred, 0, None)
    return {
        "model": name,
        "WMAPE": round(wmape(y_true, y_pred), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 3),
    }


# ---------------------------------------------------------------------------
# KNN GRAPH
# ---------------------------------------------------------------------------
def idw_predict(train_lat, train_lon, train_y_log, test_lat, test_lon, k=20, power=2):
    """Inverse Distance Weighting — classical geostatistical cold-start baseline
    (spatial interpolation per Liu et al. 2017's NYC Citi Bike cold-start work:
    gravity models + natural-neighbor interpolation / kriging). Predicts
    log1p(boardings) as a distance-weighted average of the k nearest TRAINING
    stops' log1p(boardings) — uses ONLY lat/lon, no AI23/OSM/SC features.
    Tests whether GATv2's neighbour-aggregation adds anything beyond what
    plain spatial interpolation already gives for free.
    """
    train_coords = np.radians(np.column_stack([train_lat, train_lon]))
    test_coords  = np.radians(np.column_stack([test_lat, test_lon]))
    k_use = min(k, len(train_lat))
    dist, idx = BallTree(train_coords, metric="haversine").query(test_coords, k=k_use)
    dist_km = dist * 6371.0088   # Earth radius
    w = 1.0 / np.power(dist_km + 1e-6, power)
    neighbor_y = train_y_log[idx]                      # (n_test, k)
    return np.sum(w * neighbor_y, axis=1) / np.sum(w, axis=1)


def build_knn_edge_index(lats, lons, k=K_NEIGHBORS):
    coords = np.radians(np.column_stack([lats, lons]))
    _, nbrs = BallTree(coords, metric="haversine").query(coords, k=k + 1)
    src, dst = [], []
    for i, row in enumerate(nbrs):
        for j in row[1:]:
            src += [i, j]; dst += [j, i]   # bidirectional
    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


# ---------------------------------------------------------------------------
# ROUTE CONNECTIVITY GRAPH
# ---------------------------------------------------------------------------
def build_route_edges(df_stops, data_folder="data"):
    """
    Connect consecutive stops that share the same bus route + direction.
    Source: BUSTO quarter-hour CSVs (already in data/).
    Rationale: stops on the same route carry the same passenger flow,
    so their demand is correlated in a way pure geography cannot capture.
    """
    import glob
    files = glob.glob(f"{data_folder}/*Weekday*QUARTER HOUR*.csv")
    frames = [pd.read_csv(f, dtype={"STOPCODE": "string"},
                          usecols=["ROUTE", "DIRECTION", "STOPSEQUENCE", "STOPCODE"])
              for f in files]
    raw = pd.concat(frames, ignore_index=True)

    # Map STOPCODE -> integer node index matching stops_features.csv row order
    code_to_idx = {
        c: i for i, c in enumerate(
            df_stops["STOPCODE"].astype(str).str.strip().str.upper()
        )
    }
    raw["STOPCODE"] = raw["STOPCODE"].str.strip().str.upper()
    raw = raw[raw["STOPCODE"].isin(code_to_idx)].copy()
    raw["node_idx"] = raw["STOPCODE"].map(code_to_idx)

    src, dst = [], []
    for _, grp in raw.groupby(["ROUTE", "DIRECTION"]):
        idxs = (grp.drop_duplicates("STOPCODE")
                   .sort_values("STOPSEQUENCE")["node_idx"]
                   .tolist())
        for i in range(len(idxs) - 1):
            src += [idxs[i], idxs[i + 1]]   # bidirectional
            dst += [idxs[i + 1], idxs[i]]

    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


# ---------------------------------------------------------------------------
# FUNCTIONAL SIMILARITY GRAPH (Zheng et al. 2025-style)
# ---------------------------------------------------------------------------
def build_functional_similarity_edges(df_stops, osm_cols, top_k=10, thresh=0.99):
    """
    Zheng et al. (2025) build a functional-similarity graph G2 by taking
    Pearson correlation of each stop's POI count vector (within a 500m
    buffer) and connecting pairs with rho > 0.8.

    On this dataset (6 OSM POI categories, heavy zero-inflation), rho>0.8
    produces a near-complete graph (48M directed edges, avg degree 2,680) --
    not sparse "local structure" but something closer to global averaging,
    and computationally infeasible to train on. This is a genuine property
    of the data at this scale/category-count, not a bug (see
    experiment_log.md Experiment Set 7 for the density sweep). We therefore
    sparsify to rho>0.99 (still purely correlation-threshold-based, matching
    the paper's mechanism, just recalibrated for this dataset) AND cap each
    stop to its top_k=10 highest-correlation neighbours (mirrors the K=5
    geographic KNN graph's sparsity) so a handful of near-duplicate POI
    profiles (e.g. two stops with identical all-zero-but-one profiles) can't
    each pull in thousands of edges.

    Correlation is computed over log1p(POI counts), consistent with how
    these features are used elsewhere in this pipeline. Stops with a
    zero-variance POI vector (all 6 counts identical, e.g. all-zero) have no
    well-defined correlation with anything and get zero functional-sim edges
    (they still have KNN + route edges).
    """
    P = np.log1p(df_stops[osm_cols].values).astype(np.float32)
    n = P.shape[0]
    mean = P.mean(axis=1, keepdims=True)
    Pc = P - mean
    norm = np.linalg.norm(Pc, axis=1, keepdims=True)
    zero_var = (norm[:, 0] == 0)
    norm_safe = np.where(norm == 0, 1.0, norm)
    U = (Pc / norm_safe).astype(np.float32)
    U[zero_var] = 0.0   # zero-variance rows -> zero vector -> correlation 0 with everyone

    src, dst = [], []
    CHUNK = 2000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        block = U[start:end] @ U.T          # (chunk, n) correlations
        for local_i, global_i in enumerate(range(start, end)):
            block[local_i, global_i] = -1.0  # exclude self
        # top_k highest-correlation neighbours per row, then apply threshold
        k = min(top_k, n - 1)
        top_idx = np.argpartition(-block, k, axis=1)[:, :k]
        for local_i, global_i in enumerate(range(start, end)):
            for j in top_idx[local_i]:
                if block[local_i, j] > thresh:
                    src += [global_i, j]
                    dst += [j, global_i]

    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
class GATv2Model(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1   = GATv2Conv(in_ch, HIDDEN_DIM, heads=HEADS,
                              dropout=DROPOUT, concat=True)
        self.c2   = GATv2Conv(HIDDEN_DIM * HEADS, 1, heads=1,
                              dropout=DROPOUT, concat=False)
        # Skip connection: lets model weight graph vs direct feature path.
        # Critical for cold-start where cross-borough neighbours add noise.
        self.skip = torch.nn.Linear(in_ch, HIDDEN_DIM * HEADS, bias=False)

    def forward(self, x, edge_index):
        h = F.elu(self.c1(x, edge_index)) + self.skip(x)   # graph + residual
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.c2(h, edge_index).squeeze(-1)


class MLPModel(torch.nn.Module):
    """Ablation: same capacity as GATv2 but zero message passing."""
    def __init__(self, in_ch):
        super().__init__()
        h = HIDDEN_DIM * HEADS
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_ch, h), torch.nn.ELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(h, h // 2), torch.nn.ELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(h // 2, 1),
        )

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# TRAINING (early stopping on internal val split)
# ---------------------------------------------------------------------------
def train_nn(model, x, y, ei, tr_pos, val_pos):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out  = model(x, ei)
        # Huber loss: MSE for small errors, L1 for large — more robust than MSE
        # on log1p(boardings) which still has residual heavy tail after transform.
        loss = F.huber_loss(out[tr_pos], y[tr_pos], delta=0.5)
        loss.backward(); opt.step()
        # Only check val every VAL_EVERY epochs — saves ~40% of NN compute.
        if (epoch + 1) % VAL_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            vl = F.huber_loss(model(x, ei)[val_pos], y[val_pos], delta=0.5).item()
        if vl < best_val - 1e-6:
            best_val = vl
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_w:
        model.load_state_dict(best_w)
    return model


# ---------------------------------------------------------------------------
# ONE FOLD
# ---------------------------------------------------------------------------
def run_fold(borough, df, X_raw, y_orig, full_ei, fi, nf):
    t0 = time.time()
    test_mask  = (df[BOROUGH_COL] == borough).values
    train_mask = ~test_mask
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    y_tr, y_te = y_orig[tr_idx], y_orig[te_idx]

    X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]

    # --pca-ai23: collapse the 8 (correlated, VIF up to 37.7) AI23 columns to
    # N_PCA_COMPONENTS decorrelated components. Both the pre-PCA standardizer
    # and the PCA itself are fit on TRAINING stops only, per fold — same
    # no-leakage rule as the main scaler below. Non-AI23 columns (OSM, SC,
    # lat/lon) pass through unchanged.
    if PCA_AI23 and AI23_IDX:
        other_idx = [i for i in range(X_raw.shape[1]) if i not in AI23_IDX]
        ai23_pre = StandardScaler()
        ai23_tr_std = ai23_pre.fit_transform(X_tr_raw[:, AI23_IDX])
        ai23_te_std = ai23_pre.transform(X_te_raw[:, AI23_IDX])
        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
        ai23_tr_pca = pca.fit_transform(ai23_tr_std)
        ai23_te_pca = pca.transform(ai23_te_std)
        X_tr_raw = np.hstack([ai23_tr_pca, X_tr_raw[:, other_idx]])
        X_te_raw = np.hstack([ai23_te_pca, X_te_raw[:, other_idx]])

    # Fix: fit scaler on TRAINING stops only — prevents test-set leakage.
    # (Fitting globally shifts the mean by up to 4.4% for extreme boroughs
    # like City of London, which have very high employment_all_30min values.)
    scaler = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_tr_raw)  # fit on training stops only
    X_sc_te = scaler.transform(X_te_raw)       # transform test with train params

    # 1. Historical Average
    ha = score(y_te, np.full(len(te_idx), y_tr.mean()), "HistAvg")

    # 1b. IDW spatial interpolation (Liu et al. 2017 cold-start baseline; lat/lon only)
    idw_pred = np.expm1(idw_predict(df["lat"].values[tr_idx], df["lon"].values[tr_idx],
                                     np.log1p(y_tr),
                                     df["lat"].values[te_idx], df["lon"].values[te_idx]))
    idw_s = score(y_te, idw_pred, "IDW")

    # 2. Multiple Linear Regression (Ridge, alpha=1.0) — classic direct demand model
    #    (Gutierrez et al. 2011; Lin et al. 2023 found MLR competitive with MLP).
    mlr = Ridge(alpha=1.0, random_state=SEED)
    mlr.fit(X_sc_tr, np.log1p(y_tr))
    mlr_s = score(y_te, np.expm1(mlr.predict(X_sc_te)), "MLR")

    # 3. Random Forest (uses train-only scaled features)
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_features="sqrt",
                               min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    rf.fit(X_sc_tr, np.log1p(y_tr))
    rf_s = score(y_te, np.expm1(rf.predict(X_sc_te)), "RF")

    # 4. XGBoost (Yusuf et al. 2025 found gradient boosting best-in-class for
    #    stop-level transit prediction) — sensible defaults, no heavy tuning.
    xgbr = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                        random_state=SEED, n_jobs=-1, verbosity=0)
    xgbr.fit(X_sc_tr, np.log1p(y_tr))
    xgb_s = score(y_te, np.expm1(xgbr.predict(X_sc_te)), "XGBoost")

    # Stratified val split: sample evenly across 5 demand quantiles so the
    # early-stopping signal reflects the full boarding distribution, not just
    # the majority of low-demand stops that dominate a random draw.
    rng   = np.random.RandomState(SEED + fi)
    n_val = max(5, int(len(tr_idx) * VAL_FRAC))
    log_y_tr = np.log1p(y_tr)
    quantile_labels = pd.qcut(log_y_tr, q=5, labels=False, duplicates="drop")
    vp_list = []
    n_per_q = max(1, n_val // 5)
    for q in range(5):
        q_idx = np.where(quantile_labels == q)[0]
        if len(q_idx) > 0:
            chosen = rng.choice(q_idx, min(n_per_q, len(q_idx)), replace=False)
            vp_list.extend(chosen.tolist())
    vp = np.array(vp_list)
    tp = np.setdiff1d(np.arange(len(tr_idx)), vp)
    tp_t, vp_t = torch.tensor(tp, dtype=torch.long), torch.tensor(vp, dtype=torch.long)

    # Build per-fold scaled feature matrices (train-fit scaler only)
    all_idx  = np.concatenate([tr_idx, te_idx])
    X_sc_all = np.vstack([X_sc_tr, X_sc_te])   # [train rows | test rows]

    x_tr  = torch.tensor(X_sc_tr, dtype=torch.float)
    x_ctx = torch.tensor(X_sc_all, dtype=torch.float)
    y_tr_t = torch.tensor(np.log1p(y_tr), dtype=torch.float)
    n_tr   = len(tr_idx)

    # Train subgraph (train-train edges only, re-indexed)
    tr_t    = torch.tensor(tr_idx, dtype=torch.long)
    ei_tr,_ = pyg_subgraph(tr_t, full_ei, relabel_nodes=True, num_nodes=len(df))

    # Context graph: train + test for inductive inference (test aggregates from train)
    all_t    = torch.tensor(all_idx, dtype=torch.long)
    ei_ctx,_ = pyg_subgraph(all_t, full_ei, relabel_nodes=True, num_nodes=len(df))

    n_in = X_sc_tr.shape[1]   # actual feature count (differs from len(ALL_FEAT_COLS) under --pca-ai23)

    # 3. MLP ablation
    mlp = train_nn(MLPModel(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    mlp.eval()
    with torch.no_grad():
        mlp_pred = np.expm1(mlp(x_ctx)[n_tr:].numpy())
    mlp_s = score(y_te, mlp_pred, "MLP")

    # 4. GATv2
    log_cap = np.log1p(y_tr.max() * 2)
    gat = train_nn(GATv2Model(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    gat.eval()
    with torch.no_grad():
        raw = gat(x_ctx, ei_ctx)[n_tr:].numpy()
    gat_pred = np.expm1(np.clip(raw, 0, log_cap))
    gat_s = score(y_te, gat_pred, "GATv2")

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  "
        f"HA={ha['WMAPE']:.3f}  IDW={idw_s['WMAPE']:.3f}  MLR={mlr_s['WMAPE']:.3f}  "
        f"RF={rf_s['WMAPE']:.3f}  XGB={xgb_s['WMAPE']:.3f}  MLP={mlp_s['WMAPE']:.3f}  "
        f"GATv2={gat_s['WMAPE']:.3f}  ({t1-t0:.0f}s)")

    base = {"borough": borough, "n_test": len(te_idx)}
    return [base | ha, base | idw_s, base | mlr_s, base | rf_s, base | xgb_s, base | mlp_s, base | gat_s]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw  = prep_features(df)   # log1p(AI23) + lat + lon, unscaled
    osm_used = [c for c in OSM_COLS if c in df.columns]
    if OSM_ONLY and osm_used:
        mode_tag = f"{len(osm_used)} OSM-POI only + lat + lon  [OSM-ONLY mode]"
    elif osm_used:
        mode_tag = f"{len(FEAT_COLS)} AI23 + {len(osm_used)} OSM-POI + lat + lon  [OSM enriched]"
    else:
        mode_tag = f"{len(FEAT_COLS)} AI23 + lat + lon"
    if PCA_AI23 and AI23_IDX:
        mode_tag += f"  [--pca-ai23: {len(AI23_IDX)} AI23 cols -> {N_PCA_COMPONENTS} PCA components]"
    _reported_n = len(ALL_FEAT_COLS) - (len(AI23_IDX) - N_PCA_COMPONENTS if (PCA_AI23 and AI23_IDX) else 0)
    log(f"Features: {_reported_n} ({mode_tag})")

    log(f"Building multigraph ({len(df):,} nodes)...")
    t0 = time.time()
    knn_ei   = build_knn_edge_index(df["lat"].values, df["lon"].values)
    log(f"  KNN edges (K={K_NEIGHBORS}): {knn_ei.shape[1]:,}  ({time.time()-t0:.1f}s)")
    t1 = time.time()
    route_ei = build_route_edges(df)
    log(f"  Route edges:                {route_ei.shape[1]:,}  ({time.time()-t1:.1f}s)")
    edge_sets = [knn_ei, route_ei]
    if FUNC_SIM:
        t2 = time.time()
        osm_present = [c for c in OSM_COLS if c in df.columns]
        func_ei = build_functional_similarity_edges(df, osm_present)
        log(f"  Functional-sim edges (rho>0.99, top-10):  {func_ei.shape[1]:,}  ({time.time()-t2:.1f}s)")
        edge_sets.append(func_ei)
    full_ei  = torch.unique(torch.cat(edge_sets, dim=1), dim=1)
    log(f"  Combined (deduplicated):    {full_ei.shape[1]:,} total directed edges")

    # Pre-warm PyG JIT kernel (avoids ~2min compile delay inside first fold)
    log("Pre-warming PyG kernel (one-time JIT compile)...")
    _n_in = _reported_n
    _dummy = GATv2Model(_n_in)
    _x = torch.randn(50, _n_in)
    _ei = torch.randint(0, 50, (2, 200))
    with torch.no_grad():
        _dummy(_x, _ei)
    log("  Done.")

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        log(f"\n[QUICK MODE] Running {len(boroughs)} folds only.\n")
    else:
        log(f"\nRunning {len(boroughs)}-fold leave-borough-out CV...\n")

    all_rows = []
    t_start  = time.time()
    for fi, b in enumerate(boroughs):
        rows = run_fold(b, df, X_raw, y_orig, full_ei, fi, len(boroughs))
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    suffix = "_quick" if QUICK else ""
    results.to_csv(f"results_cv_multigraph{suffix}.csv", index=False)

    order = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "GATv2"]
    agg_mean = (results.groupby("model")[["WMAPE","RMSE","MAE"]]
                .agg(["mean","std","median"]).round(4))
    agg_mean.columns = [f"{m}_{s}" for m, s in agg_mean.columns]
    agg = agg_mean.reset_index()
    agg["_o"] = agg["model"].map({m: i for i, m in enumerate(order)})
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.to_csv(f"results_summary_multigraph{suffix}.csv", index=False)

    log(f"\n{'='*76}")
    log(f"RESULTS — {len(boroughs)} leave-borough-out folds  (mean ± std  |  median)")
    log(f"{'='*76}")
    for _, r in agg.iterrows():
        log(f"  {r['model']:<8s}  "
            f"WMAPE={r['WMAPE_mean']:.4f}(±{r['WMAPE_std']:.4f}) med={r['WMAPE_median']:.4f}  "
            f"RMSE={r['RMSE_mean']:.1f}(±{r['RMSE_std']:.1f})  "
            f"MAE={r['MAE_mean']:.1f}(±{r['MAE_std']:.1f})")
    log(f"{'='*76}")
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    log(f"Saved -> results_cv_multigraph{suffix}.csv  |  results_summary_multigraph{suffix}.csv")

if __name__ == "__main__":
    main()
