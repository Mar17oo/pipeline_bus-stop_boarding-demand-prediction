"""
STEP 4H — Residual correlation between standalone MLP and GATv2 (follow-up to Experiment Set 8)
====================================================================================================
Experiment Set 8's gated-mixing model beat MLP by a small but significant
margin (0.6277 vs 0.6311, p=0.007), with a non-adaptive alpha (~0.52
everywhere, uncorrelated with where the graph helps/hurts). That pattern is
consistent with either (a) the graph carrying real complementary spatial
signal, or (b) plain ensembling of two decorrelated predictors -- and the
original run didn't save raw predictions to tell them apart.

This script retrains standalone MLPModel and GATv2Model (UNMODIFIED, same
classes as the frozen config), same folds, same seed (42, matching the
original headline run), AI23+OSM+SC only -- and this time saves the raw
per-stop predictions from both models (log1p-scale, pre-expm1, AND the
final expm1'd scale) so two things can be computed with no further model
training:
  1. Correlation between MLP's and GATv2's residuals (are they making the
     same mistakes on the same stops, or different ones?).
  2. WMAPE of a naive FIXED 50/50 average of the two models' log1p-scale
     outputs (i.e. the same mixing formula Experiment Set 8 used, but with
     alpha fixed at 0.5 instead of learned) -- if this naive, non-learned
     ensemble matches Experiment Set 8's Gated result, that is decisive
     evidence the learned gate isn't doing anything beyond generic
     ensembling.

Run with: python -u step4h_residual_correlation.py [--quick]
Output: results_residual_correlation.csv (per-stop, all 17,943 rows)
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.utils import subgraph as pyg_subgraph

from step4_model import (
    build_knn_edge_index, build_route_edges,
    FEAT_COLS, OSM_COLS, SC_COL, COORD_COLS,
    GATv2Model, MLPModel, train_nn,
    VAL_FRAC, K_NEIGHBORS, SEED,
)

QUICK = "--quick" in sys.argv

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"


def prep_features(df):
    osm_present = [c for c in OSM_COLS if c in df.columns]
    active_feat = FEAT_COLS + osm_present + [SC_COL]
    active_all  = active_feat + COORD_COLS
    X = np.zeros((len(df), len(active_all)), dtype=np.float32)
    for i, c in enumerate(active_feat):
        X[:, i] = np.log1p(df[c].values)
    X[:, len(active_feat)]     = df["lat"].values
    X[:, len(active_feat) + 1] = df["lon"].values
    return X, active_all


def run_fold(borough, df, X_raw, y_orig, full_ei, n_in, fi, nf):
    t0 = time.time()
    test_mask  = (df[BOROUGH_COL] == borough).values
    train_mask = ~test_mask
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    y_tr, y_te = y_orig[tr_idx], y_orig[te_idx]

    scaler  = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_raw[tr_idx])
    X_sc_te = scaler.transform(X_raw[te_idx])

    rng   = np.random.RandomState(SEED + fi)
    n_val = max(5, int(len(tr_idx) * VAL_FRAC))
    log_y_tr = np.log1p(y_tr)
    quantile_labels = pd.qcut(log_y_tr, q=5, labels=False, duplicates="drop")
    vp_list, n_per_q = [], max(1, n_val // 5)
    for q in range(5):
        q_idx = np.where(quantile_labels == q)[0]
        if len(q_idx) > 0:
            vp_list.extend(rng.choice(q_idx, min(n_per_q, len(q_idx)), replace=False).tolist())
    vp = np.array(vp_list)
    tp = np.setdiff1d(np.arange(len(tr_idx)), vp)
    tp_t, vp_t = torch.tensor(tp, dtype=torch.long), torch.tensor(vp, dtype=torch.long)

    all_idx  = np.concatenate([tr_idx, te_idx])
    X_sc_all = np.vstack([X_sc_tr, X_sc_te])
    x_tr   = torch.tensor(X_sc_tr, dtype=torch.float)
    x_ctx  = torch.tensor(X_sc_all, dtype=torch.float)
    y_tr_t = torch.tensor(np.log1p(y_tr), dtype=torch.float)
    n_tr   = len(tr_idx)

    tr_t    = torch.tensor(tr_idx, dtype=torch.long)
    ei_tr,_ = pyg_subgraph(tr_t, full_ei, relabel_nodes=True, num_nodes=len(df))
    all_t    = torch.tensor(all_idx, dtype=torch.long)
    ei_ctx,_ = pyg_subgraph(all_t, full_ei, relabel_nodes=True, num_nodes=len(df))

    mlp = train_nn(MLPModel(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    mlp.eval()
    with torch.no_grad():
        mlp_raw = mlp(x_ctx)[n_tr:].numpy()   # log1p-scale, pre-expm1

    gat = train_nn(GATv2Model(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    gat.eval()
    with torch.no_grad():
        gat_raw = gat(x_ctx, ei_ctx)[n_tr:].numpy()   # log1p-scale, pre-expm1

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  ({t1-t0:.0f}s)")

    return pd.DataFrame({
        "borough": borough,
        "STOPCODE": df.iloc[te_idx]["STOPCODE"].values,
        "y_true": y_te,
        "y_true_log1p": np.log1p(y_te),
        "mlp_raw_log1p": mlp_raw,
        "gat_raw_log1p": gat_raw,
    })


def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw, active_all = prep_features(df)
    n_in = len(active_all)
    log(f"Features: {n_in} (AI23+OSM+SC). Seed: {SEED} (matches original headline run)")

    log("Building multigraph (K=5 KNN + route)...")
    knn_ei   = build_knn_edge_index(df["lat"].values, df["lon"].values, k=K_NEIGHBORS)
    route_ei = build_route_edges(df)
    full_ei = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
    log(f"  {full_ei.shape[1]:,} directed edges")

    torch.manual_seed(SEED); np.random.seed(SEED)

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        log(f"\n[QUICK MODE] {len(boroughs)} boroughs.\n")

    all_rows = []
    t_start = time.time()
    for fi, b in enumerate(boroughs):
        all_rows.append(run_fold(b, df, X_raw, y_orig, full_ei, n_in, fi, len(boroughs)))

    result = pd.concat(all_rows, ignore_index=True)
    suffix = "_quick" if QUICK else ""
    result.to_csv(f"results_residual_correlation{suffix}.csv", index=False)
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    log(f"Saved -> results_residual_correlation{suffix}.csv  ({len(result)} rows)")


if __name__ == "__main__":
    main()
