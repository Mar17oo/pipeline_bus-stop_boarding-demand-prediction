"""
STEP 4E — Zheng et al. (2025)-style G1/G2/G3 fusion GAT (Experiment Set 7, option 3)
========================================================================================
Faithful-ish replica of the multigraph fusion architecture in Zheng et al.
(2025), Eqs. (8)-(12): two SEPARATE GATv2 branches, one over a geographic
adjacency graph G1 and one over a functional-similarity graph G2 (Pearson
correlation of OSM POI vectors), concatenated into a fused feature matrix,
then a THIRD GATv2 pass over the union graph G3 = G1 ∪ G2.

Deviations from the literal paper, both flagged and justified:
  1. rho>0.8 threshold on our 6-category OSM POI data produces a
     near-complete graph (48M edges, avg degree 2,680) — computationally
     infeasible and not "local structure" any more. Recalibrated to
     rho>0.99 + top-10 neighbours per stop (see build_functional_similarity_edges
     in step4_model.py, reused here). Density sweep is documented in
     experiment_log.md Experiment Set 7.
  2. A residual skip connection (raw input -> final output) is added, which
     Zheng's paper does not have. This project found (Bug 5) that a skip
     connection is critical for leave-borough-out cold-start robustness —
     omitting it here would confound "does fusion help" with "does this
     architecture lack the skip-connection fix we already know matters",
     so it is kept for a fair comparison against our own GATv2Model.
  3. G1 here = KNN(K=5) + route-connectivity edges (our existing best
     geographic graph), not pure geographic adjacency as in the paper —
     route edges are additive structural signal we already validated works,
     dropping them would not be a fair test of "fusion" specifically.

COMPUTE SAVING: HistAvg/IDW/MLR/RF/XGBoost/MLP do not depend on the graph at
all, so their per-fold numbers are identical to the headline AI23+OSM+SC run
(same folds, same seed, same data) and are copied from
results_cv_ai23_osm_sc.csv rather than recomputed. Only the new fusion GAT
is trained here.

Run with: python -u step4e_zheng_fusion.py [--quick]
Output: results_cv_zheng_fusion.csv / results_summary_zheng_fusion.csv
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import subgraph as pyg_subgraph

from step4_model import (
    build_knn_edge_index, build_route_edges, build_functional_similarity_edges,
    FEAT_COLS, OSM_COLS, SC_COL, COORD_COLS,
)

QUICK = "--quick" in sys.argv

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"
HEADLINE_CV = "results_cv_ai23_osm_sc.csv"   # reused for graph-independent models

K_NEIGHBORS = 5
HIDDEN_DIM  = 64
HEADS       = 4
DROPOUT     = 0.15
LR          = 5e-4
EPOCHS      = 500
PATIENCE    = 20
VAL_EVERY   = 5
VAL_FRAC    = 0.1
SEED        = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def prep_features(df):
    """AI23 + OSM + service_coverage + coords (headline feature set), same as
    step4_model.py's default --with-sc config."""
    osm_present = [c for c in OSM_COLS if c in df.columns]
    active_feat = FEAT_COLS + osm_present + [SC_COL]
    active_all  = active_feat + COORD_COLS
    X = np.zeros((len(df), len(active_all)), dtype=np.float32)
    for i, c in enumerate(active_feat):
        X[:, i] = np.log1p(df[c].values)
    X[:, len(active_feat)]     = df["lat"].values
    X[:, len(active_feat) + 1] = df["lon"].values
    return X, active_all, osm_present


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


class ZhengFusionGAT(torch.nn.Module):
    """Eqs (8)-(12): separate GAT over G1 (geographic) and G2 (functional
    similarity), concatenated, then a third GAT over the union graph G3.
    Residual skip added for fair comparison (see module docstring point 2)."""
    def __init__(self, in_ch):
        super().__init__()
        self.gat_g1 = GATv2Conv(in_ch, HIDDEN_DIM, heads=HEADS, dropout=DROPOUT, concat=True)
        self.gat_g2 = GATv2Conv(in_ch, HIDDEN_DIM, heads=HEADS, dropout=DROPOUT, concat=True)
        fused_dim = HIDDEN_DIM * HEADS * 2
        self.gat_g3 = GATv2Conv(fused_dim, 1, heads=1, dropout=DROPOUT, concat=False)
        self.skip = torch.nn.Linear(in_ch, 1, bias=False)

    def forward(self, x, ei_g1, ei_g2, ei_g3):
        z_adj = F.elu(self.gat_g1(x, ei_g1))
        z_f   = F.elu(self.gat_g2(x, ei_g2))
        z_fused = torch.cat([z_adj, z_f], dim=1)
        z_fused = F.dropout(z_fused, p=DROPOUT, training=self.training)
        return (self.gat_g3(z_fused, ei_g3) + self.skip(x)).squeeze(-1)


def train_nn_fusion(model, x, y, ei_g1, ei_g2, ei_g3, tr_pos, val_pos):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out  = model(x, ei_g1, ei_g2, ei_g3)
        loss = F.huber_loss(out[tr_pos], y[tr_pos], delta=0.5)
        loss.backward(); opt.step()
        if (epoch + 1) % VAL_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            vl = F.huber_loss(model(x, ei_g1, ei_g2, ei_g3)[val_pos], y[val_pos], delta=0.5).item()
        if vl < best_val - 1e-6:
            best_val, best_w, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_w:
        model.load_state_dict(best_w)
    return model


def run_fold(borough, df, X_raw, y_orig, g1_ei, g2_ei, g3_ei, n_in, fi, nf):
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
    n_all  = len(df)

    tr_t = torch.tensor(tr_idx, dtype=torch.long)
    g1_tr, _ = pyg_subgraph(tr_t, g1_ei, relabel_nodes=True, num_nodes=n_all)
    g2_tr, _ = pyg_subgraph(tr_t, g2_ei, relabel_nodes=True, num_nodes=n_all)
    g3_tr, _ = pyg_subgraph(tr_t, g3_ei, relabel_nodes=True, num_nodes=n_all)

    all_t = torch.tensor(all_idx, dtype=torch.long)
    g1_ctx, _ = pyg_subgraph(all_t, g1_ei, relabel_nodes=True, num_nodes=n_all)
    g2_ctx, _ = pyg_subgraph(all_t, g2_ei, relabel_nodes=True, num_nodes=n_all)
    g3_ctx, _ = pyg_subgraph(all_t, g3_ei, relabel_nodes=True, num_nodes=n_all)

    log_cap = np.log1p(y_tr.max() * 2)
    model = train_nn_fusion(ZhengFusionGAT(n_in), x_tr, y_tr_t, g1_tr, g2_tr, g3_tr, tp_t, vp_t)
    model.eval()
    with torch.no_grad():
        raw = model(x_ctx, g1_ctx, g2_ctx, g3_ctx)[n_tr:].numpy()
    pred = np.expm1(np.clip(raw, 0, log_cap))
    s = score(y_te, pred, "GATv2-Fusion")

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  "
        f"GATv2-Fusion={s['WMAPE']:.3f}  ({t1-t0:.0f}s)")
    return {"borough": borough, "n_test": len(te_idx)} | s


def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw, active_all, osm_present = prep_features(df)
    n_in = len(active_all)
    log(f"Features: {n_in} (AI23+OSM+SC, headline set)")

    log(f"Building G1 (geographic: KNN+route)...")
    t0 = time.time()
    knn_ei   = build_knn_edge_index(df["lat"].values, df["lon"].values, k=K_NEIGHBORS)
    route_ei = build_route_edges(df)
    g1_ei = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
    log(f"  G1: {g1_ei.shape[1]:,} directed edges  ({time.time()-t0:.1f}s)")

    log(f"Building G2 (functional similarity: rho>0.99, top-10)...")
    t0 = time.time()
    g2_ei = build_functional_similarity_edges(df, osm_present)
    log(f"  G2: {g2_ei.shape[1]:,} directed edges  ({time.time()-t0:.1f}s)")

    g3_ei = torch.unique(torch.cat([g1_ei, g2_ei], dim=1), dim=1)
    log(f"  G3 (union): {g3_ei.shape[1]:,} directed edges")

    log("Pre-warming PyG kernel...")
    _dummy = ZhengFusionGAT(n_in)
    _x = torch.randn(50, n_in)
    _ei = torch.randint(0, 50, (2, 200))
    with torch.no_grad():
        _dummy(_x, _ei, _ei, _ei)
    log("  Done.")

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        log(f"\n[QUICK MODE] Running {len(boroughs)} folds only.\n")
    else:
        log(f"\nRunning {len(boroughs)}-fold leave-borough-out CV...\n")

    fusion_rows = []
    t_start = time.time()
    for fi, b in enumerate(boroughs):
        fusion_rows.append(run_fold(b, df, X_raw, y_orig, g1_ei, g2_ei, g3_ei, n_in, fi, len(boroughs)))

    fusion_df = pd.DataFrame(fusion_rows)

    # Reuse graph-independent models' per-fold numbers from the headline run.
    headline = pd.read_csv(HEADLINE_CV)
    headline = headline[headline["borough"].isin(fusion_df["borough"])]
    combined = pd.concat([headline, fusion_df], ignore_index=True)
    suffix = "_quick" if QUICK else ""
    combined.to_csv(f"results_cv_zheng_fusion{suffix}.csv", index=False)

    order = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "GATv2", "GATv2-Fusion"]
    agg = (combined.groupby("model")[["WMAPE", "RMSE", "MAE"]]
           .agg(["mean", "std", "median"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["_o"] = agg["model"].map({m: i for i, m in enumerate(order)})
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.to_csv(f"results_summary_zheng_fusion{suffix}.csv", index=False)

    log(f"\n{'='*76}")
    log(f"RESULTS — {len(boroughs)} leave-borough-out folds (mean ± std | median)")
    log(f"{'='*76}")
    for _, r in agg.iterrows():
        log(f"  {r['model']:<14s} WMAPE={r['WMAPE_mean']:.4f}(±{r['WMAPE_std']:.4f}) "
            f"med={r['WMAPE_median']:.4f}")
    log(f"{'='*76}")
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    log(f"Saved -> results_cv_zheng_fusion{suffix}.csv | results_summary_zheng_fusion{suffix}.csv")


if __name__ == "__main__":
    main()
