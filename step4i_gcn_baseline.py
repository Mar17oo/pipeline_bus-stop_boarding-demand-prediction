"""
STEP 4I — GCN baseline: does attention matter, or would any graph convolution do?
======================================================================================
Motivated directly by Zheng et al. (2025)'s own baseline design: their ASTGCN baseline
swaps GCN in for GAT, specifically to isolate whether attention is doing real work or
whether any graph convolution performs similarly. This dissertation's baseline suite had
not asked that question until now (flagged 19 Jul 2026 when comparing against Zheng's
"Model Selection" section directly).

GCNModel mirrors GATv2Model exactly (same hidden width 256, same residual skip, same
dropout placement) with ONLY the conv layer swapped -- GCNConv instead of GATv2Conv -- so
any WMAPE difference between them isolates the attention mechanism's contribution, not a
confound from differing capacity or regularization.

One structural difference is unavoidable and worth stating plainly, not hiding: GATv2Conv
has its own internal dropout on attention weights; GCNConv has no attention weights to drop
(plain convolution), so its regularization comes entirely from the explicit F.dropout call
applied identically in both models. This is not a fixable discrepancy -- it is what "no
attention mechanism" structurally means.

Same protocol as every other GATv2-variant experiment in this project: AI23+OSM+SC only, K=5
KNN+route graph, same folds, same seed (42, matching the headline run). Graph-independent
models (HistAvg/IDW/MLR/RF/XGBoost/MLP/GATv2) are reused from results_cv_ai23_osm_sc.csv
rather than recomputed.

Run with: python -u step4i_gcn_baseline.py [--quick]
Output: results_cv_gcn.csv / results_summary_gcn.csv
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
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph as pyg_subgraph

from step4_model import (
    build_knn_edge_index, build_route_edges,
    FEAT_COLS, OSM_COLS, SC_COL, COORD_COLS,
    HIDDEN_DIM, HEADS, DROPOUT, LR, EPOCHS, PATIENCE, VAL_EVERY, VAL_FRAC,
    K_NEIGHBORS, SEED,
)

QUICK = "--quick" in sys.argv

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"
HEADLINE_CV = "results_cv_ai23_osm_sc.csv"


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


class GCNModel(torch.nn.Module):
    """GATv2Model with the attention conv swapped for plain GCN -- everything else
    (hidden width, residual skip, dropout placement) held identical, so the only
    difference is: does message weighting via learned attention matter?"""
    def __init__(self, in_ch):
        super().__init__()
        h = HIDDEN_DIM * HEADS   # 256 -- matches GATv2Model's concatenated multi-head width
        self.c1   = GCNConv(in_ch, h)
        self.c2   = GCNConv(h, 1)
        self.skip = torch.nn.Linear(in_ch, h, bias=False)

    def forward(self, x, edge_index):
        h = F.elu(self.c1(x, edge_index)) + self.skip(x)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.c2(h, edge_index).squeeze(-1)


def train_nn(model, x, y, ei, tr_pos, val_pos):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out  = model(x, ei)
        loss = F.huber_loss(out[tr_pos], y[tr_pos], delta=0.5)
        loss.backward(); opt.step()
        if (epoch + 1) % VAL_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            vl = F.huber_loss(model(x, ei)[val_pos], y[val_pos], delta=0.5).item()
        if vl < best_val - 1e-6:
            best_val, best_w, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_w:
        model.load_state_dict(best_w)
    return model


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

    log_cap = np.log1p(y_tr.max() * 2)
    model = train_nn(GCNModel(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    model.eval()
    with torch.no_grad():
        raw = model(x_ctx, ei_ctx)[n_tr:].numpy()
    pred = np.expm1(np.clip(raw, 0, log_cap))
    s = score(y_te, pred, "GCN")

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  GCN={s['WMAPE']:.3f}  ({t1-t0:.0f}s)")
    return {"borough": borough, "n_test": len(te_idx)} | s


def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw, active_all = prep_features(df)
    n_in = len(active_all)
    log(f"Features: {n_in} (AI23+OSM+SC, headline set). Seed: {SEED}")

    torch.manual_seed(SEED); np.random.seed(SEED)

    log("Building multigraph (K=5 KNN + route, same as frozen config)...")
    knn_ei   = build_knn_edge_index(df["lat"].values, df["lon"].values, k=K_NEIGHBORS)
    route_ei = build_route_edges(df)
    full_ei = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
    log(f"  {full_ei.shape[1]:,} directed edges")

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        log(f"\n[QUICK MODE] {len(boroughs)} boroughs.\n")
    else:
        log(f"\nRunning {len(boroughs)}-fold leave-borough-out CV...\n")

    rows = []
    t_start = time.time()
    for fi, b in enumerate(boroughs):
        rows.append(run_fold(b, df, X_raw, y_orig, full_ei, n_in, fi, len(boroughs)))

    gcn_df = pd.DataFrame(rows)
    headline = pd.read_csv(HEADLINE_CV)
    headline = headline[headline["borough"].isin(gcn_df["borough"])]
    combined = pd.concat([headline, gcn_df], ignore_index=True)
    suffix = "_quick" if QUICK else ""
    combined.to_csv(f"results_cv_gcn{suffix}.csv", index=False)

    order = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "GATv2", "GCN"]
    agg = (combined.groupby("model")[["WMAPE", "RMSE", "MAE"]]
           .agg(["mean", "std", "median"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["_o"] = agg["model"].map({m: i for i, m in enumerate(order)})
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.to_csv(f"results_summary_gcn{suffix}.csv", index=False)

    log(f"\n{'='*76}")
    log(f"RESULTS — {len(boroughs)} leave-borough-out folds (mean ± std | median)")
    log(f"{'='*76}")
    for _, r in agg.iterrows():
        log(f"  {r['model']:<10s} WMAPE={r['WMAPE_mean']:.4f}(±{r['WMAPE_std']:.4f}) med={r['WMAPE_median']:.4f}")
    log(f"{'='*76}")
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    log(f"Saved -> results_cv_gcn{suffix}.csv | results_summary_gcn{suffix}.csv")


if __name__ == "__main__":
    main()
