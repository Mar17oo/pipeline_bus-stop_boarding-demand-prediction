"""
STEP 4F — Pre-registered experiment: learned per-node MLP/GATv2 mixing gate
================================================================================
ONE experiment, pre-registered interpretation, no hyperparameter search, no
architecture variants, no second attempt if it loses. See the task spec this
was built from (pasted into the coding chat, dated 18-19 Jul 2026) for the
full pre-registration; the interpretation rules are reproduced at the bottom
of this file's docstring and are applied verbatim to whatever the numbers
turn out to be.

HYPOTHESIS: message passing dilutes a strong node-local signal (evidence:
MLP-GATv2 gap is 0.75pp WITHOUT service_coverage, 8.76pp WITH it). If true,
an architecture that can freely choose how much neighbour information to use
will learn to use almost none.

ARCHITECTURE:
    h_out = alpha * MLP(x_self) + (1 - alpha) * GATv2_aggregate(neighbours)
    alpha = sigmoid(Linear(in_ch, 1)(x_self))   -- per-node, from x_self only

Implementation choice (documented, not a deviation): MLP(x_self) and
GATv2_aggregate(neighbours) are the EXISTING, UNMODIFIED MLPModel and
GATv2Model classes imported directly from step4_model.py -- this is what
makes "alpha=1 recovers the MLP exactly" and "alpha=0 recovers pure message
passing" literally true, including GATv2Model's own residual skip
connection (kept, per the spec's "keep everything else identical... residual
skip"). The gate is the ONLY new component: one Linear(in_ch, 1) + sigmoid.

Feature set: AI23+OSM+SC only (the headline set). K=5 KNN + route graph,
same as the frozen config. >=3 random seeds (this run uses exactly 3: 42,
142, 242) -- each a full 33-fold leave-borough-out CV. No hyperparameter
search: HIDDEN_DIM/HEADS/DROPOUT/LR/EPOCHS/PATIENCE/VAL_EVERY/K_NEIGHBORS
are copied verbatim from step4_model.py's frozen config.

Run with: python -u step4f_gated_mixing.py [--quick]
Output: results_cv_gated_seed<SEED>.csv (one per seed) +
        results_gated_alpha_seed<SEED>.csv (per-stop alpha, one per seed) +
        results_summary_gated_all_seeds.csv (aggregated, after all seeds run)

PRE-REGISTERED INTERPRETATION (written here BEFORE seeing results; applied
verbatim in experiment_log.md regardless of outcome):
  - mean alpha HIGH (>0.8) and WMAPE approaches MLP (0.6311): the model
    learned to switch the graph off. CONFIRMS Finding 3 from inside the
    architecture -- strongest available evidence, not a rescue of the GNN.
  - alpha MODERATE and WMAPE beats GATv2-Fusion (0.7070) but not MLP
    (0.6311): graph info is weakly useful but net-negative when forced.
    Report as a bounded, honest partial gain.
  - beats MLP (0.6311) with p<0.05 after Holm-Bonferroni across the 2
    comparisons (vs MLP, vs plain GATv2): Finding 3 must be REWRITTEN, not
    deleted -- the loss was architectural (forced mixing), not evidential
    (graphs are useless here).
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
from torch_geometric.utils import subgraph as pyg_subgraph

from step4_model import (
    build_knn_edge_index, build_route_edges,
    FEAT_COLS, OSM_COLS, SC_COL, COORD_COLS,
    GATv2Model, MLPModel,
    HIDDEN_DIM, HEADS, DROPOUT, LR, EPOCHS, PATIENCE, VAL_EVERY, VAL_FRAC,
    K_NEIGHBORS,
)

QUICK = "--quick" in sys.argv
SEEDS = [42, 142, 242]   # >=3 random seeds, per spec

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"


def prep_features(df):
    """AI23 + OSM + service_coverage + coords -- the headline feature set,
    per spec ("Feature set: AI23+OSM+SC only")."""
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


class GatedGATv2Model(torch.nn.Module):
    """h_out = alpha*MLP(x) + (1-alpha)*GATv2(x, ei); alpha=sigmoid(gate(x)).
    MLPModel and GATv2Model are used UNMODIFIED (imported from
    step4_model.py) so alpha=1/alpha=0 recover them exactly."""
    def __init__(self, in_ch):
        super().__init__()
        self.mlp  = MLPModel(in_ch)
        self.gat  = GATv2Model(in_ch)
        self.gate = torch.nn.Linear(in_ch, 1)

    def forward(self, x, edge_index):
        mlp_out = self.mlp(x)
        gat_out = self.gat(x, edge_index)
        alpha   = torch.sigmoid(self.gate(x)).squeeze(-1)
        h_out   = alpha * mlp_out + (1 - alpha) * gat_out
        return h_out, alpha


def train_nn_gated(model, x, y, ei, tr_pos, val_pos):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val, best_w, no_imp = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out, _ = model(x, ei)
        loss = F.huber_loss(out[tr_pos], y[tr_pos], delta=0.5)
        loss.backward(); opt.step()
        if (epoch + 1) % VAL_EVERY != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_out, _ = model(x, ei)
            vl = F.huber_loss(val_out[val_pos], y[val_pos], delta=0.5).item()
        if vl < best_val - 1e-6:
            best_val, best_w, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_w:
        model.load_state_dict(best_w)
    return model


def run_fold(borough, df, X_raw, y_orig, full_ei, n_in, fi, nf, seed):
    t0 = time.time()
    test_mask  = (df[BOROUGH_COL] == borough).values
    train_mask = ~test_mask
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    y_tr, y_te = y_orig[tr_idx], y_orig[te_idx]

    scaler  = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_raw[tr_idx])
    X_sc_te = scaler.transform(X_raw[te_idx])

    rng   = np.random.RandomState(seed + fi)
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
    model = train_nn_gated(GatedGATv2Model(n_in), x_tr, y_tr_t, ei_tr, tp_t, vp_t)
    model.eval()
    with torch.no_grad():
        raw, alpha_ctx = model(x_ctx, ei_ctx)
    raw = raw[n_tr:].numpy()
    alpha_test = alpha_ctx[n_tr:].numpy()
    pred = np.expm1(np.clip(raw, 0, log_cap))
    s = score(y_te, pred, "Gated")

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  "
        f"Gated={s['WMAPE']:.3f}  mean_alpha={alpha_test.mean():.3f}  ({t1-t0:.0f}s)")

    fold_row = {"borough": borough, "n_test": len(te_idx)} | s
    alpha_rows = [{"borough": borough, "STOPCODE": df.iloc[i]["STOPCODE"], "alpha": a}
                  for i, a in zip(te_idx, alpha_test)]
    return fold_row, alpha_rows


def run_one_seed(seed, df, X_raw, y_orig, full_ei, n_in, boroughs):
    torch.manual_seed(seed); np.random.seed(seed)
    log(f"\n{'='*76}\nSEED {seed}\n{'='*76}")
    fold_rows, alpha_rows = [], []
    t0 = time.time()
    for fi, b in enumerate(boroughs):
        fr, ar = run_fold(b, df, X_raw, y_orig, full_ei, n_in, fi, len(boroughs), seed)
        fold_rows.append(fr)
        alpha_rows.extend(ar)
    cv_df = pd.DataFrame(fold_rows)
    alpha_df = pd.DataFrame(alpha_rows)
    suffix = "_quick" if QUICK else ""
    cv_df.to_csv(f"results_cv_gated_seed{seed}{suffix}.csv", index=False)
    alpha_df.to_csv(f"results_gated_alpha_seed{seed}{suffix}.csv", index=False)
    log(f"Seed {seed} done in {(time.time()-t0)/60:.1f} min. "
        f"Mean WMAPE={cv_df['WMAPE'].mean():.4f}  Mean alpha={alpha_df['alpha'].mean():.4f}")
    return cv_df, alpha_df


def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw, active_all = prep_features(df)
    n_in = len(active_all)
    log(f"Features: {n_in} (AI23+OSM+SC, headline set)")

    log("Building multigraph (K=5 KNN + route, same as frozen config)...")
    t0 = time.time()
    knn_ei   = build_knn_edge_index(df["lat"].values, df["lon"].values, k=K_NEIGHBORS)
    route_ei = build_route_edges(df)
    full_ei = torch.unique(torch.cat([knn_ei, route_ei], dim=1), dim=1)
    log(f"  {full_ei.shape[1]:,} directed edges  ({time.time()-t0:.1f}s)")

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        seeds = SEEDS[:1]
        log(f"\n[QUICK MODE] {len(boroughs)} boroughs, {len(seeds)} seed only.\n")
    else:
        seeds = SEEDS
        log(f"\nFull run: {len(boroughs)} boroughs x {len(seeds)} seeds.\n")

    all_cv, all_alpha = [], []
    t_start = time.time()
    for seed in seeds:
        cv_df, alpha_df = run_one_seed(seed, df, X_raw, y_orig, full_ei, n_in, boroughs)
        cv_df["seed"] = seed
        alpha_df["seed"] = seed
        all_cv.append(cv_df)
        all_alpha.append(alpha_df)

    combined_cv = pd.concat(all_cv, ignore_index=True)
    combined_alpha = pd.concat(all_alpha, ignore_index=True)
    suffix = "_quick" if QUICK else ""
    combined_cv.to_csv(f"results_cv_gated_all_seeds{suffix}.csv", index=False)
    combined_alpha.to_csv(f"results_gated_alpha_all_seeds{suffix}.csv", index=False)

    log(f"\n{'='*76}")
    log(f"ALL SEEDS DONE. Total time: {(time.time()-t_start)/60:.1f} min")
    log(f"Mean WMAPE across all seeds x folds: {combined_cv['WMAPE'].mean():.4f} "
        f"(sd={combined_cv['WMAPE'].std():.4f})")
    per_seed = combined_cv.groupby("seed")["WMAPE"].mean()
    log(f"Per-seed mean WMAPE: {per_seed.to_dict()}")
    log(f"Mean alpha across all seeds x test stops: {combined_alpha['alpha'].mean():.4f}")
    log(f"{'='*76}")
    log(f"Saved -> results_cv_gated_all_seeds{suffix}.csv | results_gated_alpha_all_seeds{suffix}.csv")


if __name__ == "__main__":
    main()
