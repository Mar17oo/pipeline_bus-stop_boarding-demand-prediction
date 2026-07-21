"""
STEP 4J — Per-fold hyperparameter tuning for RF / XGBoost / Ridge
======================================================================================
Motivation (19 Jul 2026, comparing against Zheng et al. 2025's own "Model Selection"
section): Zheng grid-searched every baseline, including the classical ones. This
dissertation's RF/XGBoost/Ridge were run at sensible literature-aligned defaults, never
tuned -- a real asymmetry, since GATv2 went through five architectural interventions while
the tabular baselines it is compared against received none. This script fixes that.

METHODOLOGY (leakage discipline, stated explicitly):
For each of the 33 leave-borough-out OUTER folds, hyperparameters are selected using ONLY
that fold's training data, via a RandomizedSearchCV with an INNER 3-fold random split --
the held-out borough never influences its own fold's hyperparameter choice, exactly
mirroring the no-leakage discipline already used for the per-fold StandardScaler.

Search budget (kept deliberately modest -- this is a robustness check, not a full grid
search across 33 folds x 3 models, which would take far longer than is proportionate here):
  RF:      n_iter=15, inner cv=3  -> 45 fits/fold
  XGBoost: n_iter=15, inner cv=3  -> 45 fits/fold
  Ridge:   RidgeCV (efficient generalized CV, not RandomizedSearchCV -- single
           hyperparameter, closed-form-efficient, no need for a random search)

Best hyperparameters per fold are saved (not just discarded) for transparency --
see results_tuned_hyperparams.csv.

Same protocol as every other headline-set experiment: AI23+OSM+SC only, same 33 folds,
same StandardScaler discipline. Graph-based models (GATv2, GCN) and non-tuned baselines
(HistAvg, IDW, MLP) are reused from results_cv_ai23_osm_sc.csv, since this experiment only
concerns the 3 tuned models.

Run with: python -u step4j_tuned_baselines.py [--quick]
Output: results_cv_tuned.csv / results_summary_tuned.csv / results_tuned_hyperparams.csv
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from step4_model import FEAT_COLS, OSM_COLS, SC_COL, COORD_COLS, SEED

QUICK = "--quick" in sys.argv

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"
HEADLINE_CV = "results_cv_ai23_osm_sc.csv"

N_ITER   = 15
INNER_CV = 3

RF_SPACE = {
    "n_estimators": [100, 150, 200, 300, 400],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}
XGB_SPACE = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}
RIDGE_ALPHAS = np.logspace(-2, 2, 25)   # 0.01 .. 100


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


def run_fold(borough, df, X_raw, y_orig, fi, nf):
    t0 = time.time()
    test_mask  = (df[BOROUGH_COL] == borough).values
    train_mask = ~test_mask
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    y_tr, y_te = y_orig[tr_idx], y_orig[te_idx]

    scaler  = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_raw[tr_idx])
    X_sc_te = scaler.transform(X_raw[te_idx])
    y_tr_log = np.log1p(y_tr)

    inner_cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=SEED + fi)

    # --- Ridge: RidgeCV (efficient built-in generalized CV, not RandomizedSearchCV) ---
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv)
    ridge.fit(X_sc_tr, y_tr_log)
    mlr_s = score(y_te, np.expm1(ridge.predict(X_sc_te)), "MLR-tuned")
    ridge_best = {"alpha": float(ridge.alpha_)}

    # --- RF: RandomizedSearchCV ---
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=SEED),
        RF_SPACE, n_iter=N_ITER, cv=inner_cv, random_state=SEED,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    rf_search.fit(X_sc_tr, y_tr_log)
    rf_s = score(y_te, np.expm1(rf_search.predict(X_sc_te)), "RF-tuned")
    rf_best = rf_search.best_params_

    # --- XGBoost: RandomizedSearchCV ---
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        XGB_SPACE, n_iter=N_ITER, cv=inner_cv, random_state=SEED,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    xgb_search.fit(X_sc_tr, y_tr_log)
    xgb_s = score(y_te, np.expm1(xgb_search.predict(X_sc_te)), "XGBoost-tuned")
    xgb_best = xgb_search.best_params_

    t1 = time.time()
    log(f"  [{fi+1:2d}/{nf}] {borough:<30s}  n={len(te_idx):4d}  "
        f"MLR={mlr_s['WMAPE']:.3f}  RF={rf_s['WMAPE']:.3f}  XGB={xgb_s['WMAPE']:.3f}  "
        f"({t1-t0:.0f}s)")

    base = {"borough": borough, "n_test": len(te_idx)}
    hp_row = {"borough": borough, "ridge_alpha": ridge_best["alpha"],
              **{f"rf_{k}": v for k, v in rf_best.items()},
              **{f"xgb_{k}": v for k, v in xgb_best.items()}}
    return [base | mlr_s, base | rf_s, base | xgb_s], hp_row


def main():
    df     = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    X_raw, active_all = prep_features(df)
    log(f"Features: {len(active_all)} (AI23+OSM+SC, headline set)")
    log(f"Search budget: RF/XGBoost n_iter={N_ITER}, inner cv={INNER_CV} "
        f"({N_ITER*INNER_CV} fits/fold/model); Ridge via RidgeCV over "
        f"{len(RIDGE_ALPHAS)} alphas")

    boroughs = sorted(df[BOROUGH_COL].unique())
    if QUICK:
        boroughs = boroughs[:5]
        log(f"\n[QUICK MODE] {len(boroughs)} boroughs.\n")
    else:
        log(f"\nRunning {len(boroughs)}-fold leave-borough-out CV...\n")

    rows, hp_rows = [], []
    t_start = time.time()
    for fi, b in enumerate(boroughs):
        fold_rows, hp_row = run_fold(b, df, X_raw, y_orig, fi, len(boroughs))
        rows.extend(fold_rows)
        hp_rows.append(hp_row)

    tuned_df = pd.DataFrame(rows)
    headline = pd.read_csv(HEADLINE_CV)
    headline = headline[headline["borough"].isin(tuned_df["borough"].unique()) &
                         headline["model"].isin(["HistAvg", "IDW", "MLP", "GATv2"])]
    combined = pd.concat([headline, tuned_df], ignore_index=True)
    suffix = "_quick" if QUICK else ""
    combined.to_csv(f"results_cv_tuned{suffix}.csv", index=False)
    pd.DataFrame(hp_rows).to_csv(f"results_tuned_hyperparams{suffix}.csv", index=False)

    order = ["HistAvg", "IDW", "MLR-tuned", "RF-tuned", "XGBoost-tuned", "MLP", "GATv2"]
    agg = (combined.groupby("model")[["WMAPE", "RMSE", "MAE"]]
           .agg(["mean", "std", "median"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["_o"] = agg["model"].map({m: i for i, m in enumerate(order)})
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.to_csv(f"results_summary_tuned{suffix}.csv", index=False)

    log(f"\n{'='*76}")
    log(f"RESULTS — {len(boroughs)} leave-borough-out folds (mean ± std | median)")
    log(f"{'='*76}")
    for _, r in agg.iterrows():
        log(f"  {r['model']:<14s} WMAPE={r['WMAPE_mean']:.4f}(±{r['WMAPE_std']:.4f}) med={r['WMAPE_median']:.4f}")
    log(f"{'='*76}")
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    log(f"Saved -> results_cv_tuned{suffix}.csv | results_summary_tuned{suffix}.csv | "
        f"results_tuned_hyperparams{suffix}.csv")


if __name__ == "__main__":
    main()
