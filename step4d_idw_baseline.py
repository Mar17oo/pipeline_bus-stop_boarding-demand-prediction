"""
STEP 4D — IDW spatial-interpolation baseline (fast, no NN retraining)
=========================================================================
Classical geostatistical cold-start baseline, per Liu et al. (2017)'s NYC
Citi Bike cold-start work (gravity models + spatial interpolation: natural-
neighbor interpolation, kriging). Predicts a held-out stop's log1p(boardings)
as an inverse-distance-weighted average of its k=20 nearest TRAINING stops'
log1p(boardings) — uses ONLY lat/lon, no AI23/OSM/service_coverage features
at all. Tests whether GATv2's learned neighbour-aggregation adds anything
beyond what plain spatial interpolation already gives for free.

IDW is feature-set-independent (same stops, same target, same coordinates
regardless of which feature columns are "active"), so this is run ONCE on
the full 17,943-stop set and the resulting mean WMAPE applies identically to
every column of the consolidated table — the same way HistAvg does.

Run time: ~10-20 sec total (no training, just distance-weighted averaging).
Output: results_cv_idw.csv / results_summary_idw.csv
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import BallTree

def log(msg):
    print(msg, flush=True)

DATA_FILE   = "stops_features_osm.csv"
BOROUGH_COL = "lad_name"
TARGET_COL  = "total_boardings"
K           = 20
POWER       = 2


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


def idw_predict(train_lat, train_lon, train_y_log, test_lat, test_lon, k=K, power=POWER):
    train_coords = np.radians(np.column_stack([train_lat, train_lon]))
    test_coords  = np.radians(np.column_stack([test_lat, test_lon]))
    k_use = min(k, len(train_lat))
    dist, idx = BallTree(train_coords, metric="haversine").query(test_coords, k=k_use)
    dist_km = dist * 6371.0088
    w = 1.0 / np.power(dist_km + 1e-6, power)
    neighbor_y = train_y_log[idx]
    return np.sum(w * neighbor_y, axis=1) / np.sum(w, axis=1)


def main():
    df = pd.read_csv(DATA_FILE)
    y_orig = df[TARGET_COL].values.astype(float)
    boroughs = sorted(df[BOROUGH_COL].unique())

    rows = []
    t0 = time.time()
    for fi, borough in enumerate(boroughs):
        test_mask  = (df[BOROUGH_COL] == borough).values
        train_mask = ~test_mask
        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]
        y_tr, y_te = y_orig[tr_idx], y_orig[te_idx]

        pred = np.expm1(idw_predict(df["lat"].values[tr_idx], df["lon"].values[tr_idx],
                                     np.log1p(y_tr),
                                     df["lat"].values[te_idx], df["lon"].values[te_idx]))
        s = score(y_te, pred, "IDW")
        rows.append({"borough": borough, "n_test": len(te_idx)} | s)
        log(f"  [{fi+1:2d}/{len(boroughs)}] {borough:<30s} n={len(te_idx):4d}  IDW={s['WMAPE']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv("results_cv_idw.csv", index=False)

    agg = (results.groupby("model")[["WMAPE", "RMSE", "MAE"]]
           .agg(["mean", "std", "median"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv("results_summary_idw.csv", index=False)

    log(f"\nDone in {(time.time()-t0):.1f}s")
    for _, r in agg.iterrows():
        log(f"  {r['model']:<8s} WMAPE={r['WMAPE_mean']:.4f}(+/-{r['WMAPE_std']:.4f}) "
            f"med={r['WMAPE_median']:.4f}")
    log("Saved -> results_cv_idw.csv | results_summary_idw.csv")


if __name__ == "__main__":
    main()
