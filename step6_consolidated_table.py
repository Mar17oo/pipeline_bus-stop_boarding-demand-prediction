"""
STEP 6 — Consolidated results table (Chapter 4 deliverable)
=============================================================
Assembles ONE table: rows = {HistAvg, MLR, RF, XGBoost, MLP, GATv2,
GATv2-fairness-variant}, columns = {AI23-only, OSM-only, AI23+OSM,
AI23+OSM+SC}, cells = mean WMAPE over 33 leave-borough-out folds.

Sources per column:
  AI23-only / OSM-only / AI23+OSM:
    HistAvg/MLR/RF/XGBoost <- results_summary_<tag>_fastbaselines.csv (new, v3 protocol)
    MLP/GATv2              <- results_summary_<tag>.csv (v1, pre-fix — documented caveat)
  AI23+OSM+SC:
    all rows <- results_summary_ai23_osm_sc.csv (v3, current code, headline experiment)
  GATv2-fairness-variant (K=10, --with-sc):
    <- results_summary_gnn_fairness.csv, only populated in the AI23+OSM+SC column
       (the fairness check was run on the best feature set only, per the frozen plan)

Output: consolidated_results_table.csv + printed markdown table.
"""

import pandas as pd

ROWS = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "GATv2", "GCN",
        "GATv2-fairness (K=10)", "GATv2-func-sim", "GATv2-Fusion (G1/G2/G3)",
        "MLR-tuned", "RF-tuned", "XGBoost-tuned"]
COLS = ["AI23-only", "OSM-only", "AI23+OSM", "AI23+OSM+SC", "PCA(AI23)+OSM+SC"]

FAST_TAG = {"AI23-only": "ai23_only", "OSM-only": "osm_only", "AI23+OSM": "ai23_osm"}
OLD_FILE = {"AI23-only": "results_summary_ai23_only.csv",
            "OSM-only":  "results_summary_osm_only.csv",
            "AI23+OSM":  "results_summary_ai23_osm.csv"}

# FALLBACK: results_summary_ai23_only.csv / results_summary_osm_only.csv were
# lost when the "05_archive_deleted_20260716_0701" staging folder was deleted
# (not by this pipeline) on 17 Jul 2026, before the consolidated table could be
# built from the live CSVs. The MLP/GATv2 MEAN WMAPE for those two v1 (pre-fix)
# feature sets survive only as text in experiment_log.md / PROJECT_HANDOFF.md
# ("Experiment Set 1"). Per-fold numbers for those two cells are NOT
# reproducible/traceable to a CSV any more — flagged here and in
# experiment_log.md. AI23+OSM and all +SC columns are unaffected (their CSVs
# are intact).
ARCHIVED_MEANS = {
    ("MLP",   "AI23-only"): 0.8114,
    ("GATv2", "AI23-only"): 0.8241,
    ("MLP",   "OSM-only"):  0.8058,
    ("GATv2", "OSM-only"):  0.8259,
}


def load_mean(path, model, col=None):
    import os
    if not os.path.exists(path):
        if col is not None and (model, col) in ARCHIVED_MEANS:
            print(f"  [fallback] {model}/{col}: CSV missing ({path}), "
                  f"using archived text value from experiment_log.md")
            return ARCHIVED_MEANS[(model, col)]
        print(f"  [missing] {model}/{col}: {path} not found, no fallback available")
        return None
    df = pd.read_csv(path)
    row = df[df["model"] == model]
    if row.empty:
        return None
    return float(row["WMAPE_mean"].iloc[0])


def main():
    table = pd.DataFrame(index=ROWS, columns=COLS, dtype=float)

    for col in ["AI23-only", "OSM-only", "AI23+OSM"]:
        fast_path = f"results_summary_{FAST_TAG[col]}_fastbaselines.csv"
        old_path  = OLD_FILE[col]
        for model in ["HistAvg", "MLR", "RF", "XGBoost"]:
            table.loc[model, col] = load_mean(fast_path, model, col)
        for model in ["MLP", "GATv2"]:
            table.loc[model, col] = load_mean(old_path, model, col)

    sc_path = "results_summary_ai23_osm_sc.csv"
    for model in ["HistAvg", "MLR", "RF", "XGBoost", "MLP", "GATv2"]:
        table.loc[model, "AI23+OSM+SC"] = load_mean(sc_path, model)

    # Experiment Set 6: AI23's 8 features collapsed to 3 PCA components
    # (VIF up to 37.7 pairwise -> 94.4% variance retained in 3 components).
    # Same headline feature set otherwise (OSM + SC + coords unchanged).
    pca_path = "results_summary_pca_ai23_osm_sc.csv"
    for model in ["HistAvg", "MLR", "RF", "XGBoost", "MLP", "GATv2"]:
        table.loc[model, "PCA(AI23)+OSM+SC"] = load_mean(pca_path, model)

    # IDW is feature-independent (lat/lon + target only) — one run applies
    # identically to every column, same as HistAvg.
    idw_mean = load_mean("results_summary_idw.csv", "IDW")
    for col in COLS:
        table.loc["IDW", col] = idw_mean

    fair_path = "results_summary_gnn_fairness.csv"
    table.loc["GATv2-fairness (K=10)", "AI23+OSM+SC"] = load_mean(fair_path, "GATv2")

    # Experiment Set 7: functional-similarity edges (Zheng et al. 2025-style,
    # POI Pearson correlation, recalibrated rho>0.99 top-10 for this dataset).
    func_sim_path = "results_summary_func_sim.csv"
    table.loc["GATv2-func-sim", "AI23+OSM+SC"] = load_mean(func_sim_path, "GATv2")

    fusion_path = "results_summary_zheng_fusion.csv"
    table.loc["GATv2-Fusion (G1/G2/G3)", "AI23+OSM+SC"] = load_mean(fusion_path, "GATv2-Fusion")

    # Experiment Set 9: GCN baseline (attention vs. plain convolution, mirrors
    # Zheng et al.'s ASTGCN-vs-MF_STGAT logic).
    gcn_path = "results_summary_gcn.csv"
    table.loc["GCN", "AI23+OSM+SC"] = load_mean(gcn_path, "GCN")

    # Experiment Set 10: per-fold-tuned RF/XGBoost/Ridge (RandomizedSearchCV,
    # inner 3-fold CV on training data only -- see step4j_tuned_baselines.py).
    tuned_path = "results_summary_tuned.csv"
    for model in ["MLR-tuned", "RF-tuned", "XGBoost-tuned"]:
        table.loc[model, "AI23+OSM+SC"] = load_mean(tuned_path, model)

    table = table.round(4)
    table.to_csv("consolidated_results_table.csv")

    print("\nConsolidated results table (mean WMAPE, 33-fold leave-borough-out CV)\n")
    print("| Model | " + " | ".join(COLS) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for model in ROWS:
        cells = []
        for col in COLS:
            v = table.loc[model, col]
            cells.append(f"{v:.4f}" if pd.notna(v) else "—")
        print(f"| {model} | " + " | ".join(cells) + " |")

    print(f"\nSaved -> consolidated_results_table.csv")


if __name__ == "__main__":
    main()
