"""
STEP 4G — Full analysis of the pre-registered gated-mixing experiment
=========================================================================
Consumes results_cv_gated_all_seeds.csv and results_gated_alpha_all_seeds.csv
(step4f_gated_mixing.py output). Computes everything the pre-registration
required (see step4f's docstring for the interpretation rules applied here).

Run with: python step4g_gated_analysis.py
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, pearsonr

cv = pd.read_csv("results_cv_gated_all_seeds.csv")
alpha = pd.read_csv("results_gated_alpha_all_seeds.csv")

headline = pd.read_csv("results_cv_ai23_osm_sc.csv")
mlp_by_borough = headline[headline["model"] == "MLP"].set_index("borough")["WMAPE"]
gat_by_borough = headline[headline["model"] == "GATv2"].set_index("borough")["WMAPE"]
MLP_MEAN, GAT_MEAN = 0.6311, 0.7187

print("="*78)
print("1. MEAN WMAPE -- across 33 folds, and across folds AND seeds")
print("="*78)
per_seed_mean = cv.groupby("seed")["WMAPE"].mean()
print("Per-seed mean WMAPE (33-fold average within each seed):")
print(per_seed_mean.to_string())
print(f"\nAcross-seed mean of per-seed means: {per_seed_mean.mean():.4f}  "
      f"(sd across seeds: {per_seed_mean.std():.4f})")
print(f"Overall mean across all {len(cv)} (seed x borough) rows: {cv['WMAPE'].mean():.4f}  "
      f"(sd across all rows: {cv['WMAPE'].std():.4f})")

# Seed-averaged per-borough WMAPE -- the primary series used for all paired tests below
borough_seed_avg = cv.groupby("borough")["WMAPE"].mean()

print()
print("="*78)
print("2. WILCOXON SIGNED-RANK vs MLP and vs GATv2 plain, paired by borough")
print("   (using the seed-averaged per-borough WMAPE, n=33)")
print("="*78)
common_boroughs = sorted(set(borough_seed_avg.index) & set(mlp_by_borough.index))
gated_vec = borough_seed_avg.loc[common_boroughs].values
mlp_vec   = mlp_by_borough.loc[common_boroughs].values
gat_vec   = gat_by_borough.loc[common_boroughs].values

stat_mlp, p_mlp = wilcoxon(gated_vec, mlp_vec)
stat_gat, p_gat = wilcoxon(gated_vec, gat_vec)
print(f"Gated vs MLP:        mean(Gated-MLP)={np.mean(gated_vec-mlp_vec):+.4f}  "
      f"W={stat_mlp:.1f}  p={p_mlp:.5f}")
print(f"Gated vs GATv2 plain: mean(Gated-GATv2)={np.mean(gated_vec-gat_vec):+.4f}  "
      f"W={stat_gat:.1f}  p={p_gat:.5f}")

# Holm-Bonferroni across these 2 comparisons
pvals = sorted([("vs_MLP", p_mlp), ("vs_GATv2", p_gat)], key=lambda t: t[1])
print("\nHolm-Bonferroni correction (family of 2 tests, alpha=0.05):")
reject_all = True
for rank, (name, p) in enumerate(pvals, start=1):
    thresh = 0.05 / (2 - rank + 1)
    sig = p < thresh
    reject_all = reject_all and sig
    print(f"  {name}: p={p:.5f}  threshold={thresh:.4f}  {'SIGNIFICANT' if sig else 'not significant'}")
    if not sig:
        break  # Holm-Bonferroni stops at first non-rejection

print()
print("="*78)
print("3. DISTRIBUTION OF LEARNED ALPHA (per test-stop, pooled across seeds)")
print("="*78)
a = alpha["alpha"]
print(f"Mean:   {a.mean():.4f}")
print(f"Median: {a.median():.4f}")
print(f"IQR:    [{a.quantile(0.25):.4f}, {a.quantile(0.75):.4f}]")
print(f"Fraction alpha > 0.9 (effectively ignoring neighbours): {(a > 0.9).mean()*100:.2f}%")
print(f"Fraction alpha < 0.1 (effectively pure message passing): {(a < 0.1).mean()*100:.2f}%")

print()
print("="*78)
print("4. MEAN ALPHA PER BOROUGH vs WMAPE GAP (Gated - MLP) PER BOROUGH")
print("="*78)
alpha_by_borough = alpha.groupby("borough")["alpha"].mean()
gap_vs_mlp = borough_seed_avg - mlp_by_borough
merged = pd.DataFrame({"mean_alpha": alpha_by_borough, "gap_vs_mlp": gap_vs_mlp}).dropna()
r, p = pearsonr(merged["mean_alpha"], merged["gap_vs_mlp"])
print(f"corr(mean_alpha_per_borough, WMAPE_gap_vs_MLP): Pearson r={r:.3f}  p={p:.4f}")
print()
print(merged.sort_values("gap_vs_mlp", ascending=False).to_string())

print()
print("="*78)
print("5. PER-BOROUGH WMAPE TABLE (seed-averaged Gated vs MLP vs GATv2 plain vs RF)")
print("="*78)
rf_by_borough = headline[headline["model"] == "RF"].set_index("borough")["WMAPE"]
table = pd.DataFrame({
    "Gated": borough_seed_avg,
    "MLP": mlp_by_borough,
    "GATv2": gat_by_borough,
    "RF": rf_by_borough,
}).dropna()
table["gap_gated_mlp"] = table["Gated"] - table["MLP"]
table = table.sort_values("gap_gated_mlp", ascending=False)
table.to_csv("borough_extract_gated.csv")
print(table.to_string())
print("\nSaved -> borough_extract_gated.csv")

print()
print("="*78)
print("SUMMARY / PRE-REGISTERED INTERPRETATION CHECK")
print("="*78)
overall_mean = per_seed_mean.mean()
print(f"Gated mean WMAPE: {overall_mean:.4f}  |  MLP: {MLP_MEAN}  |  GATv2 plain: {GAT_MEAN}")
print(f"Mean alpha: {a.mean():.4f}")
if a.mean() > 0.8 and abs(overall_mean - MLP_MEAN) < 0.01:
    print(">> HIGH alpha, WMAPE approaches MLP: model learned to switch the graph off.")
    print(">> This CONFIRMS Finding 3 from inside the architecture.")
elif overall_mean < GAT_MEAN and overall_mean > MLP_MEAN:
    print(">> MODERATE alpha, WMAPE beats GATv2-Fusion(0.7070) but not MLP: partial gain.")
    print(">> Report as bounded, honest partial gain -- graph info weakly useful, net-negative when forced.")
elif overall_mean < MLP_MEAN and reject_all:
    print(">> Gated beats MLP with significance surviving Holm-Bonferroni.")
    print(">> Finding 3 must be REWRITTEN: loss was architectural (forced mixing), not evidential.")
else:
    print(">> Does not cleanly match any pre-registered bucket -- report numbers as-is, judge case by case.")
