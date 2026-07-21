# RUNBOOK — Full file inventory + commands to reproduce everything
Generated 16 Jul 2026. Companion to PROJECT_HANDOFF.md (narrative/claims) and
experiment_log.md (chronological bug/experiment history). This file is the
"what's on disk and how to (re)run it" reference.

---

## 1. Documents / write-up

| File | What it is |
|---|---|
| `PROJECT_HANDOFF.md` | Master handoff: coding-chat prompt (frozen Tier1/Tier2 plan) + writing-chat handoff (3 findings, claims discipline, lit spine). Paste this into any new chat. |
| `DISSERTATION_BRIEF.md` | Early planning-phase brief. Superseded by the "WRITING HANDOFF" section of PROJECT_HANDOFF.md — kept for history only. |
| `dissertation.tex` | Main LaTeX draft (Ch.1 drafted, skeleton for the rest). |
| `dissertation_colab.ipynb` | Colab notebook mirror of the pipeline (useful if you ever need GPU). |
| `experiment_log.md` | Chronological log: 8 bugs found/fixed (v1→v2→v3) + all experiment results. **Needs a v4 entry** for today's MLR/XGBoost additions, the K=10 fairness run, and Bug 9 (see §5). |
| `RUNBOOK.md` | This file. |

## 2. Raw data (`data/` folder — do not edit, ~1GB)

| File(s) | Used by |
|---|---|
| `2023_24 Weekday TOTAL DEMAND BY ROUTE BY QUARTER HOUR *.csv` (4 files) | step1 (aggregation), step4_model.py `build_route_edges()` (reads these **every run** to build route edges — this is why each `step4_model.py` invocation has a ~30s–2min startup cost) |
| `2023_24 Saturday / Sunday ... .csv` (7 files) | **Present but unused** — Saturday/Sunday data was evaluated and explicitly rejected as a scope addition. Do not wire in without reopening that decision. |
| `Bus_Stops.csv` | step2 (stop coordinates) |
| `access_*.csv` (8 files) | step3 (AI23 LSOA accessibility features) |
| `LSOA11_WD21_LAD21_EW_LU_V2.xlsx`, `lsoa21_to_lsoa11_lookup.csv` | step3 (LSOA/borough lookups) |

## 3. Pipeline scripts (run in order, only if rebuilding from raw data)

| Script | Input → Output | Time |
|---|---|---|
| `step1_aggregate_busto.py` | data/ BUSTO CSVs → `busto_stop_level_boardings.csv` | ~2 min |
| `step2_join_coordinates.py` | busto + `data/Bus_Stops.csv` → `stops_with_coords.csv` | ~1 min |
| `step3_lsoa_features.py` | stops_with_coords + `data/access_*.csv` → `stops_features.csv` | ~2 min |
| `step3b_osm_features.py` | stops_features → `stops_features_osm.csv` (+OSM POI cols, via osmnx/live API) | ~30 min |
| `step3c_add_scenic.py` | stops_features_osm → +poi_scenic col | ~10 min |
| `step3d_add_service_coverage.py` | stops_features_osm + busto → +service_coverage col | <5 sec |
| `extract_route_edges.py` | data/ BUSTO CSVs → `route_edges.csv` (43,220 edges) | analysis/cache only — **not read by step4_model.py**, which rebuilds route edges from `data/` directly every run. Safe to ignore unless doing separate route-network analysis. |

**You will not normally re-run any of these** — steps 1–3d are already done and their outputs are on disk. Only rerun if raw data changes.

## 4. Modelling scripts (the ones you'll actually reuse)

### `step4_model.py` — main experiment script (UPDATED 16 Jul: +MLR, +XGBoost, +`--k10`)
```powershell
cd "C:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"

# Full 33-fold runs (each ~1.5-2h CPU-only). Output ALWAYS goes to
# results_cv_multigraph.csv / results_summary_multigraph.csv — rename/copy
# immediately after each run or the next run overwrites it.
python -u step4_model.py --ai23-only          # AI23 only (10 feat)
python -u step4_model.py --osm-only           # OSM only (8 feat)
python -u step4_model.py                      # AI23+OSM (default, 16 feat)
python -u step4_model.py --with-sc            # AI23+OSM+SC (headline, 17 feat)
python -u step4_model.py --osm-only --with-sc # OSM+SC (9 feat)
python -u step4_model.py --with-sc --k10      # GNN-fairness check: K=10 not K=5
python -u step4_model.py --with-sc --pca-ai23 # PCA(AI23, 3 comp.)+OSM+SC — multicollinearity check

# Add --quick to any of the above for a 5-fold smoke test (~15-25 min)
```
Rows per run: HistAvg, IDW, MLR, RF, XGBoost, MLP, GATv2 × 33 boroughs.

### `step4d_idw_baseline.py` — IDW spatial-interpolation baseline (~1 sec)
```powershell
python -u step4d_idw_baseline.py
```
Classical geostatistical cold-start baseline (Liu et al. 2017, NYC Citi
Bike-style: inverse-distance weighting over the k=20 nearest training stops,
lat/lon only, no AI23/OSM/SC features). Feature-independent — one run
applies to all 4 columns of the consolidated table, same as HistAvg. Also
added permanently to `step4_model.py` (runs automatically in every future
full CV as the "IDW" row) — this script exists only to get the number
quickly without a 2h GATv2 retrain.

### `step4c_fast_baselines.py` — HistAvg/MLR/RF/XGBoost only, no GATv2/MLP (~1 min total)
```powershell
python -u step4c_fast_baselines.py
```
Only needed once (already run — see §6). Rerun only if `stops_features.csv` /
`stops_features_osm.csv` change. Produces the MLR/XGBoost numbers for the 3
pre-fix feature sets (AI23-only, OSM-only, AI23+OSM) without redoing the
expensive (and version-mismatched) GATv2/MLP training — see the caveat this
script's own docstring explains.

### `step5a_borough_map.py` — choropleth (~1 min, needs internet for ONS boundaries)
```powershell
python -u step5a_borough_map.py
```
Fixed today (Bug 9, see §5) — previously only rendered 1/33 boroughs.

### `step5b_vc_experiment.py` — optional V/C (overcrowding) target experiment (~1.5-2h)
```powershell
python -u step5b_vc_experiment.py
```

### `run_tier1_overnight.sh` — driver that chains the pending Tier-1 runs
```bash
bash run_tier1_overnight.sh 2>&1 | tee tier1_overnight.log
```
Runs, in order: OSM+SC → AI23+OSM+SC → GNN-fairness(K=10) → V/C, copying each
`results_*_multigraph.csv` to its proper name before the next run overwrites
it. Continues past a failed step rather than aborting the queue. **This is
what's running right now** (see §6 for live status).

### `step6_consolidated_table.py` — Chapter 4 table (run AFTER the overnight queue finishes)
```powershell
python step6_consolidated_table.py
```
Assembles `consolidated_results_table.csv`: rows = HistAvg/MLR/RF/XGBoost/MLP/
GATv2/GATv2-fairness, cols = AI23-only/OSM-only/AI23+OSM/AI23+OSM+SC.

### `step7_borough_extract.py` — Camden/Hillingdon + worst-3/closest-3 (run AFTER headline run finishes)
```powershell
python step7_borough_extract.py results_cv_ai23_osm_sc.csv
```

### `build_excel.py` — regenerates the supervisor Excel comparison
```powershell
python build_excel.py
```
Rerun once all new results are in, to refresh `supervisor_comparison_results.xlsx`.

## 5. Bugs found today (logged in experiment_log.md as Bug 6)

**Bug 6 — ONS borough-boundary query only matched "City of London" (FIXED)**
`step5a_borough_map.py`'s WHERE clause was `LAD23NM LIKE '%London%' OR LAD23NM='City of London'`
— but borough names like "Camden" or "Hackney" don't contain the literal word
"London", so only 1 of 33 boundaries was ever fetched (confirmed: log showed
"Merged: 1 boroughs matched"). Fixed to filter on ONS code prefix `LAD23CD LIKE 'E09%'`
(the 32 London boroughs + City of London = 33 LADs). Regenerated map now shows
all 33 boroughs matching the CV fold count.

## 6. Current state as of 17 Jul 2026 — Tier 1 + Tier 2 + IDW addition all DONE

**Everything in the frozen plan is complete:**
- `service_coverage` merged into `stops_features_osm.csv`.
- `step4_model.py`: 7 models now (HistAvg, IDW, MLR, RF, XGBoost, MLP, GATv2), `--k10` flag.
- All 4 overnight Tier-1 runs finished cleanly (OSM+SC, AI23+OSM+SC headline,
  K=10 fairness, optional V/C) — see `tier1_overnight.log`.
- Bug 6 (ONS map query) fixed, `borough_wmape_map.png` shows all 33 boroughs.
- `05_archive_deleted_20260716_0701/` (containing the v1 AI23-only/OSM-only
  per-fold CSVs) was restored by the student on 17 Jul and copied back to
  root — `consolidated_results_table.csv` now reads live CSVs everywhere,
  no fallback needed.
- IDW spatial-interpolation baseline (Liu et al. 2017-style) added and run
  (`step4d_idw_baseline.py`) — WMAPE 0.8487, feature-independent.
- `consolidated_results_table.csv` and `borough_extract.csv` built from the
  headline run.
- `experiment_log.md` fully updated: version-history table, Bug 6, the
  restored-archive note, Experiment Sets 3-6, revised Conclusions.
- Dataset audit found severe AI23 multicollinearity (VIF up to 37.7). Tested
  via PCA(8→3 components, 94.4% variance) + full 7-model rerun
  (`--pca-ai23`) — result: <0.4pp change on every model, a null result.
  Raw 8 AI23 features kept (more interpretable, PCA buys nothing).

**Headline numbers (AI23+OSM+SC, 33-fold leave-borough-out):**
HistAvg=1.0822, IDW=0.8487, MLR=0.6404, RF=0.6428, XGBoost=0.6437,
**MLP=0.6311 (best)**, GATv2=0.7187, GATv2 K=10 fairness=0.7372,
GATv2 PCA(AI23) variant=0.7171 (no meaningful change).

**Experiment Set 7 (17 Jul, exploratory — not part of frozen headline):**
functional-similarity edges (Zheng et al. 2025-style, POI Pearson
correlation, recalibrated ρ>0.99 top-10 since the paper's literal ρ>0.8
threshold gives a near-complete 48M-edge graph on this data). Two variants:
(a) added into the same single graph as KNN+route: GATv2=0.7352, **worse**
than plain (0.7187). (b) Zheng's actual G1/G2/G3 dual-branch fusion
architecture (separate GAT per graph + skip connection):
GATv2-Fusion=**0.7070**, **better** than plain by 1.17pp — the single most
effective graph intervention tried — but still 7.59pp behind MLP. Student's
call whether to promote either into the dissertation's main comparison.

**Statistical significance (18 Jul, Wilcoxon signed-rank across 33 folds):**
GATv2 loses to MLP/RF/MLR at p<0.00001 in every variant tested (plain, K=10,
PCA, func-sim, Fusion) — the core finding is statistically solid. MLP beats
RF/MLR significantly (p<0.003) but NOT XGBoost (p=0.06, borderline) — say
"MLP and XGBoost tied for best," not "MLP is unambiguously best." The
Fusion-vs-plain-GATv2 improvement (Experiment Set 7) is NOT significant
(p=0.126) — report as a trend, not a confirmed effect. Full table in
experiment_log.md's "Statistical Significance Testing" section.

**Known caveat carried forward:** the Camden/Hillingdon anecdote in earlier
drafts (near-tie in Camden, worst gap in peripheral Hillingdon) does not
hold on the v4 headline data, AND the "which boroughs are hardest" picture
changes again under GATv2-Fusion (Hillingdon improves to a near-tie with RF,
Camden gets worse and becomes the single largest gap) — re-read
`borough_extract.csv` **for the specific run being discussed** before
writing Ch.5, don't reuse a borough example across different model configs.

**Experiment Set 8 (18-19 Jul, pre-registered) + follow-up (19 Jul):** a
learned per-node gate mixing MLP and GATv2 beat MLP significantly (0.6277
vs 0.6311, p=0.007, 3 seeds) but with a non-adaptive alpha (~0.52 everywhere,
uncorrelated with the per-borough gap, r=-0.026). A follow-up ruled out
generic ensembling (residuals highly correlated r=0.90; naive 50/50 blend is
WORSE than MLP, not better) but found MLP's own run-to-run training variance
(0.63pp between two seed-42 runs) exceeds the claimed effect (0.35pp) —
report as suggestive, not settled. Full reasoning in experiment_log.md's
Experiment Set 8 + follow-up sections.

**Comprehensive technical report generated (19 Jul):**
`Dissertation_Technical_Report.pdf` (17 pages, LaTeX source in `pdf_report/
report.tex`, compiled with the local TinyTeX install) — covers the full
pipeline, all model architectures with code, every experiment set's results
and statistical tests, the mechanism analysis, file inventory, and
write-up recommendations. Meant to be read alongside the codebase, not as a
replacement for `experiment_log.md`.

**Still open / your call:**
- `supervisor_comparison_results.xlsx` — `build_excel.py` is currently broken
  (points at `results_cv.csv`/`results_summary.csv`, deleted before this
  session started). Not part of the Tier 1/2 checklist; rewrite only if asked.
- `reorganise_folders.ps1` — not yet run, root folder still flat. Safe to run
  now that all experiments are finished.
- Leftover, safe to delete: `results_cv_multigraph_quick.csv`,
  `results_summary_multigraph_quick.csv` (from an early `--quick` smoke test).
- `dissertation_colab.ipynb` — **updated and smoke-tested 17 Jul 2026**: now
  matches the current pipeline (7 models incl. MLR/XGBoost/IDW, OSM+SC
  features, skip-connection GATv2, Huber loss, stratified val, GPU support).
  To use: upload `stops_features_osm.csv` and `route_edges.csv` to
  `MyDrive/dissertation/` in Google Drive, open the notebook in Colab, run
  cells top to bottom. Verified end-to-end locally (2-borough, truncated-
  epoch smoke test) before being called done — see §4 below for the
  equivalent local command if you want to rerun that check yourself.
