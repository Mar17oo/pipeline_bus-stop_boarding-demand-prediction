# RUNBOOK — reproduce everything, start to finish

Repo: `dissartation_2` — cold-start bus-stop boarding demand prediction
(GATv2 graph neural network vs. tabular baselines, leave-borough-out spatial
cross-validation, 17,943 London bus stops). **This is the single canonical
entry point for this repository** — if any other `.md` file here disagrees
with this one, this one is current. See §8 for what the other docs are and
why they still exist.

Last rewritten: 20 Aug 2026 (consolidated from `PROJECT_STRUCTURE.md`, which
was accurate as of 31 Jul 2026). One exploratory script (`step4o_tabular_ensemble.py`)
was built and run the same day but is deliberately kept out of the core
reproduction sequence — see §3.3a.

---

## 0. Quickstart — prerequisites and a from-scratch run

**Hardware/software this was built and timed on:** Windows 11, 13th Gen Intel
Core i5-13450HX (10 cores/16 threads), ~24GB RAM, **no GPU**
(`torch.cuda.is_available()` is `False` — every model, including the GNNs,
trains on CPU). Python 3.14.2. Nothing in the pipeline requires a GPU; it
will simply be slower on weaker CPUs.

```powershell
# 1. Install exact pinned dependencies (CPU-only PyTorch build)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Raw data goes in data/ (gitignored, ~453MB, not in this repo — see §4.0
#    for exactly which files are read). Everything downstream assumes it's there.

# 3. Full pipeline, in order (§3 has the full table; this is the minimal
#    path from raw data to the headline number):
python step1_aggregate_busto.py
python step2_join_coordinates.py
python step3_lsoa_features.py
python step3b_osm_features.py        # ~30 min, needs internet (Overpass API)
python step3c_add_scenic.py
python step3d_add_service_coverage.py
python -u step4_model.py --with-sc   # ~1.5-2h CPU-only -> the headline result
```

If you only want to inspect/extend results — every intermediate and result
CSV is already committed on disk. You don't need to re-run steps 1–4 unless
raw data changes; jump to whichever script you need from §3.

---

## 1. What this project is

Cold-start prediction of weekday boardings at 17,943 London bus stops with
**no historical demand data at test stops**: an inductive GATv2 graph
attention network over a KNN+route multigraph, compared against tabular
baselines (HistAvg, IDW, Ridge/MLR, Random Forest, XGBoost, MLP), evaluated
under 33-fold leave-borough-out spatial cross-validation (each fold removes
an entire London borough and predicts it from zero). Target: `log1p(total
weekday boardings)` from TfL's BUSTO 2023/24 survey. Metric: WMAPE
(macro-averaged across boroughs — see §5's note on what that means).
Headline finding: a plain MLP beats every graph variant tried, and the
reason is diagnosed, not just observed (see §6).

---

## 2. Repository layout

```
dissartation_2/
├── data/                             raw source files (~453MB, gitignored — Weekday-only, see §4.0)
├── step1_aggregate_busto.py ... step8_dissertation_figures.py   the pipeline (§3)
├── merge_results.py                  consolidates per-config result CSVs (§3.1, §4.2)
├── all_results_cv.csv, all_results_summary.csv   merged per-fold / summary results (§4.2)
├── reproduce_tables_and_stats.ipynb  fast (<1 min), no-retrain notebook that regenerates every
│                                     table/statistic from the committed CSVs — see below
├── dissertation_colab.ipynb          full pipeline, Colab-hosted (needs manual Drive upload of
│                                     stops_features_osm.csv + route_edges.csv — see its own header)
├── *.log                             run logs — evidence for the compute-cost claims (§7)
├── *.png                             figures
├── experiment_log.md                 chronological bug/experiment history (still current, read alongside this file)
├── diagnostics_report.md             read-only verification audit (still current)
├── COMMANDS_EXPLAINED.md             why each library call was used the way it was (still current, evergreen)
└── RUNBOOK.md                        this file
```

**Two ways to check the results, not just one:** `dissertation_colab.ipynb`
re-*runs* the 33-fold CV pipeline (hours, needs Colab + manually-uploaded
feature files). `reproduce_tables_and_stats.ipynb` does not retrain
anything — it loads the CSVs this repo already committed and regenerates
every table in the dissertation from them, plus several statistics
(feature ranges, target skewness, VIF, AI23×OSM cross-correlation, the five
Wilcoxon significance tests) that previously had no committed, re-runnable
script anywhere in this repo at all — verified only via numbers pasted into
`experiment_log.md`/`diagnostics_report.md`. It ends with a reconciliation
table that flags any cell where its fresh computation doesn't match the
dissertation text — as of 23 Aug 2026 that's the two VIF values and the
strongest AI23-internal correlation (see the notebook's own output for
current values; the dissertation text has not been corrected to match,
since this repo doesn't contain the manuscript source — see §8's history).

`.gitignore`d locally (present on disk, not on GitHub): `data/`,
`dissertation.tex`, `pdf_report/`, `Dissertation_Technical_Report.pdf`,
`data_and_software_availability_FILLED.tex` (manuscript-side, authored
elsewhere).

---

## 3. Pipeline scripts — purpose, category, status

### 3.1 Core pipeline (required to reproduce the frozen headline result from raw data)

| Script | Input → Output | Runtime | Notes |
|---|---|---|---|
| `step1_aggregate_busto.py` | `data/*BUSTO*.csv` → `busto_stop_level_boardings.csv` | ~2 min | Sums `Boardings` per stop across all weekday route/direction/quarter-hour rows. Also computes `n_route_dir_qhr_rows` here (later renamed to `service_coverage` in step3d). |
| `step2_join_coordinates.py` | + `data/Bus_Stops.csv` → `stops_with_coords.csv` | ~1 min | Joins on STOPCODE, prints a match-rate verdict. |
| `step3_lsoa_features.py` | + `data/access_*.csv` → `stops_features.csv` | ~2 min | Postcode → 2011 LSOA via postcodes.io, ONS 2021→2011 crosswalk, joins the 8 AI23 accessibility columns, filters to 33 London boroughs. Final: 17,943 stops. |
| `step3b_osm_features.py` | + live Overpass API → `stops_features_osm.csv` (+6 OSM POI cols) | ~30 min | The headline AI23+OSM+SC config depends on this file's output. |
| `step3c_add_scenic.py` | → +`poi_scenic` col | ~10 min | 7th Zheng-et-al.-style POI category. |
| `step3d_add_service_coverage.py` | + `busto_stop_level_boardings.csv` → +`service_coverage` col | <5 sec | Merges in the count computed back in step1. |
| `step4_model.py` | `stops_features*.csv` → `results_cv_<tag>.csv` / `results_summary_<tag>.csv` | ~1.5–2h/run (CPU) | **Main experiment script.** Trains HistAvg/IDW/MLR/RF/XGBoost/MLP/GATv2 under 33-fold LBO CV. Flags: `--ai23-only` / `--osm-only` / `--with-sc` / `--k10` / `--pca-ai23` / `--func-sim` / `--quick`. Output filename auto-tagged from active flags (`run_tag()`, top of file) — no manual renaming needed. |
| `merge_results.py` | 18 per-config `results_cv_<tag>.csv`/`results_summary_<tag>.csv` pairs → `all_results_cv.csv` / `all_results_summary.csv` | seconds | step6/step7 read the merged files, not the per-config ones directly — re-run this after generating a fresh `results_cv_<tag>.csv`. Row-count-asserts against the source files. |

### 3.2 Experiment scripts (each is one specific, self-contained experiment; not needed to rebuild the headline result, but each produced a reported number)

| Script | Experiment | What it does |
|---|---|---|
| `step4c_fast_baselines.py` | v4 supplement | HistAvg/MLR/RF/XGBoost only (no NN retrain) for the 3 pre-SC feature sets. |
| `step4d_idw_baseline.py` | Set 5 | Inverse-distance-weighting spatial baseline (Liu et al. 2017-style), feature-independent, ~1 sec. |
| `step4e_zheng_fusion.py` | Set 7b | Zheng-et-al.-style G1/G2/G3 dual-branch fusion GAT. Imports shared graph builders from `step4_model.py`. |
| `step4f_gated_mixing.py` | Set 8 (pre-registered) | Learned per-node gate mixing MLP + GATv2, 3 seeds. |
| `step4g_gated_analysis.py` | Set 8 stats | Wilcoxon + Holm-Bonferroni, alpha distribution, per-borough correlation. |
| `step4h_residual_correlation.py` | Set 8 follow-up | Retrains standalone MLP+GATv2 (seed 42), saves raw per-stop predictions to check for naive-ensembling artefacts. |
| `step4i_gcn_baseline.py` | Set 9 | `GCNModel` = `GATv2Model` with `GATv2Conv` swapped for `GCNConv`, isolating attention's specific contribution. |
| `step4j_tuned_baselines.py` | Set 10 | Per-fold `RandomizedSearchCV`/`RidgeCV` for RF/XGBoost/MLR, leakage-safe (inner CV on training fold only). |
| `step4k_rf_feature_importance.py` | Set 11 (post-freeze) | RF `.feature_importances_`, AI23-only. |
| `step4l_multiseed_mlp.py` | Set 12 (post-freeze, pre-registered D-10) | Standalone MLP, 5 seeds — establishes the MLP noise floor (~0.63pp run-to-run). |
| `step4m_alt_target_activity.py` | D-9 | Alternative target (boardings+alightings). **Confirmed unused, not part of the dissertation's current scope** — only a `_quick` (5-fold) result exists. |
| `step4n_gatv2_edge_attrs.py` | Set 13 (post-freeze, pre-registered D-8) | GATv2 with Haversine distance + bearing as edge attributes. Lost to plain GATv2 by 0.35pp. |
| `step5a_borough_map.py` | Choropleth | 4-panel HistAvg/RF/MLP/GATv2 map. **Reads `results_cv_ai23_osm.csv`** — the pre-SC feature set, not the AI23+OSM+SC headline; scoped to that config on purpose. |
| `step5b_vc_experiment.py` | V/C target (appendix/exploratory) | Predicts peak volume/capacity ratio instead of boardings. Its `StandardScaler` leakage bug is fixed in code, but the existing results (`results_cv_vc_ai23_osm.csv`) predate the fix and were not re-run (project is frozen). Caveat this table if used. Also has independently-drifted `GATv2Model`/`MLPModel` training regime (MSE not Huber, 1000 epochs not 500, `RF_TREES=300` not 150) — don't treat V/C numbers as on equal footing with the headline ones. |
| `step6_consolidated_table.py` | Ch.4/5 table | Assembles `consolidated_results_table.csv` from ~14 separate result CSVs. |
| `step7_borough_extract.py` | Discussion examples | Builds `borough_extract.csv`, prints Camden/Hillingdon/worst-3/closest-3 vs RF. |
| `step8_dissertation_figures.py` | Final figures | Generates the dissertation's figures from the frozen results. |

### 3.3 Utility / ambiguous

| Script | Status |
|---|---|
| `extract_route_edges.py` | **Possibly dead.** Precomputes `route_edges.csv`, but neither `step4_model.py` nor `step5b_vc_experiment.py` reads that file — both rebuild route edges in memory from `data/` directly every run. Kept for standalone route-network analysis if wanted. |

### 3.3a External / future work — NOT part of the current submission

| Script | Status |
|---|---|
| `step4o_tabular_ensemble.py` | Built and run 20 Aug 2026 as an exploratory check (pure-tabular MLR+RF+XGBoost+MLP blend, fitted and equal-weight variants, 3 seeds, on the headline AI23+OSM+SC feature set). Result: neither blend beats standalone MLP — fitted blend significantly worse (Wilcoxon p=0.011), equal-weight blend statistically tied (p=0.805). Deliberately **kept out of the core reproduction sequence** (§9) and out of the results table (§5) — this was an external test, not a step in rebuilding the headline result, and isn't part of the current dissertation's scope. Results are saved (`results_cv_tabular_ensemble.csv`, `results_summary_tabular_ensemble.csv`, `results_ensemble_weights.csv`, `tabular_ensemble_run.log`) in case it's picked up as future work; not written up further for now. Also surfaced a general finding worth keeping in mind: standalone MLP hit the full 500-epoch ceiling on every fold tested (33/33 for seed 42) rather than early-stopping, which likely applies to the headline MLP too, not just this script. |

### 3.4 Model-class duplication — audited, mostly a non-issue

`GATv2Model`/`MLPModel` are defined once, in `step4_model.py`, and imported
unmodified everywhere else (`step4e/f/g/h/i/l/m/n/o`) — the "variant"
classes (`GatedGATv2Model`, `GCNModel`, `GATv2EdgeAttrModel`, the Zheng
fusion model) are genuine new architectures built on top of the shared
originals, not copy-paste drift. The one exception is
`step5b_vc_experiment.py` (see §3.2) — its model *architecture* is
identical, but its training loop and scaler discipline have drifted.

---

## 4. Data / results files

### 4.0 Raw data (`data/`) — NOT everything in it is used

Every script that touches `data/` globs specifically for
`*Weekday*QUARTER HOUR*.csv` — the 7 Saturday/Sunday BUSTO files were
evaluated and explicitly rejected as a scope addition (not an oversight) and
deleted from local disk 31 Jul 2026. Everything else in `data/` (4 Weekday
BUSTO files, `Bus_Stops.csv`, 8 `access_*.csv`, the LSOA/borough lookup
files) is confirmed read by the pipeline.

**Where to get it** (added during the reproducibility audit, 23 Aug 2026 —
previously these links existed only in `data_and_software_availability_FILLED.tex`,
which is gitignored and so invisible to anyone who only clones this repo):

- **BUSTO boardings** (the 4 Weekday `*QUARTER HOUR*.csv` files): TfL
  Crowding Data portal, <https://crowding.data.tfl.gov.uk/> — BUSTO
  2023–2024 release, version 1.0 (see Sec.5's provenance note for why v1.0
  specifically). Licensed under TfL's Transport Data Licence (an OGL v2.0
  derivative with TfL-specific amendments):
  <https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service>.
- **AI23 accessibility indicators** (the 8 `access_*.csv` files): archived
  at the Urban Big Data Centre Data Hub,
  <https://data.ubdc.ac.uk/dataset/public-transport-accessibility-indicators-2022>,
  and mirrored on Zenodo, DOI
  [10.5281/zenodo.8037156](https://doi.org/10.5281/zenodo.8037156) (OGL
  v3.0). Described in Verduzco Torres & McArthur (2024), *Scientific Data*.
- **`Bus_Stops.csv`**: TfL's stop-location register (source not re-verified
  during this audit — check the TfL Open Data portal).
- **LSOA/borough lookup files**: ONS Open Geography Portal,
  <https://geoportal.statistics.gov.uk/> — the exact dataset page used was
  not recorded anywhere in this repo; confirm the specific 2011 LSOA-to-2021-LSOA
  and LSOA-to-borough lookup tables before re-downloading.
- **OpenStreetMap POIs**: not a static file — fetched live via the Overpass
  API through OSMnx in `step3b_osm_features.py`/`step3c_add_scenic.py`
  (© OpenStreetMap contributors, ODbL — <https://www.openstreetmap.org/copyright>).

### 4.1 Raw pipeline inputs/intermediates (keep — needed to rebuild from scratch)
`busto_stop_level_boardings.csv`, `stops_with_coords.csv`,
`stops_features.csv`, `stops_features_osm.csv`, `stops_features_vc.csv`,
`route_edges.csv`.

### 4.2 Results — merged 31 Jul 2026 from ~56 per-config files down to 2

**`all_results_cv.csv`** / **`all_results_summary.csv`** hold every per-fold
/ summary result sharing the standard schema
(`borough,n_test,model,WMAPE,RMSE,MAE` / `model,WMAPE_mean,...`), tagged by
a `config` column (19 configs as of 23 Aug 2026 — see the merge script for
the full list). Built and row-count-verified by `merge_results.py`.

**Gap found and fixed 23 Aug 2026:** `merge_results.py`'s `CONFIGS` list was
last edited 5 Aug 2026, before the `ai23_sc` config existed (added 22 Aug
2026 — see §5's "AI23+SC column added" note). So until this fix,
`all_results_cv.csv`/`all_results_summary.csv` silently had no AI23+SC
rows at all, despite that column appearing in the dissertation's headline
results table — anyone trying to reproduce that column from the merged
files alone would have found it missing. `ai23_sc` has been added to
`CONFIGS`; **re-run `python merge_results.py`** to pick it up (it will not
happen automatically — this file only fixes the script, not its output).
Until you do, `results_cv_ai23_sc.csv` / `results_summary_ai23_sc.csv`
remain the only source for that column (as they still are for
`reproduce_tables_and_stats.ipynb`, see below).

**Kept as separate files** (different schema): `results_cv_colab.csv` /
`results_summary_colab.csv`, `results_cv_multiseed_mlp.csv` /
`results_summary_multiseed_mlp.csv`, `results_cv_gated_seed{42,142,242}.csv`,
`results_cv_gated_all_seeds.csv`, `results_rf_feature_importance[_perfold].csv`,
`results_tuned_hyperparams.csv`, `results_gated_alpha_all_seeds.csv` + per-seed
variants, `borough_extract.csv`, `borough_extract_gated.csv`,
`consolidated_results_table.csv`, `residual_correlation_by_borough.csv`,
`results_residual_correlation.csv`, and (added 20 Aug 2026)
`results_cv_tabular_ensemble.csv` / `results_summary_tabular_ensemble.csv` /
`results_ensemble_weights.csv`.

---

## 5. Results (frozen headline, 33-fold leave-borough-out CV, WMAPE — lower is better)

| Model | AI23 | OSM | AI23+OSM | AI23+SC | AI23+OSM+SC |
|---|---|---|---|---|---|
| HistAvg | 1.0822 | 1.0822 | 1.0822 | 1.0822 | 1.0822 |
| IDW (no features) | 0.8487 | 0.8487 | 0.8487 | 0.8487 | 0.8487 |
| MLR (Ridge) | 0.8120 | 0.8078 | 0.7982 | 0.6420 | 0.6404 |
| RF (150 trees) | 0.8128 | 0.8064 | 0.7970 | 0.6452 | 0.6428 (tuned: 0.6339) |
| XGBoost (300) | 0.8207 | 0.8084 | 0.8075 | 0.6584 | 0.6437 |
| **MLP** | — | — | — | 0.6365 | **0.6311** |
| GATv2 (plain) | — | — | — | 0.7262 | 0.7187 |
| GATv2 K=10 | — | — | — | — | 0.7372 |
| GATv2 func-sim edges | — | — | — | — | 0.7352 |
| GATv2-Fusion | — | — | — | — | 0.7070 |
| GCN | — | — | — | — | 0.7006 |
| Gated (MLP+GATv2 blend, 3 seeds) | — | — | — | — | 0.6277 (not confirmed beyond MLP's own 0.63pp seed noise) |

**Provenance note, decided 2026-08-22:** MLP/GATv2 are omitted (not
footnoted-with-caveat, deliberately dropped) for AI23, OSM, and AI23+OSM —
those runs predate the residual-skip-connection and validation-loss-leakage
fixes (**v1**: no skip connection, MSE loss not Huber, DROPOUT=0.3/LR=1e-3
not 0.15/5e-4, and critically still carrying Bug 3) and were never
regenerated under the current, bug-fixed `step4_model.py`. A rerun to
replace them was considered and explicitly declined (more compute for a
result already expected not to change the conclusion) — dropping the cells
was judged cleaner than reporting a footnoted pre-fix number. MLR/RF/XGBoost
in those same three columns ARE current-protocol (via
`step4c_fast_baselines.py`'s same-seed/same-split supplement, documented in
its own docstring) — only MLP/GATv2 are affected, and only in these three
pre-SC columns. The actual headline result (AI23+OSM+SC, where MLP beats
GATv2) was run fresh under the fixed code from the start, confirmed
separately, and is completely unaffected by any of this.

**Not yet reconciled: `fig2_grouped_feature_sets.png`** (built by
`step8_dissertation_figures.py`) still shows these same MLP/GATv2 cells,
hatched, with a caveat footnote, rather than omitted — deliberately left
as-is since that figure's job is a full grid overview ("here's exactly what
does and doesn't exist"), a different purpose than this table's clean
headline summary. Revisit if that inconsistency ever needs resolving.

**AI23+SC column added 2026-08-22** (`step4_model.py --ai23-only --with-sc`,
run fresh under the current fixed code, `results_cv_ai23_sc.csv`) — the file
selection had its own bug (`--ai23-only` unconditionally loaded
`stops_features.csv`, which never got `service_coverage` merged into it, so
`--with-sc` silently did nothing) fixed the same day; see the comment above
`DATA_FILE` in `step4_model.py` for the fix. Notably, AI23+SC alone comes
within 0.16–0.75pp of the full AI23+OSM+SC headline for every model except
XGBoost (1.47pp) — `service_coverage` accounts for nearly all of the
0.80→0.64 improvement, OSM's marginal contribution on top of it is small.

Best confirmed: MLP (0.6311) and tuned RF (0.6339) are statistically tied
(p=0.292). Every GNN variant loses to every tabular model at p<0.00001
(Wilcoxon, paired by borough). Mechanism: cross-borough target assortativity
is lower than within-borough (0.335 vs 0.509) — a held-out stop's nearest
neighbours carry systematically less transferable signal, so more graph
expressiveness makes things worse, not better (a finding, not a modelling
failure).

**Note on WMAPE aggregation** (see `diagnostics_report.md` §D3): the
headline number is a **macro-average** — the unweighted arithmetic mean of
33 independently-computed per-borough WMAPEs, not a single pooled WMAPE over
all 17,943 stops. Every borough contributes equally regardless of stop count
(City of London, n=101, is weighted identically to Bromley, n=1,179) — worth
stating explicitly in the methods chapter.

---

## 6. Why GATv2 underperforms tabular models — root cause

Consistent with Grinsztajn et al. (2022, NeurIPS) and Shwartz-Ziv & Tishby
(2022): GNNs don't reliably beat tabular methods when graph structure is
noisy relative to feature signal. Here specifically: in leave-borough-out
CV, a held-out borough's K=5 nearest training neighbours are systematically
from adjacent-but-different boroughs, so GATv2's message passing aggregates
contaminated cross-borough signal that RF/MLP/MLR — which use features
directly, no aggregation step — are immune to. This is a genuine, diagnosed
finding, not an unexplained negative result: 7 independent graph-architecture
interventions (K=10, PCA-AI23, func-sim edges, GATv2-Fusion, GCN,
gated-mixing, edge-attrs) have all lost to tabular baselines at p<0.00001.

**Assortativity figure corrected 23 Aug 2026 (reproducibility audit).** This
section previously read "cross-borough target assortativity is lower than
within-borough (0.335 vs 0.509)". Those numbers came from the first D6
diagnostic session (`diagnostics_report.md`, 19–20 Jul 2026) and are stale:
the current, dissertation-reported figures — confirmed directly against the
text rendered on `fig7_assortativity_scatter.png`, produced by
`step8b_assortativity_figure.py` — are **within-borough r=0.48 (n=128,929)
vs across-borough r=0.40 (n=6,816)**, a 16% relative drop, not the 34%
implied by the old 0.335/0.509 pair. The underlying mechanism claim (a
held-out stop's nearest neighbours carry systematically less transferable
signal) is unchanged; only the magnitude quoted here was wrong. Why the
D6-era numbers differ from the current script's output has not been
investigated — `step8b_assortativity_figure.py`'s own docstring still cites
the old D6 figures (128,685 total edges, r=0.5088/0.3354) as its expected
sanity-check values, which the script's actual current output no longer
matches either. Worth a closer look before citing D6 for anything else.

---

## 7. Hardware & software invested

Local Windows 11 machine, 13th Gen Intel Core i5-13450HX (10 cores/16
threads), ~24GB RAM, no GPU. Software per `requirements.txt`. Total logged
compute across all run logs through 31 Jul 2026: ~1,999 minutes (33.3
hours). Mean per-fold time: tuned tabular baselines ~39.1s/fold; a single
GATv2 model ~171.1s/fold; a single MLP fold in `step4o_tabular_ensemble.py`
~124-184s/fold (MLP hit the full 500-epoch ceiling rather than early
stopping in both folds sampled during dev — see that script's docstring).
Neural models are stochastic and PyTorch determinism flags were never
enabled — quantified run-to-run noise is ~0.04pp (within a controlled
5-seed batch) vs. ~0.63pp observed across two isolated sessions at
different times.

---

## 8. About the other `.md` files in this repo

This repo accumulated several planning/handoff documents over the project's
life. As of this rewrite (20 Aug 2026), their status is:

- **`experiment_log.md`** — still current, still the authoritative
  chronological record of every bug and experiment. Read alongside this file.
- **`diagnostics_report.md`** — still current, read-only verification audit
  (D1-D6, V1-V11). Referenced throughout this file.
- **`COMMANDS_EXPLAINED.md`** — still current, evergreen "why this library
  call, not that one" reference. Not affected by which experiments are frozen.
- **`PROJECT_STRUCTURE.md`, `PROJECT_HANDOFF.md`, `DISSERTATION_BRIEF.md`** —
  **superseded by this file.** They were accurate at various earlier points
  (`PROJECT_STRUCTURE.md` as recently as 31 Jul 2026) but drifted — e.g.
  instructing a manual `copy results_cv_multigraph.csv ...` step that
  `step4_model.py`'s auto-tagging fix made both unnecessary and broken. Kept
  on disk for history, not for instructions — if you're about to run a
  command from one of them, check it against this file first.

**Correction (23 Aug 2026 audit):** this section previously claimed
`build_excel.py` "was deleted" — false. It is tracked in git, present on
disk, and was run as recently as 20 Aug 2026 (produces
`supervisor_comparison_results.xlsx`, a table for comparing results against
supervisor feedback — not part of the dissertation pipeline itself). It
needs `openpyxl`, now added to `requirements.txt`. It's intentionally still
absent from the pipeline table (§3) and command reference (§9) below, same
treatment as `step4o_tabular_ensemble.py` (§3.3a) — it's a supplementary,
non-core script, not a broken/missing one.

---

## 9. Full command reference — every script, in the order you'd actually run them

```powershell
cd dissartation_2

# 1. Build features from raw data (only needed once; already done — outputs are on disk)
python step1_aggregate_busto.py
python step2_join_coordinates.py
python step3_lsoa_features.py
python step3b_osm_features.py      # ~30 min, needs internet (Overpass API)
python step3c_add_scenic.py
python step3d_add_service_coverage.py

# 2. Headline + ablation runs (each ~1.5-2h CPU-only; output filenames auto-tagged)
python -u step4_model.py --ai23-only            # -> results_cv_ai23_only.csv
python -u step4_model.py --osm-only             # -> results_cv_osm_only.csv
python -u step4_model.py                        # -> results_cv_ai23_osm.csv
python -u step4_model.py --with-sc              # -> results_cv_ai23_osm_sc.csv   (HEADLINE)
python -u step4_model.py --osm-only --with-sc   # -> results_cv_osm_sc.csv
python -u step4_model.py --with-sc --k10        # -> results_cv_gnn_fairness.csv
python -u step4_model.py --with-sc --pca-ai23   # -> results_cv_pca_ai23_osm_sc.csv
python -u step4_model.py --with-sc --func-sim   # -> results_cv_func_sim.csv
# Add --quick to any of the above for a 5-fold smoke test (~15-25 min)

# 3. Standalone experiment scripts (each self-contained, see §3.2)
python step4d_idw_baseline.py
python step4e_zheng_fusion.py
python step4f_gated_mixing.py && python step4g_gated_analysis.py && python step4h_residual_correlation.py
python step4i_gcn_baseline.py
python step4j_tuned_baselines.py
python step4k_rf_feature_importance.py
python step4l_multiseed_mlp.py
python step4n_gatv2_edge_attrs.py
# step4o_tabular_ensemble.py is NOT part of this sequence -- external/future-work
# exploratory check, not a step in reproducing the headline result. See §3.3a.

# 4. Merge the per-config result CSVs into the 2 consolidated files
python merge_results.py

# 5. Consolidation and figures
python step6_consolidated_table.py
python step7_borough_extract.py results_cv_ai23_osm_sc.csv
python step5a_borough_map.py        # needs internet (ONS boundaries)
python step8_dissertation_figures.py

# Optional, exploratory / appendix-only:
python step5b_vc_experiment.py      # V/C target -- note the leakage-bug caveat in §3.2
```

Total wall-clock for a full from-scratch reproduction: roughly 33+ hours of
CPU time, spread across the scripts above.
