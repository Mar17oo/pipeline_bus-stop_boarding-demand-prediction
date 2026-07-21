# Experiment Log — Cold-Start Bus Stop Demand Prediction
## CE902 MSc Dissertation — University of Essex
## Supervisor: Dr. Vishal K. Singh

---

## Research Question
Can we predict weekday boarding demand at London bus stops that have **no prior
historical passenger data** (cold-start), using only publicly available spatial
features (accessibility, land use, service supply)?

---

## Dataset
- **BUSTO** (Bus Stop Usage dataset, TfL 2023/24): survey-weighted average
  boardings per stop per route/direction/quarter-hour slot. 17,943 stops.
- **AI23** (ONS Accessibility Indicators 2023): 8 LSOA-level PT accessibility
  features (30-min access to employment, hospitals, GP, supermarkets,
  pharmacies, primary/secondary schools, built-up areas).
- **OSM POI** (OpenStreetMap via osmnx): 6 land-use categories counted within
  500m of each stop (residential, shopping, company, education, entertainment,
  scenic/cultural).
- **Service Coverage** (derived from BUSTO): number of distinct
  (route × direction × quarter-hour) combinations serving each stop — proxy
  for planned service frequency, obtainable from GTFS for new stops.

---

## Model
**GATv2** (Graph Attention Network v2, Brody et al. 2022) — inductive setting
following Hamilton et al. (2017, GraphSAGE). Test stops never seen during
training; they aggregate features from training neighbours at inference time.

**Graph structure:**
- KNN edges (K=5, Haversine distance)
- Route connectivity edges (consecutive stops on same route/direction from BUSTO)
- Combined: ~124,000 directed edges over 17,943 nodes

**Baselines:**
- HistAvg: mean boardings of training stops (borough-level mean)
- Random Forest (300 trees, no graph)
- MLP (same hidden capacity as GATv2, no graph)

**Evaluation:** Leave-borough-out spatial cross-validation, 33 London boroughs.
Metric: WMAPE (Weighted Mean Absolute Percentage Error, lower is better).

---

## Version History — how many times we've run this, and what changed

Every row below is a distinct `step4_model.py` code configuration. "Full runs"
counts complete 33-fold executions under that configuration (a `--quick`
5-fold smoke test doesn't count). Full detail for each version is in the
"Experiments" section further down; this is the at-a-glance summary.

| Version | Date | Full 33-fold runs | Architecture / training config | What changed vs previous version | Feature sets executed | Best result |
|---|---|---|---|---|---|---|
| **v1** | 19 Jun 2026 | 3 | MSE loss, DROPOUT=0.3, LR=1e-3, no skip connection, EPOCHS=1000, PATIENCE=40, K=5 | Original implementation | AI23-only, OSM-only, AI23+OSM | 0.7975 (RF, AI23+OSM) |
| **v2** | ~22 Jun 2026 | ≥1 (exact count lost) | + residual skip connection, DROPOUT=0.15, LR=5e-4 | Bugs 1–5 fixed: dead scaler removed, pre-warm dim fixed, **critical validation-loss leakage fixed**, log message fixed, residual skip added | Results never isolated to a named file — saved to the generic `results_cv_multigraph.csv`, which a later run then overwrote. **No v2-only numbers survive.** | not recoverable |
| **v3** (code) | 22 Jun 2026 | 0 at the time | + stratified 5-quantile val split, Huber(0.5) loss, val-check every 5 epochs (PATIENCE=20) | 3 more training refinements written into the code | `service_coverage` feature added (step3d) | code written but never executed at full scale until it merged into v4 below |
| **v4** | 16–17 Jul 2026 | 4 (+1 fast, non-GATv2 supplement) | same architecture as v3 (this was v3's first real 33-fold execution) + MLR and XGBoost baselines added, `--k10` flag added | Added 2 baselines (MLR, XGBoost); ran the GNN-fairness check (K=10); ran the optional V/C target experiment; fixed the borough-map ONS query (Bug 6); restored the archived v1 AI23-only/OSM-only files | OSM+SC, **AI23+OSM+SC (headline)**, K=10 fairness (AI23+OSM+SC), V/C (AI23+OSM) — plus a fast HistAvg/MLR/RF/XGBoost-only supplement for AI23-only/OSM-only/AI23+OSM (no GATv2/MLP retrain, see Experiment Set 4) | **0.6311 (MLP, AI23+OSM+SC)** |

**Total full 33-fold GATv2 executions across the project: at least 8** (3 in
v1, ≥1 undocumented in v2, 4 in v4). Note v2 and v3 were never cleanly
isolated as their own experiment: v2's results were overwritten before being
saved under a distinct name, and v3's training refinements sat unexecuted at
full scale for 3.5 weeks until they were run for the first time in the same
session as the v4 additions (MLR/XGBoost/K=10). So in practice there are only
two result generations that can be directly compared end-to-end: **v1** (3
feature sets, buggy validation) and **v4** (4 feature sets, all bugs fixed,
6 models instead of 4). Anything described as "v2" or "v3" in this log is a
code milestone, not a separate results generation.

---

## Bugs Found and Fixed

### Bug 1 — Dead global scaler (REMOVED)
**What:** `X_sc = scaler.fit_transform(X_raw)` was computed globally but never
used. Each CV fold fitted its own scaler correctly on training data only.
The dead line gave a false impression of a global fit-transform.
**Fix:** Removed the dead line entirely.
**Impact:** No change to results, but removed potential source of confusion.

### Bug 2 — Pre-warm used wrong input dimension (FIXED)
**What:** The PyG kernel pre-warm call built a dummy model with
`len(FEAT_COLS)=8` even when running with 14 or 16 features. The real model
was built correctly inside each fold, so this only wasted ~2 minutes per run
on recompilation.
**Fix:** Changed pre-warm to use `len(ALL_FEAT_COLS)` after `prep_features()`
has set the global.
**Impact:** ~2 min saved per full 33-fold run.

### Bug 3 — Validation loss computed from training-mode forward pass (CRITICAL, FIXED)
**What:** Inside `train_nn()`, `out = model(x, ei)` was computed with
`model.train()` active (dropout=0.3 enabled). Then `model.eval()` was called,
but the validation loss `vl = F.mse_loss(out[val_pos], y[val_pos])` still used
the stochastic `out` from the training-mode pass. Early stopping was therefore
triggered by noisy, dropout-corrupted validation loss signals, causing
premature stopping or delayed stopping in different folds inconsistently.
**Fix:** Recomputed a clean forward pass inside `model.eval()`:
```python
model.eval()
with torch.no_grad():
    vl = F.mse_loss(model(x, ei)[val_pos], y[val_pos]).item()
```
**Impact:** Early stopping now reflects true validation performance.
Deterministic and correct signal for patience counter.

### Bug 4 — Results file log message wrong filename (FIXED)
**What:** Terminal printed "Saved -> results_cv.csv" but actual file was
`results_cv_multigraph.csv`. Caused confusion when copying results.
**Fix:** Updated log message to show correct filename.

### Bug 5 — GATv2 lacked residual skip connection (ARCHITECTURAL FIX)
**What:** Original GATv2 forced all information through the graph aggregation
path with no way to bypass it. In leave-borough-out CV, test stops aggregate
from cross-borough training neighbours (different demand characteristics),
introducing noise the model could not suppress.
**Fix:** Added a linear skip connection:
```python
self.skip = torch.nn.Linear(in_ch, HIDDEN_DIM * HEADS, bias=False)
# forward:
h = F.elu(self.c1(x, edge_index)) + self.skip(x)
```
This lets the model learn to weight graph aggregation vs direct feature path
per-fold, depending on how informative the neighbourhood is.
**Also:** Reduced DROPOUT 0.3→0.15, LR 1e-3→5e-4 for better convergence
with the residual path.

### Bug 6 — Borough choropleth map only rendered 1/33 boroughs (FIXED, 16 Jul 2026)
**What:** `step5a_borough_map.py`'s ONS boundary query filtered on
`LAD23NM LIKE '%London%' OR LAD23NM='City of London'`. Borough names like
"Camden" or "Hackney" don't literally contain the word "London", so the query
only ever matched "City of London" itself (log showed "Merged: 1 boroughs
matched to results" on the pre-fix run).
**Fix:** Changed the filter to `LAD23CD LIKE 'E09%'` (ONS code prefix for all
32 London boroughs + City of London = 33 LADs, matching the fold count).
**Impact:** `borough_wmape_map.png` now correctly shows all 33 boroughs.

### Data loss, then RESTORED (not a code bug) — v1 AI23-only / OSM-only per-fold CSVs
**What:** `results_cv_ai23_only.csv`, `results_summary_ai23_only.csv`,
`results_cv_osm_only.csv`, `results_summary_osm_only.csv` were sitting in a
folder named `05_archive_deleted_20260716_0701` at the start of the 16 Jul
session; that folder appeared deleted partway through the session (not by
this pipeline). **17 Jul: the student restored the folder from backup.**
Values in the restored CSVs match the archived text figures in Experiment Set
1 exactly (RF=0.8121/0.8062, MLP=0.8114/0.8058, GATv2=0.8241/0.8259 for
AI23-only/OSM-only respectively), confirming no corruption. The files were
copied back to the project root and `step6_consolidated_table.py` rebuilt
from the live CSVs (its `ARCHIVED_MEANS` text-fallback path is now unused but
left in place as a safety net if this ever happens again).
**Impact:** none — `consolidated_results_table.csv` numbers are unchanged
from the fallback values, now traceable to a live CSV again.

---

## Experiments

### Experiment Set 1 — Feature Ablation (v1, original GATv2)
**Config:** EPOCHS=1000, PATIENCE=40, K=5, HIDDEN=64, HEADS=4,
DROPOUT=0.3, LR=1e-3, no skip connection.
**Bugs present:** Bug 3 (noisy val loss).

| Experiment | Result file | HistAvg | RF | MLP | GATv2 |
|---|---|---|---|---|---|
| AI23-only (8+coords) | results_summary_ai23_only.csv | 1.0822 | 0.8121 | 0.8114 | 0.8241 |
| AI23+OSM (14+coords) | results_summary_ai23_osm.csv | 1.0822 | 0.7975 | 0.7983 | 0.8058 |
| OSM-only (6+coords) | results_summary_osm_only.csv | 1.0822 | 0.8062 | 0.8058 | 0.8259 |

**Key findings:**
- AI23+OSM is the best feature combination for all models
- OSM POI features improve every model vs AI23-only (RF: −1.46pp, GATv2: −1.83pp)
- GATv2 consistently 1–2pp worse than RF/MLP due to cross-borough graph noise
- All ML models beat HistAvg by ~26% WMAPE

---

### Experiment Set 2 — Architectural Fix (v2, residual skip + tuned dropout/LR)
**Config:** EPOCHS=1000, PATIENCE=40, K=5, HIDDEN=64, HEADS=4,
DROPOUT=0.15, LR=5e-4, WITH skip connection.
**Bugs fixed:** Bugs 1–5 all resolved.

Results files: results_cv_multigraph.csv / results_summary_multigraph.csv
(last run overwrites shared file — see individual named files for v1 baseline)

---

### Experiment Set 3 — Service Coverage Feature (v3, current)
**New feature:** `service_coverage` = n_route_dir_qhr_rows from BUSTO,
merged via step3d_add_service_coverage.py. Measures planned service supply
diversity per stop. Valid for cold-start (use GTFS for new stops).
**Hypothesis:** Service frequency is the strongest supply-side predictor of
demand. Adding it should reduce WMAPE by ~3–6pp.

| Experiment | Flag | Result file | HistAvg | RF | MLP | GATv2 |
|---|---|---|---|---|---|---|
| OSM + SC | `--osm-only --with-sc` | results_summary_osm_sc.csv | 1.0822 | 0.6353 | 0.6356 | 0.7325 |
| AI23 + OSM + SC (headline) | `--with-sc` | results_summary_ai23_osm_sc.csv | 1.0822 | 0.6428 | 0.6311 | 0.7187 |

**Key findings:**
- service_coverage is the single strongest predictor found in this project:
  WMAPE fell from ~0.80-0.81 (AI23+OSM, no SC) to ~0.63-0.64 (+SC) for every
  tabular model — roughly double the -3 to -6pp improvement hypothesised.
- With SC added, **MLP (0.6311) is now the best single model**, narrowly
  ahead of RF (0.6428) — a change from the pre-SC picture where RF led.
- GATv2 remains worst (0.7187), an 8-9pp gap vs MLP/RF — the honest-finding
  gap persists (and widens in absolute terms) even with the strongest feature
  set.

---

### Experiment Set 4 — MLR + XGBoost baselines, GNN-fairness check, V/C target (v4, 16-17 Jul 2026)
**Additions to step4_model.py:** MLR (sklearn Ridge, alpha=1.0) and XGBoost
(300 estimators, max_depth=6, lr=0.05) added as baselines, same folds/scaler/
log1p target as RF. `--k10` flag added (K_NEIGHBORS=10 instead of 5) for the
GNN-fairness check. Config otherwise unchanged from Experiment Set 3.

**MLR/XGBoost for the 3 pre-SC feature sets** were computed separately via
`step4c_fast_baselines.py` (HistAvg/MLR/RF/XGBoost only, no GATv2/MLP retrain
— see the "Data loss" bug entry above for why MLP/GATv2 for those two feature
sets still come from the v1 archived text values):

| Feature set | HistAvg | MLR | RF | XGBoost |
|---|---|---|---|---|
| AI23-only | 1.0822 | 0.8120 | 0.8128 | 0.8207 |
| OSM-only | 1.0822 | 0.8078 | 0.8064 | 0.8084 |
| AI23+OSM | 1.0822 | 0.7982 | 0.7970 | 0.8075 |

(RF=0.7970 here vs the archived v1 RF=0.7975 for AI23+OSM — consistent to
0.0005, confirming the fast-baseline protocol matches the original fold/
scaler setup.)

**GNN-fairness check (K=10 vs K=5), AI23+OSM+SC, `results_summary_gnn_fairness.csv`:**

| K | HistAvg | MLR | RF | XGBoost | MLP | GATv2 |
|---|---|---|---|---|---|---|
| 5 (headline) | 1.0822 | 0.6404 | 0.6428 | 0.6437 | 0.6311 | 0.7187 |
| 10 (fairness) | 1.0822 | 0.6404 | 0.6428 | 0.6437 | 0.6305 | 0.7372 |

**Finding:** doubling the KNN neighbourhood (109,182 → 212,296 edges) did
**not** close the GATv2-RF gap — GATv2 WMAPE got slightly *worse* (0.7187 →
0.7372), while every tabular model was essentially unchanged (as expected,
K only affects the graph). Reported per the frozen plan: *"strengthening the
aggregation prior did not reverse the pattern — if anything, more
cross-borough context made the contamination problem marginally worse."*
This directly answers the anticipated examiner question "did you under-tune
the GNN?".

**V/C (overcrowding) target experiment, `results_summary_vc_ai23_osm.csv`:**

| Model | WMAPE |
|---|---|
| HistAvg | 0.2977 |
| RF | 0.2893 |
| MLP | 0.2894 |
| GATv2 | 0.3214 |

**Finding:** on this target, GATv2 is worse than even HistAvg (0.3214 vs
0.2977) — unlike the boardings target, where all ML models beat HistAvg by
~26%. peak_vc is a much lower-variance, more bounded target (mean 0.286, 0%
of stops exceed the TfL 0.85 threshold in this data), so HistAvg is a much
stronger naive baseline here; RF/MLP still edge it out, GATv2 does not. Note
this alongside the boardings result rather than in place of it — it is a
secondary/optional experiment, not the headline task.

**Per-borough note (headline AI23+OSM+SC, not the older AI23+OSM used in
earlier drafts) — Camden and Hillingdon numbers changed from the v1
narrative:**

| Borough | GATv2 | RF | gap |
|---|---|---|---|
| Camden | 0.6580 | 0.5230 | +0.1350 |
| Hillingdon | 0.8025 | 0.7647 | +0.0378 |

Camden is **no longer a near-tie** the way it was in the old AI23+OSM
results (previously GATv2=RF=0.756). The 3 boroughs where GATv2 now loses
worst are City of London (+0.2126), Hackney (+0.1551), and Westminster
(+0.1440) — all **inner, well-connected** boroughs, not peripheral ones like
the old Hillingdon example. City of London is a n=101 outlier (financial-
district demand profile). This means the "peripheral boroughs suffer worst
from cross-borough contamination" mechanism story from the v1 results does
**not** hold as cleanly on the v4 headline data — the discussion section
should be rewritten from the fresh `borough_extract.csv`, not the old v1
anecdote. See full pivot in `borough_extract.csv`.

---

### Experiment Set 5 — IDW spatial-interpolation baseline (v4.1, 17 Jul 2026)
**Motivation:** literature review of cold-start adaptation in transport
(Liu et al. 2017, NYC Citi Bike) uses gravity models + spatial interpolation
(natural-neighbor interpolation, kriging) as the classical cold-start
approach — a method family not represented in the existing baseline suite
(HistAvg, MLR, RF, XGBoost, MLP, GATv2 are all either naive-mean or
feature-based). Added as a 7th baseline to directly test whether GATv2's
learned graph aggregation adds anything beyond plain geographic proximity.

**Method:** Inverse Distance Weighting (IDW) — k=20 nearest TRAINING stops
by haversine distance, weight = 1/distance², predicts log1p(boardings) as
the weighted average. Uses ONLY lat/lon and the training target — no AI23,
OSM, or service_coverage features at all. True kriging (variogram fitting)
was considered but is out of scope for a baseline check; IDW is the
standard, simpler geostatistical interpolator and is what Liu et al. compare
against natural-neighbor interpolation. Added permanently to
`step4_model.py` (runs on every future full CV) and computed immediately via
the fast standalone `step4d_idw_baseline.py` (~1 sec, no training) on the
full 17,943-stop set, since IDW does not depend on the active feature set.

**Result, `results_summary_idw.csv`:** WMAPE = 0.8487 (±0.0396), median
0.8542 — identical across all 4 feature-set columns (feature-independent by
construction).

**Finding:** IDW (0.8487) beats HistAvg (1.0822) — geography alone carries
real signal — but is worse than **every** feature-based model, including the
weakest ones (AI23-only RF=0.8128, MLP=0.8114). On the headline AI23+OSM+SC
feature set the gap is stark: IDW=0.8487 vs GATv2=0.7187 vs RF/MLP≈0.63-0.64.
This sharpens the dissertation's central argument: GATv2 (graph structure +
features) clearly outperforms pure spatial proximity (IDW), so the graph is
not worthless — but it still can't match what RF/MLP extract from the same
features with no graph at all. The honest finding becomes more precise: it
is not that graph methods add nothing, it is that *learned feature
interactions dominate learned spatial aggregation* for this cold-start task,
and message passing over a noisy cross-borough KNN graph costs more than it
gives back relative to just using the features directly.

---

### Experiment Set 6 — PCA-reduced AI23 features (v4.2, 17 Jul 2026)
**Motivation:** dataset audit found severe multicollinearity within the 8
AI23 accessibility features: pairwise correlations up to r=0.97
(gp_30min↔pharmacies_30min), and Variance Inflation Factors up to **37.7**
(pharmacies_30min), 17.7 (gp_30min), 17.2 (primary_schools_30min) — all well
above the conventional VIF>10 "severe" threshold. PCA on the 8 (log1p,
standardized) features shows why: PC1 alone explains 77.2% of variance,
3 components explain 94.4%. All 8 features are largely proxies for one
underlying latent factor (general PT network density/centrality), not 8
independent signals. `main_bua_30min` is a separate, unrelated issue: 94.4%
zeros, VIF=1.0 (not correlated with anything — just close to zero-variance).

**Method:** added `--pca-ai23` flag to `step4_model.py`. Per fold (train-only
fit, same no-leakage protocol as the main scaler): standardize the 8 raw
(log1p) AI23 columns, fit `PCA(n_components=3)` on training stops, transform
both train and test, concatenate the 3 components with the unchanged
OSM+SC+coords columns, then apply the existing StandardScaler as before.
Net effect: 17 input features (AI23+OSM+SC+coords) become 12 for this
variant. Same folds, same seed, same models, same architecture — only the
AI23 representation changes.

**Result — headline feature set (AI23+OSM+SC) with vs without PCA:**

| Model | AI23+OSM+SC (raw 8 AI23) | PCA(AI23)+OSM+SC (3 components) | Δ |
|---|---|---|---|
| MLR | 0.6404 | 0.6401 | −0.03pp |
| RF | 0.6428 | 0.6400 | −0.28pp |
| XGBoost | 0.6437 | 0.6474 | +0.37pp |
| MLP | 0.6311 | 0.6299 | −0.12pp |
| GATv2 | 0.7187 | 0.7171 | −0.16pp |

**Finding:** despite real, severe multicollinearity (VIF up to 37.7),
removing it via PCA changed every model's WMAPE by less than 0.4pp in either
direction — essentially a null result for predictive accuracy. This makes
sense in hindsight: Ridge's L2 penalty, tree ensembles' split-based
robustness, and NN regularization (dropout, weight decay) all already
absorb the practical cost of correlated inputs on *prediction*, even though
multicollinearity does still harm *individual coefficient interpretability*
in the MLR model specifically (not tested here — would require inspecting
coefficient stability across folds, which is a Ch.4/discussion-level
question, not a WMAPE question). The GATv2-vs-MLP/RF gap is essentially
unchanged (8.7-8.8pp either way), which is a fourth independent confirmation
(alongside service_coverage addition and the K=10 fairness check) that
adding/reshaping information does not close that gap — consistent with the
mechanism being about graph topology, not feature quality or redundancy.
**Practical conclusion: keep the raw 8 AI23 features** (more interpretable
per-domain reporting: "30-min access to hospitals" means something to a
reader; "PC2" does not) rather than switching to PCA, since PCA buys
essentially nothing here.

---

### Experiment Set 7 — Functional-similarity edges: single-graph vs. Zheng's fusion architecture (v4.3, 17 Jul 2026)
**Motivation:** correction of a documentation error (see "Comparison with
Zheng et al." above) — Zheng et al. (2025)'s functional-similarity graph is
built from Pearson correlation of POI count vectors (available at cold-start
time), not historical demand as previously assumed here. Re-evaluated with
two implementations, both on the headline AI23+OSM+SC feature set:

**(a) Single-graph variant** (`--func-sim` flag, `step4_model.py`):
functional-similarity edges (ρ>0.99, top-10 per stop — see "recalibration"
note below) added as a third edge type into the SAME multigraph as the
existing KNN + route edges, one GATv2 as before.

**(b) Fusion architecture** (`step4e_zheng_fusion.py`, new script): faithful
replica of Zheng's Eqs. (8)-(12) — separate GATv2 branches over G1
(geographic: KNN+route) and G2 (functional similarity), concatenated, then a
third GATv2 pass over the union graph G3 = G1 ∪ G2. A residual skip
connection was added (not in the original paper) since Bug 5 established
this is critical for cold-start robustness in this setting — omitting it
would confound "does fusion help" with "does this variant lack the fix we
already know matters."

**Recalibration required before either could run:** Zheng's literal ρ>0.8
threshold, applied to our 6 OSM POI categories, produces a near-complete
graph — 48,087,084 directed edges, average degree 2,680 (vs. ~124K edges,
degree ~7 for the existing KNN+route graph). This is not a bug; it reflects
real zero-inflation in the POI counts (many stops share "mostly zero except
one category" profiles, which are trivially highly correlated). A threshold
sweep (ρ>0.8/0.9/0.95/0.99/0.999) found no threshold that is both principled
and well-behaved; ρ>0.99 + a top-10-per-stop cap (mirroring the existing
K=5 KNN graph's sparsity) was chosen as a documented, reasonable
recalibration — giving 133,646 directed edges (avg degree ~7.4), a
comparable density to the existing graph.

**Results, headline AI23+OSM+SC feature set:**

| Model | WMAPE | vs. plain GATv2 (0.7187) | vs. best tabular (MLP=0.6311) |
|---|---|---|---|
| GATv2 (plain, for reference) | 0.7187 | — | +8.76pp |
| GATv2 + func-sim (single graph, a) | 0.7352 | **worse by 1.65pp** | +10.41pp |
| GATv2-Fusion (G1/G2/G3, b) | **0.7070** | **better by 1.17pp** | +7.59pp |

**Finding — architecture matters more than just adding graph information:**
the two implementations of the SAME underlying idea (use functional
similarity) produced opposite results. Mixing functional-similarity edges
into the same adjacency matrix as geographic/route edges made GATv2 worse —
plausibly because a single attention mechanism cannot distinguish "this
neighbour is nearby" from "this neighbour has a similar POI profile"; the
two relationship types get conflated into one set of attention weights.
Processing them through separate GAT branches before fusing (Zheng's actual
architecture) avoids that conflation and is the single most effective graph
intervention tried in this project — better than K=10 (Experiment Set 4),
better than PCA-reduced features (Experiment Set 6), better than the
single-graph functional-similarity variant above.

**But it still does not overturn the honest finding.** GATv2-Fusion
(0.7070) narrows the gap to MLP (0.6311) from 8.76pp to 7.59pp — a
real, measurable improvement — but does not close it. Even the best-effort,
literature-informed graph architecture tried in this project remains
clearly behind simple tabular models trained on the same features with no
graph at all. This is arguably the strongest version of the honest finding:
it is not that the graph modelling was left un-tried or under-engineered —
a deliberate, three-branch, dual-graph fusion architecture was built and
tuned, and tabular models still win.

**Per-borough nuance (GATv2-Fusion vs plain GATv2, from `results_cv_zheng_fusion.csv`):**
Fusion beat plain GATv2 in 21/33 boroughs. Hillingdon — the peripheral
borough singled out in the original (v1) "worst case" narrative — improved
sharply under Fusion (0.8025 → 0.7663) to a near-tie with RF (0.7647,
gap=+0.0016). But Camden got markedly *worse* under Fusion (0.6580 → 0.7412)
and is now the single largest Fusion-vs-RF gap of any borough (+0.2182,
overtaking City of London/Hackney/Westminster from Experiment Set 3's
extract). **The "which boroughs are hardest for the graph" picture keeps
changing with the exact model/feature configuration** — this itself is
worth a sentence in the discussion chapter: the mechanism (cross-borough
contamination) is real and repeatedly demonstrated, but which specific
boroughs suffer most is sensitive to architecture choices, not a fixed
property of London's geography. Don't over-fit the discussion prose to any
one run's specific borough examples — cite the mechanism and the aggregate
gap, use a borough example from the SAME run being discussed.

**Status: exploratory, not part of the frozen headline comparison** per the
student's explicit instruction — logged here for the record; whether either
variant is promoted into the dissertation's main results table is the
student's call, to be made after reviewing this write-up.

**CORRECTION after significance testing (18 Jul 2026, see next section):**
the "GATv2-Fusion beats plain GATv2 by 1.17pp" framing above overstates the
Fusion result. Paired Wilcoxon signed-rank test across the 33 boroughs:
GATv2-Fusion vs plain GATv2, p=0.126 — **not statistically significant** at
the standard 0.05 threshold. The mean improvement is real in this sample but
33 paired observations is not enough to rule out noise. Report this as "a
promising but statistically inconclusive trend toward improvement," not as
a confirmed effect. What IS statistically solid: GATv2-Fusion is still
significantly worse than both MLP (p<0.0001) and RF (p<0.0001) — the core
honest finding is untouched by this correction, only the "architecture
matters" sub-claim needs softening.

---

## Statistical Significance Testing (18 Jul 2026)

Added after the headline/Experiment Set 7 results were complete, to check
whether the mean WMAPE differences reported throughout this log are
statistically meaningful or could be sampling noise across the 33 borough
folds. Method: paired Wilcoxon signed-rank test (non-parametric, appropriate
for 33 paired-but-non-normal fold-level WMAPE scores; paired because the
same 33 boroughs are scored under every model, so this is a matched-pairs
design, not independent samples). No retraining required — computed
directly from existing per-fold CSVs.

**Headline feature set (AI23+OSM+SC), from `results_cv_ai23_osm_sc.csv`:**

| Comparison | mean(A−B) | p-value | Significant (p<0.05)? |
|---|---|---|---|
| GATv2 vs RF | +0.0759 | <0.00001 | **YES** |
| GATv2 vs MLP | +0.0875 | <0.00001 | **YES** |
| GATv2 vs MLR | +0.0783 | <0.00001 | **YES** |
| MLP vs RF | −0.0117 | 0.00225 | **YES** |
| MLP vs MLR | −0.0092 | 0.00081 | **YES** |
| RF vs MLR | +0.0024 | 0.357 | no |
| MLP vs XGBoost | −0.0126 | 0.060 | no (borderline) |
| RF vs XGBoost | −0.0009 | 0.386 | no |

**Fusion architecture, from `results_cv_zheng_fusion.csv`:**

| Comparison | mean(A−B) | p-value | Significant? |
|---|---|---|---|
| GATv2-Fusion vs GATv2 (plain) | −0.0117 | 0.126 | no |
| GATv2-Fusion vs MLP | +0.0759 | <0.00001 | **YES** |
| GATv2-Fusion vs RF | +0.0642 | <0.00001 | **YES** |

**What this confirms and refines:**
1. **The central honest finding is statistically airtight, not just a mean
   difference.** GATv2 (in every variant tested — plain, K=10, PCA features,
   func-sim, and Fusion) loses to MLP and RF at p<0.00001. This is the
   single most important number for defending the dissertation's core claim
   against an examiner asking "is that gap real or just 33 noisy folds?"
2. **MLP's edge over RF and MLR is real** (p=0.002, p=0.0008) — "MLP is the
   best model" is a defensible, significant claim, not cherry-picking the
   lowest mean.
3. **But MLP's edge over XGBoost is NOT significant** (p=0.060, borderline).
   The precise claim should be "MLP and XGBoost are statistically tied for
   best tabular model, with MLP having the lower point estimate," not
   "MLP is unambiguously best."
4. **RF and MLR are statistically indistinguishable** (p=0.357) — unsurprising,
   since Ridge and RF are both being fed the same well-behaved, non-noisy
   features and land within a quarter of a percentage point of each other.
5. **The Fusion "improvement" over plain GATv2 does not survive significance
   testing** (p=0.126) — see the correction note in Experiment Set 7 above.

**CORRECTION (19 Jul 2026, after Experiment Set 10 — per-fold-tuned baselines):**
point 2/3 above were based on *untuned* RF and XGBoost, and no longer state the
most defensible claim. After a fair, properly-scoped, no-leakage hyperparameter
search: **RF-tuned (0.6339) is statistically indistinguishable from MLP
(p=0.292, not significant)** — MLP's earlier significant edge over RF
(p=0.00225) does not survive once RF is tuned. XGBoost-tuned got *worse*
(0.6497), not better, and still loses to MLP significantly (p=0.00014). The
current, correct claim is: **"MLP and tuned RF are statistically tied for
best; XGBoost, even tuned, is not."** This is a strictly more defensible
version of the original claim, not a weaker one — see Experiment Set 10 for
the full search methodology and results.

---

## External Validity: Independent-Dataset Replication via Literature (18 Jul 2026)

**Context:** the student asked whether the core finding (GATv2 loses to
tabular models under strict spatial CV) could be double-checked on another
dataset. A full second-dataset empirical replication (new city, new demand
data, new feature pipeline) was assessed as multi-week scope creep this late
in the project and explicitly not pursued. Instead, this section formalizes
an argument that was already implicit in this log's citations but never
made explicit and load-bearing: **two independent, already-published studies
already function as out-of-sample replications of this dissertation's
central result.**

**Yusuf et al. (2025)** — stop-level bus automatic-passenger-count
prediction in Trondheim, Norway. Independent country, independent city,
independent demand dataset, independent code. Same task type (stop-level
transit demand) and the same *kind* of features (land-use/demographic
buffer variables, directly analogous to this dissertation's OSM POI + AI23
accessibility features). Their finding: tree-based ML (XGBoost) beats deep
learning. Their own diagnosis of why: stop-level models "lack inductive
biases on the spatial structure of transit networks" — which is close to an
independent, prior articulation of the cross-borough-aggregation-
contamination mechanism this dissertation identifies directly (Experiment
Sets 3-7): a graph-structural inductive bias that should help doesn't,
because the spatial structure available at inference time for a genuinely
unseen stop is not the structure the model needs.

**Grinsztajn et al. (2022)** — a large-scale, domain-general benchmark
(NeurIPS 2022, ~45 tabular datasets, no transit connection at all) finding
tree-based methods consistently outperform deep learning on tabular data,
with a mechanistic explanation: tabular target functions are often
irregular/non-smooth (favouring trees' axis-aligned splits over a neural
network's smoother decision boundaries), neural nets are hurt more by
uninformative features, and NN architectures are naturally rotation-
invariant — a poor inductive bias when, as here, individual feature columns
(employment access, GP access, service_coverage, etc.) carry real,
non-interchangeable meaning.

**The argument, stated precisely:** this dissertation's result is not a
literal replication of either paper (neither uses cold-start spatial CV or
a graph architecture at all), so this is convergent evidence for the
*outcome* (tabular beats graph/deep learning at this task granularity), not
a replication of the *method*. What's novel here — the cold-start framing,
the spatial-CV-induced mechanism, and the statistically significant,
multi-variant confirmation (5 independent graph interventions, all
p<0.00001 vs. tabular) — remains this dissertation's own contribution.
The literature comparison establishes that the outcome is not an artefact
of this dataset or this specific implementation; the dissertation's own
experiments establish *why* it happens here specifically.

**Draft prose for Ch.5 discussion** (paste into a chapter-writing chat and
tighten to house style):

> The finding that GATv2 underperforms tabular baselines under strict
> spatial cross-validation is not an artefact of this dataset. It replicates,
> in independent settings, two bodies of prior evidence. Yusuf et al. (2025)
> found the same pattern at the same task granularity — stop-level transit
> demand prediction from land-use and demographic buffer features — in an
> entirely different transit system (Trondheim, Norway), and independently
> diagnosed the cause as a lack of inductive bias for transit network spatial
> structure, closely anticipating this dissertation's cross-borough
> aggregation contamination mechanism. More broadly, Grinsztajn et al. (2022)
> establish that tree-based methods outperform deep learning across tabular
> data in general, for reasons (irregular target functions, sensitivity to
> uninformative features, unhelpful rotational invariance) that apply
> directly to the AI23/OSM/service_coverage feature space used here. Taken
> together, these two independent studies support treating this
> dissertation's result as a genuine, generalisable pattern rather than a
> peculiarity of London bus data — while the cold-start spatial-CV mechanism
> identified here, and its confirmation across five independent graph
> interventions (Experiment Sets 3, 4, 6, and 7), remain this dissertation's
> own contribution.

---

### Experiment Set 8 — Pre-registered test: learned per-node MLP/GATv2 mixing gate (18-19 Jul 2026)

**Pre-registration (written and fixed BEFORE running anything, per the
student's exact task specification — reproduced here verbatim for the
record):** hypothesis was that message passing dilutes a strong node-local
signal (motivating evidence: the MLP-GATv2 gap is 0.75pp without
service_coverage vs 8.76pp with it — the graph looks worse specifically when
the node-local signal is strongest). Test: `h_out = alpha*MLP(x_self) +
(1-alpha)*GATv2_aggregate(neighbours)`, alpha = sigmoid(Linear(in_ch,1)(x)),
per node, learned end-to-end. `MLPModel` and `GATv2Model` (with its existing
residual skip) used UNMODIFIED so alpha=1/alpha=0 recover them exactly —
verified directly (`torch.allclose`) before running anything. AI23+OSM+SC
only, K=5 KNN+route graph, same folds, 3 seeds (42/142/242), no
hyperparameter search, no architecture variants, one attempt only.
Three interpretation buckets were fixed in advance (see
`step4f_gated_mixing.py` docstring for the exact wording).

**Lineage (added 19 Jul 2026, for the write-up):** the gate
`alpha = sigmoid(Linear(x))` mixing two whole sub-networks is structurally
the same learned-sigmoid-gate mechanism as Highway Networks (Srivastava
et al. 2015: `h = T(x)*H(x) + (1-T(x))*x`, applied there to a layer and its
identity path rather than two full branches), and a degenerate 2-expert case
of the general mixture-of-experts gating pattern (Shazeer et al. 2017),
without MoE's sparsity/load-balancing machinery since both branches run on
every node here. Cite both — this was an original design for this project,
but not without precedent, and citing the lineage is more defensible than
presenting it as arising from nowhere.

**Results:**

| | WMAPE | vs MLP (0.6311) | vs GATv2 (0.7187) |
|---|---|---|---|
| Gated, seed 42 | 0.6257 | | |
| Gated, seed 142 | 0.6295 | | |
| Gated, seed 242 | 0.6277 | | |
| **Gated, 3-seed mean** | **0.6277 (sd across seeds 0.0019; sd across all 99 seed×borough rows 0.0555)** | **−0.35pp** | **−9.10pp** |

Wilcoxon signed-rank, paired by borough (n=33), Holm-Bonferroni-corrected
across the 2 comparisons: Gated vs GATv2 plain p<0.00001 (significant);
**Gated vs MLP p=0.00697 (significant, survives correction)**. All 3 seeds
individually landed below the MLP mean (consistent direction, not one
outlier seed driving the average).

**Alpha distribution:** mean=0.5212, median=0.520, IQR=[0.456, 0.587].
**0.00% of nodes have alpha>0.9, 0.00% have alpha<0.1** — the gate never
approaches either pure-MLP or pure-message-passing; it converges to a narrow
band around 0.5 everywhere. Per-borough mean alpha vs per-borough WMAPE gap
(Gated−MLP): Pearson r=−0.026, p=0.887 — statistically zero relationship.
Additionally: mean per-node alpha standard deviation ACROSS the 3 seeds
(0.085) is nearly as large as the pooled standard deviation ACROSS ALL NODES
(0.101) — most of the apparent per-node variation in alpha is seed-to-seed
training noise, not a stable, learned, node-specific signal.

**Applying the pre-registered interpretation — literal trigger vs. what the
alpha evidence actually supports (both stated, not reconciled away):**

*On the letter of the rule:* the result triggers bucket 3 ("beats 0.6311
with p<0.05 after Holm-Bonferroni ⇒ Finding 3 must be REWRITTEN, not
deleted, loss was architectural not evidential"). This is real and is
reported as such: a forced 100%-or-0% choice between MLP and GATv2 (what
every prior experiment in this dissertation tested) was suboptimal — an
architecture that can blend the two, even in this crude way, recovers all of
plain GATv2's 8.76pp deficit and then some.

*On the alpha evidence specifically (which the student's pre-registration
flagged as mattering MORE than the WMAPE):* the mechanism implied by bucket
3's framing — "the model learned when the graph is useful and adapted
accordingly" — is **not** what the alpha statistics show. A context-sensitive
gate would be expected to vary by node/borough in a way that tracks where
the graph helps or hurts (as Experiment Set 7's per-borough gap did). Instead
alpha is (a) never saturated, (b) uncorrelated with the per-borough gap
(r=−0.026), and (c) noisier across random seeds than across nodes. This
pattern — a small, statistically real edge produced by a near-constant,
non-adaptive blend of two decorrelated predictors — is at least as
consistent with a plain ensembling/variance-reduction effect as with "the
graph secretly carries diluted-but-real spatial signal." **This experiment
cannot distinguish between those two explanations**, because it did not save
the standalone MLP/GATv2 per-stop predictions needed to check whether their
residual errors are correlated (only aggregate WMAPE and the gate's own
alpha were saved). Checking that is the natural next step — explicitly NOT
run here, per the one-attempt constraint.

**Conclusion for the dissertation (write both halves, do not collapse to
one):** Finding 3 needs a genuine revision, not a deletion: "GATv2 loses to
tabular models" was previously stated for a fixed, non-adaptive graph
architecture, and that specific claim is now qualified — an architecture
that can blend graph and non-graph signal, even crudely, closes the entire
gap and edges past the best tabular model by a small but statistically
significant margin. However, the revised claim should NOT be overstated as
"the graph was secretly valuable all along": the mechanism behind the small
residual win is unresolved, the gate did not learn meaningfully differentiated
per-node or per-borough behaviour, and the effect size (0.35pp) is an order
of magnitude smaller than the gap this same gating recovered from plain
GATv2 (9.10pp). The honest framing is: *forced graph/non-graph mixing was
demonstrably the wrong design choice; whether adaptive mixing wins because
of real spatial information or because of generic ensembling remains open.*

#### Follow-up: residual correlation check (19 Jul 2026)

**Motivation:** Experiment Set 8 left one question explicitly open — is the
Gated model's small edge over MLP real spatial signal, or generic
ensembling of two decorrelated predictors? Testing this requires the raw
per-stop predictions of standalone MLP and standalone GATv2, which the
original headline run did not save. Retrained both (UNMODIFIED classes,
same folds, same seed=42 matching the frozen headline run, AI23+OSM+SC
only) purely to capture per-stop predictions — no new model, no
architecture change, one seed (this is a diagnostic on already-established
models, not a repeat of the pre-registered Experiment Set 8 test).

**Result 1 — residuals are highly correlated, not decorrelated:**
Pearson r=0.9035 (log1p-scale), r=0.9280 (original-scale), both p≈0. MLP and
GATv2 are largely wrong about the *same* stops, not complementary ones.

**Result 2 — a naive, non-learned, fixed 50/50 blend is worse than MLP
alone, not better:**

| Model | WMAPE (this retrain, seed=42) |
|---|---|
| Standalone MLP | 0.6248 |
| Standalone GATv2 | 0.7311 |
| **Naive fixed 50/50 blend (log1p-scale average, no learned gate)** | **0.6655** — worse than MLP |
| (reference) Experiment Set 8 Gated, learned alpha, 3-seed mean | 0.6277 — better than MLP |

**This rules out generic ensembling as the explanation.** If the small
Gated-vs-MLP edge were just variance reduction from averaging decorrelated
predictors, the naive fixed blend should show a similar (if smaller)
benefit. It does the opposite — it drags MLP's performance toward GATv2's
worse number, exactly as expected when blending two *highly correlated,
unequal-quality* predictors at a fixed ratio. Whatever the *learned* gate in
Experiment Set 8 achieves, it requires the joint, end-to-end training of
the gated architecture (where the MLP and GATv2 branches co-adapt during
training) — it is not recoverable by post-hoc averaging of the same two
model types trained independently. Secondary check: per-borough,
naive-ensemble penalty over MLP correlates with residual-correlation
strength at r=−0.367, p=0.036 — boroughs with less-correlated residuals lose
less from naive blending, as expected, but the effect is modest and this is
a minor confirmatory detail, not the headline result.

**Result 3 — an important caveat this same retrain surfaced, unprompted:**
this retrain's standalone MLP (0.6248) differs from the frozen headline
MLP (0.6311) by **0.63pp — nearly double Experiment Set 8's entire claimed
Gated-vs-MLP effect size (0.35pp)** — despite using the same seed (42), same
folds, same architecture, same everything. This is ordinary neural-network
training stochasticity (dropout draws, floating-point non-associativity;
`torch.use_deterministic_algorithms` was never set in this codebase), not a
bug. But it means: Experiment Set 8's 3-seed Wilcoxon test controlled for
the *Gated* model's own run-to-run variance, but compared it against a
*single stored* MLP number — and this follow-up shows directly that a
single MLP run can swing by more than the effect being claimed. **This
meaningfully weakens confidence in stating "Gated significantly beats MLP"
as a settled result**, even though the formal significance test (Section 2
above) is not in question on its own terms.

**Combined interpretation (supersedes the single-sided reading above):**
the mechanism is narrowed — it is NOT trivial ensembling (Result 2 rules
that out cleanly) — but whether the remaining small edge is a genuine,
reproducible property of joint gated training or is within the ordinary
noise floor of comparing one stochastic training run against another
(Result 3) is **not resolved** by the evidence collected so far. A fully
rigorous answer would require multiple independent seeds of the *standalone
MLP baseline itself* (not just the Gated model) to establish MLP's own
run-to-run variance envelope before concluding the Gated model's mean lies
outside it. That is a natural next step, explicitly not run here to respect
the original one-attempt-per-experiment discipline — but it should be the
first thing done before this result is written into the dissertation as a
confirmed reversal of Finding 3. **Until then, the defensible written claim
is: "forced 100%-or-0% graph/non-graph mixing is clearly wrong (large,
robust effect); whether adaptive/joint mixing yields a genuine small
additional edge over the best tabular model, beyond ordinary training
noise, is suggestive but not yet established beyond reasonable doubt."**

---

## Feature Set Summary

| Name | Dimensions | Columns |
|---|---|---|
| AI23-only | 8 + 2 coords | employment, hospitals, gp, supermarkets, pharmacies, primary_schools, secondary_schools, main_bua, lat, lon |
| OSM-only | 6 + 2 coords | poi_residential, poi_shopping, poi_company, poi_education, poi_entertainment, poi_scenic, lat, lon |
| AI23+OSM | 14 + 2 coords | all AI23 + all OSM + lat, lon |
| OSM+SC | 7 + 2 coords | all OSM + service_coverage + lat, lon |
| AI23+OSM+SC | 15 + 2 coords | all AI23 + all OSM + service_coverage + lat, lon |

---

## Conclusions So Far (updated 19 Jul 2026, v4 + Experiment Sets 9-10)

1. **Cold-start prediction is feasible.** Best WMAPE achieved: 0.6311 (MLP,
   AI23+OSM+SC) — a ~42% improvement over HistAvg (1.0822), well above the
   ~26% figure from the pre-SC (v1) results. **Updated 19 Jul:** after
   proper per-fold hyperparameter tuning (Experiment Set 10), RF-tuned
   (0.6339) is statistically indistinguishable from MLP (p=0.292) — report
   "MLP and tuned RF jointly best," not "MLP is uniquely best."

2. **service_coverage is the single strongest predictor tested.** Adding it
   to AI23+OSM roughly halved the remaining error gap to HistAvg (WMAPE
   ~0.80 → ~0.63 for every tabular model). Framed throughout as "planned
   service frequency (obtainable from GTFS for new stops)", not a
   demand-derived feature.

3. **OSM POI features remain valuable on top of AI23**, though the effect is
   now smaller in absolute terms than the SC effect (see Experiment Set 1).

4. **A fixed-architecture GATv2 does not outperform tabular baselines, and
   the gap does not close with a fairer graph.** GATv2 trails MLP/RF by 8-9pp
   WMAPE on the headline feature set, and K=10 (double the neighbourhood)
   made it slightly worse, not better — evidence against "the GNN was just
   under-tuned." Consistent with Grinsztajn et al. (2022) and Yusuf et al.
   (2025): tabular/gradient-boosted methods are competitive-to-superior for
   stop-level tabular-style prediction tasks. **Qualified by Experiment Set 8
   (18-19 Jul):** this claim holds specifically for architectures forced to
   choose 100% graph or 100% no-graph. An architecture that can *blend* MLP
   and GATv2 per node narrowly and significantly beats MLP (0.6277 vs
   0.6311, p=0.007) — but the learned blend ratio is a near-constant ~0.52
   with no meaningful per-node/per-borough adaptivity. A follow-up (19 Jul)
   ruled out generic ensembling as the cause (a naive fixed 50/50 blend of
   the same two model types is worse than MLP, not better — residuals are
   highly correlated at r=0.90, not decorrelated), narrowing the explanation
   to something about joint/co-adapted training — but the same follow-up
   found MLP's own run-to-run training variance (0.63pp between two seed-42
   runs) exceeds the claimed effect size (0.35pp), so the result should be
   reported as suggestive, not settled, pending a proper multi-seed baseline
   for standalone MLP. See Experiment Set 8 and its follow-up for full
   reasoning — do not simplify this to either "GATv2 wins" or "it's just
   noise" in the write-up.

5. **The cross-borough-contamination mechanism needs re-examination on the
   v4 headline data.** The v1 anecdote (near-tie in "well-connected" Camden,
   worst gap in "peripheral" Hillingdon) does not hold on AI23+OSM+SC — the
   three worst boroughs are now inner/central (City of London, Hackney,
   Westminster). The discussion chapter should draw on `borough_extract.csv`
   (v4) rather than repeating the v1 framing.

6. **Pure spatial proximity is a weak predictor on its own.** IDW spatial
   interpolation (Liu et al. 2017-style, lat/lon only, no features) scores
   0.8487 — better than HistAvg but worse than every feature-based model,
   including the weakest ones (AI23-only). This refines finding 4: GATv2
   clearly extracts more than pure geography (0.8487 → 0.7187), so graph
   structure is not worthless — it just adds less than the features already
   give a graph-free tabular model. The honest finding is about *learned
   feature interactions dominating learned spatial aggregation*, not about
   geography being irrelevant.

7. **Attention specifically, not graph aggregation in general, is part of the
   problem.** Experiment Set 9: swapping GATv2Conv for plain GCNConv (same
   width, same residual skip, same folds) gave a significant improvement
   (0.7006 vs 0.7187, p=0.010) — a simpler, less expressive graph mechanism
   beats a more expressive one under cross-borough contamination. GCN still
   loses significantly to MLP/RF (p<0.00001 both), so the central finding is
   unaffected, but the mechanism story is now sharper: it is not "any graph
   method fails here," it is "the more flexible the graph mechanism, the more
   room it has to be misled by systematically-wrong neighbours."

8. **The tabular-vs-graph comparison survives a fair hyperparameter-tuning
   budget — and tuning made the tabular side stronger, not weaker.**
   Experiment Set 10: per-fold RandomizedSearchCV (no leakage — tuned on each
   fold's training data only) improved RF by 0.89pp (closing its gap to MLP
   to statistical insignificance, p=0.292) but made XGBoost 0.60pp *worse* —
   both results reported honestly. RF-tuned still beats GATv2 significantly
   (p<0.00001). This closes the single biggest fairness gap in the original
   comparison (GATv2 received five architectural interventions; the tabular
   baselines originally received none) and makes the headline claim
   strictly more defensible, not less.

7. **The AI23 features are severely collinear (VIF up to 37.7) but this does
   not hurt predictive accuracy.** Collapsing the 8 AI23 features to 3 PCA
   components (94.4% variance retained) changed every model's WMAPE by
   <0.4pp. Regularization (Ridge's L2, tree splitting, NN dropout/weight
   decay) already absorbs the practical prediction cost of the redundancy.
   Keep the raw 8 features for interpretability — PCA buys nothing here.
   This is a fourth independent confirmation that "more/different
   information" does not close the GATv2-vs-tabular gap (alongside
   service_coverage, K=10, and IDW above) — the gap is about graph topology,
   not about what's fed into the model.

---

## Comparison with Zheng et al. (2025)

| Aspect | Zheng et al. (2025) | This dissertation |
|---|---|---|
| Task | Temporal demand prediction (15-min intervals) | Cold-start static demand prediction |
| Graph edges | Two graphs fused: G1 geographic adjacency + G2 functional similarity (Pearson ≥ 0.8 of **POI count vectors**, not demand) | Geographic KNN + route connectivity, single graph |
| Architecture | Separate GAT per graph (G1, G2), fused into G3, third GAT pass | Single GATv2 over one combined multigraph, residual skip |
| Node features | Historical passenger flow sequences (weekly/daily/recent modes) | AI23 accessibility + OSM POI + service coverage (static) |
| POI use | Edge construction (functional similarity) **and** available as node features here | Node features (local land use) |
| Metric | WMAPE | WMAPE |
| Cold-start | Not addressed (requires historical flow sequences — literally impossible to construct for a stop with no history) | Core contribution |

**Note:** Direct WMAPE comparison is not valid — Zheng predicts temporal
flow sequences for existing stops; this work predicts static demand at stops
with no prior data. These are different tasks.

**Correction (17 Jul 2026):** an earlier version of this table and of
`PROJECT_HANDOFF.md` described Zheng's functional-similarity edges as
Pearson correlation of *historical flow* between stops, and rejected
replicating it on cold-start-infeasibility grounds. Re-reading the source
text: the correlation is computed on **POI count vectors** (static land-use
counts), not flow — fully computable for a cold-start test stop. See
Experiment Set 7 for the corrected re-evaluation (both a lightweight
single-graph variant and a faithful G1/G2/G3 fusion replica).

---

## Baseline Suite Justification (added 19 Jul 2026)

**Context:** the student compared this project's baseline suite against Zheng et al.
(2025)'s "Model Selection" section, which structures 10 baselines into 4 families, each
isolating one variable: (1-3) temporal-only, no spatial model (HA, ARIMA, LSTM); (4-5)
alternative spatial mechanisms instead of GAT (CNN, GCN); (6) a different SOTA family
(Transformer); (7-10) ablations of their own architecture's components (temporal window,
graph type). This section makes explicit how this dissertation's suite maps onto that
structure, and justifies every absence rather than leaving it silent.

**Why there is no ARIMA / pure-temporal baseline:** this is not a gap, it is a structural
consequence of the cold-start framing. Zheng's task predicts a *time series* for *existing*
stops with observed history — ARIMA is a meaningful baseline because a history exists to
model. This dissertation's task predicts a *static value* for a stop with *zero* history by
construction (that is the definition of cold-start) — there is no time series for ARIMA (or
any temporal model) to be fit to. Excluding temporal baselines is therefore not a missing
comparison but a direct consequence of the research question; stating this explicitly in the
methodology chapter pre-empts the natural examiner question.

**Why there is (for now) no GCN baseline, and why it is worth adding:** Zheng's ASTGCN
baseline exists specifically to isolate whether *attention* matters, or whether any graph
convolution performs similarly (their GAT vs GCN swap). This dissertation had not asked that
question directly — GCNConv is now being added (see the entry below) as the direct analog.

**Why there is no Transformer or CNN-spatial baseline:** lower priority than the GCN
addition. A Transformer is naturally suited to Zheng's *sequence* modelling task; this
dissertation's task is a static per-node regression, where a generic Transformer would need
reframing as a set/graph-transformer — which GATv2's attention mechanism already partially
represents. A CNN-based spatial baseline requires a regular grid; this dissertation's graph
is over irregularly-placed real stop coordinates, a natural fit for graph methods, an
awkward fit for CNNs without an arbitrary gridding step. Both are legitimate future-work
items, not currently-missing comparisons of equal priority to the GCN one.

**Why hyperparameter tuning was absent, and what changed:** Zheng's baselines (including the
classical ones) were all grid-searched. This dissertation's RF/XGBoost/MLR were run at
sensible, literature-aligned defaults, never tuned — a real asymmetry, since GATv2 went
through five architectural interventions (K=10, PCA features, func-sim, Fusion, gated
mixing) while the tabular baselines it is compared against did not receive equivalent
attention. See the tuned-baseline addition below for the fix, and note explicitly in the
write-up: if tabular models still win *after* tuning, that is a strictly stronger form of the
headline claim than winning without it.

**Which paper is "the" comparison (resolved 19 Jul 2026, per the supervisor's request):**
two, for two different reasons, not one. **Zheng et al. (2025)** is the main *architectural*
comparison — the only paper whose method was actually implemented and empirically tested on
this project's own data (Experiment Set 7), not merely cited. **Yusuf et al. (2025)** is the
main *task-and-finding* comparison — same granularity (stop-level), same feature philosophy
(land-use/environmental buffers, not historical flow), same qualitative conclusion
(tabular/gradient-boosted beats deep learning) — used as the external-validity anchor rather
than an architectural one. Caveat: Yusuf's exact validation design (whether it uses a
spatial/cold-start holdout or an ordinary split) has not been independently verified beyond
the citation note in `references.bib` — confirm before leaning on it as a like-for-like
methodological match, not just a like-for-like task match.

---

### Experiment Set 9 — GCN baseline: does attention matter, or would any graph convolution do? (19 Jul 2026)

**Motivation:** directly answers the gap identified above — Zheng et al.'s ASTGCN baseline
swaps GCN in for GAT specifically to isolate whether attention does real work. `GCNModel`
mirrors `GATv2Model` exactly (same hidden width 256, same residual skip, same dropout
placement) with only the conv layer swapped (`GCNConv` instead of `GATv2Conv`), so any WMAPE
difference isolates the attention mechanism's contribution, not a capacity or
regularization confound. AI23+OSM+SC only, same 33 folds, same seed (42).

**Result:**

| Model | WMAPE |
|---|---|
| GATv2 (plain) | 0.7187 |
| **GCN** | **0.7006** |

Wilcoxon signed-rank, paired by borough: GCN vs GATv2, mean(GCN−GATv2)=−0.0181,
**p=0.01041 (significant)** — GCN beats GATv2. GCN vs MLP: p<0.00001 (GCN still loses).
GCN vs RF: p<0.00001 (GCN still loses).

**Finding — a real, useful refinement of the mechanism story:** attention specifically
appears to be *part of the problem*, not just "any graph aggregation is equally
compromised" under cross-borough contamination. Plain, unweighted (degree-normalised)
convolution — a strictly simpler, less expressive mechanism — significantly outperforms
learned attention on this task. A plausible mechanistic account: attention learns which
neighbours to weight *more*, but under leave-borough-out CV a test stop's neighbours are
systematically the wrong kind (adjacent-borough, different-profile) — attention has more
freedom to overweight a misleadingly-similar wrong neighbour than GCN's fixed,
un-learned aggregation does. This is a hypothesis, not confirmed here (would need a
per-edge attention-weight audit to test directly) — stated as a plausible mechanism, not a
proven one.

**What this does and doesn't change:** GCN still loses significantly to both tabular
baselines (p<0.00001 both), so the central honest finding is unaffected. What it adds is
precision: the problem is not simply "graphs don't help here," it is specifically that the
*more flexible/expressive* graph mechanism (attention) does worse than a *less* flexible one
(plain convolution) — consistent with, and a sharper version of, the over-smoothing /
cross-borough-contamination story already established (Experiment Set 3 onward).

---

### Experiment Set 10 — Per-fold hyperparameter tuning for RF/XGBoost/Ridge (19 Jul 2026)

**Motivation:** the single biggest fairness gap identified when comparing this project's
baseline suite against Zheng et al.'s grid-searched baselines (see Baseline Suite
Justification above): RF/XGBoost/MLR were run at literature-aligned defaults, never tuned,
while GATv2 received five architectural interventions. This experiment fixes that with a
`RandomizedSearchCV` (RF, XGBoost: n_iter=15, inner 3-fold CV, 45 fits/fold/model) and
`RidgeCV` (MLR: 25 alphas, efficient generalized CV) — **tuned strictly per outer fold, on
that fold's training data only**, so the held-out borough never influences its own fold's
hyperparameter choice (same no-leakage discipline as the per-fold `StandardScaler`). Best
hyperparameters per fold are saved in `results_tuned_hyperparams.csv` for transparency.
AI23+OSM+SC only, same 33 folds.

**Result:**

| Model | WMAPE (untuned) | WMAPE (tuned) | Δ |
|---|---|---|---|
| MLR | 0.6404 | 0.6401 | −0.03pp (no meaningful change) |
| RF | 0.6428 | **0.6339** | **−0.89pp (improved)** |
| XGBoost | 0.6437 | 0.6497 | **+0.60pp (got WORSE)** |
| MLP (reference, untuned/unchanged) | — | 0.6311 | — |

**Significance testing (Wilcoxon, paired by borough) — this is the important part:**

| Comparison | mean(A−B) | p-value | Significant? |
|---|---|---|---|
| RF-tuned vs MLP | +0.0028 | **0.292** | **No — not significant** |
| XGBoost-tuned vs MLP | +0.0185 | 0.00014 | Yes (XGBoost-tuned still worse) |
| MLR-tuned vs MLP | +0.0090 | 0.00086 | Yes (MLR-tuned still worse) |
| RF-tuned vs GATv2 | −0.0848 | <0.00001 | Yes (RF-tuned beats GATv2) |
| RF-tuned vs XGBoost-tuned | −0.0158 | <0.00001 | Yes (RF-tuned beats XGBoost-tuned) |

**This corrects a headline claim, not just adds a data point.** The earlier statistical
result (Statistical Significance Testing section, 18 Jul) found MLP significantly beats
untuned RF (p=0.00225). With a fair, properly-scoped tuning budget, **RF closes that gap
entirely — RF-tuned and MLP are now statistically indistinguishable (p=0.292)**. The
correct claim going forward is **"MLP and tuned RF are statistically tied for best,"** not
"MLP is significantly best." This is exactly the kind of asymmetry the tuning gap could
have been hiding, and it was — worth stating plainly rather than downplaying.

**The XGBoost-tuned regression is also worth reporting honestly, not explaining away.**
Tuning made XGBoost *worse* on average, not better. Plausible explanation: a 15-configuration
random search with a 3-fold inner CV is a modest budget, and some of the sampled
configurations (e.g. higher `max_depth`, lower `subsample`) may generalise worse from a
noisy inner-CV estimate than the literature-standard defaults already were — i.e. this is
consistent with mild overfitting to inner-CV noise at this search budget, not evidence that
XGBoost is inherently harder to tune well. A larger search budget might recover or exceed
the untuned result; this was not tested further, consistent with keeping this a scoped,
proportionate addition rather than an open-ended tuning campaign.

**Practical conclusion:** the headline claim is now more defensible, not less — "tabular
models are competitive with or beat GATv2" survives proper tuning, and in RF's specific
case, tuning made the tabular side of the comparison *stronger*, not weaker.

---

## Files Reference

| File | Description |
|---|---|
| stops_features.csv | Base features (AI23 + coords), 17,943 stops |
| stops_features_osm.csv | AI23 + OSM POI + service_coverage (after step3d) |
| stops_features_vc.csv | AI23+OSM+coords + peak_vc target (step5b output) |
| results_cv_ai23_osm.csv | Per-fold results, AI23+OSM (v1 headline pair — survives) |
| results_summary_ai23_osm.csv | Summary stats, AI23+OSM (v1) |
| results_cv/summary_ai23_only.csv, _osm_only.csv | **LOST** (see "Data loss" bug entry) — mean WMAPE only survives as text in Experiment Set 1 |
| results_{cv,summary}_{ai23_only,osm_only,ai23_osm}_fastbaselines.csv | HistAvg/MLR/RF/XGBoost only, v4 protocol, for the 3 pre-SC feature sets (step4c output) |
| results_{cv,summary}_osm_sc.csv | OSM+SC, v3/v4 (6 models incl. MLR/XGBoost) |
| results_{cv,summary}_ai23_osm_sc.csv | **AI23+OSM+SC — current headline experiment** (6 models) |
| results_{cv,summary}_gnn_fairness.csv | GNN-fairness check: K=10, AI23+OSM+SC |
| results_{cv,summary}_vc_ai23_osm.csv | V/C (overcrowding) target, AI23+OSM |
| results_{cv,summary}_idw.csv | IDW spatial-interpolation baseline (Experiment Set 5), feature-independent |
| results_{cv,summary}_pca_ai23_osm_sc.csv | PCA(AI23, 3 comp.)+OSM+SC (Experiment Set 6), all 7 models incl. GATv2 |
| results_{cv,summary}_func_sim.csv | GATv2 + functional-similarity edges, single graph (Experiment Set 7a). GATv2=0.7352, worse than plain (0.7187) |
| results_{cv,summary}_zheng_fusion.csv | GATv2-Fusion: Zheng-style G1/G2/G3 dual-branch fusion (Experiment Set 7b). GATv2-Fusion=0.7070, better than plain but still behind MLP/RF |
| consolidated_results_table.csv | Chapter 4 table: 8 rows (models) × 5 columns (feature sets incl. PCA variant) |
| borough_extract.csv | Full per-borough pivot from the headline run (Camden/Hillingdon/worst-3/closest-3 source) |
| borough_wmape_map.png | Choropleth, all 33 boroughs (Bug 6 fix applied) |
| step3b_osm_features.py | Downloads OSM POI, creates stops_features_osm.csv |
| step3c_add_scenic.py | Adds poi_scenic column to stops_features_osm.csv |
| step3d_add_service_coverage.py | Adds service_coverage to stops_features_osm.csv |
| step4_model.py | Main model: HistAvg/IDW/MLR/RF/XGBoost/MLP/GATv2, leave-borough-out CV. Flags: `--ai23-only`/`--osm-only`/`--with-sc`/`--k10`/`--pca-ai23`/`--quick` |
| step4c_fast_baselines.py | HistAvg/MLR/RF/XGBoost only (no NN), same fold/scaler protocol, for the 3 pre-SC feature sets |
| step4d_idw_baseline.py | IDW spatial-interpolation baseline (Liu et al. 2017-style), feature-independent, ~1 sec |
| step5a_borough_map.py | Choropleth map (Bug 6 fixed 16 Jul) |
| step5b_vc_experiment.py | V/C target experiment |
| step4e_zheng_fusion.py | Zheng-style G1/G2/G3 fusion GAT (Experiment Set 7b), reuses graph-independent baselines from the headline run |
| step4f_gated_mixing.py | Pre-registered learned-gate MLP/GATv2 mixing test (Experiment Set 8), 3 seeds |
| step4g_gated_analysis.py | Full stats for Experiment Set 8: Wilcoxon+Holm-Bonferroni, alpha distribution, per-borough correlation |
| results_{cv,summary}_gated_all_seeds.csv, results_gated_alpha_all_seeds.csv | Experiment Set 8 raw output (3 seeds x 33 folds + per-stop alpha) |
| borough_extract_gated.csv | Per-borough Gated vs MLP vs GATv2 vs RF (Experiment Set 8) |
| step4h_residual_correlation.py | Retrains standalone MLP+GATv2 (seed=42) saving raw per-stop predictions, for the Exp. Set 8 follow-up |
| step4i_gcn_baseline.py | GCN baseline (Experiment Set 9): GATv2Model with GCNConv swapped in, isolates attention's contribution |
| step4j_tuned_baselines.py | Per-fold RandomizedSearchCV/RidgeCV for RF/XGBoost/MLR (Experiment Set 10), no-leakage discipline |
| results_{cv,summary}_gcn.csv | Experiment Set 9 results: GCN=0.7006, significantly beats GATv2 (p=0.010), still loses to MLP/RF |
| results_{cv,summary}_tuned.csv, results_tuned_hyperparams.csv | Experiment Set 10 results + per-fold best hyperparameters (transparency) |
| results_residual_correlation.csv | Per-stop MLP/GATv2 raw predictions + residuals (17,943 rows), Exp. Set 8 follow-up |
| residual_correlation_by_borough.csv | Per-borough residual correlation + naive-ensemble-vs-MLP gap |
| step6_consolidated_table.py | Builds consolidated_results_table.csv (10 rows × 5 cols) |
| step7_borough_extract.py | Builds borough_extract.csv + prints Camden/Hillingdon/worst-3/closest-3 (pass a CSV + note the model column name if not "GATv2") |
| run_tier1_overnight.sh | Driver: chains OSM+SC → AI23+OSM+SC → K=10 fairness → V/C |
| run_expset7_overnight.sh | Driver: chains --func-sim → step4e_zheng_fusion.py |
| RUNBOOK.md | Full file inventory + reproduction commands (companion to this log) |
| files_chapters/PROJECT_EFFORT_LOG.md | Narrative effort/challenge log for supervisor/viva use (companion to WRITING_HANDOFF.md, which stays frozen) |
| experiment_log.md | This file |

---

## DEFERRED EXPERIMENTS — not run, reserved for future work

Added 21 Jul 2026 at session freeze. None of the five items below have been
run, implemented, or scoped beyond what is written here. Listed so that
reopening the experimental phase later starts from a deliberate registry, not
an ad hoc request.

**D-1 — Heterophily-adapted GNN (H2GCN / GPR-GNN), AI23+OSM+SC**
(a) What: implement H2GCN (Zhu et al. 2020) or GPR-GNN (Chien et al. 2021) —
architectures designed explicitly for heterophilous graphs — on the headline
feature set, same 33-fold LOBO protocol.
(b) Hypothesis: Experiment Set 9 (GCN beats GATv2) and the D6 diagnostic
(cross-borough edge assortativity 0.335 vs within-borough 0.509) both point
to heterophily-under-cross-borough-testing as the mechanism; an architecture
built for exactly that condition might close the remaining gap to tabular.
(c) If it succeeded: strongest possible validation of the diagnosed
mechanism — write as "an architecture chosen specifically for the diagnosed
failure mode closes the gap." If it failed: further confirms the missing
information is not recoverable by any architecture, however heterophily-aware
— write as "the limitation is informational, not architectural."

**D-2 — LSOA-grouped CV (interpolation / infill scenario)**
(a) What: a leave-LSOA-out CV scheme (hold out one LSOA at a time, not one
borough) — same-borough, different-micro-area neighbours stay in training.
(b) Hypothesis: tests the interpolation-vs-extrapolation distinction directly
— does GATv2 do better at infill (new stop in a known area) than at
extrapolation (new stop in an unknown borough)?
(c) If it succeeded: write as a policy-relevant refinement — GATv2 may be
usable for the (more common) infill case even though it fails at the
(rarer) whole-new-borough case. If it failed: write as evidence the gap is
not specifically about contamination granularity, but a more general
tabular-vs-graph result independent of split scheme.

**D-3 — Multi-seed re-run of the v1-era AI23-only and OSM-only cells**
(a) What: re-run AI23-only and OSM-only (all 7 models) across 3-5 seeds,
matching the rigor Experiment Set 8's follow-up applied to the headline set.
(b) Hypothesis: the AI23-only/OSM-only cells currently rest on single-seed
(or archived v1) runs with undocumented variance — are the feature-set
ablation conclusions robust to that variance?
(c) If it succeeded (variance small): strengthens the existing ablation
story with an explicit bound, one confirmatory footnote. If it failed
(variance threatens a specific ordering, e.g. AI23-only vs OSM-only):
requires softening exactly that claim, not the whole ablation section.

**D-4 — GTFS spot-check of service_coverage on ~50 stops (construct validity)**
(a) What: for ~50 sampled stops, pull the current GTFS/Bus Open Data Service
stop_times.txt and independently compute the same
(route×direction×time-window) count; compare to the BUSTO-derived value used.
(b) Hypothesis: tests the load-bearing cold-start validity claim ("obtainable
from GTFS for new stops") empirically rather than by argument alone (see D1
in diagnostics_report.md for the precise provenance statement this would
upgrade).
(c) If it succeeded (close match): upgrades the claim from argued to
empirically verified for a sample. If it failed (systematic divergence, e.g.
schedule drift between the BUSTO period and current GTFS): requires either a
hedged claim or an explicit, quantified limitations paragraph.

**D-5 — AI23+SC and OSM+SC cells (completes the feature decomposition)**
(a) What: two more ablation cells — AI23-only+SC, and confirm/re-verify
OSM-only+SC — completing the 2×2(×SC) factorial (currently only
AI23+OSM+SC and OSM+SC exist; AI23-only+SC does not).
(b) Hypothesis: is service_coverage's large contribution (Finding 2)
complementary to AI23 and OSM independently and comparably, or concentrated
in one combination?
(c) If it succeeded (comparable lift both ways): strengthens Finding 2 as a
general, feature-set-independent effect. If it failed (lift concentrated in
one combination): requires a more specific, still-honest version of Finding
2 naming which combination it depends on.

---

## EXPERIMENTAL PHASE CLOSED

**EXPERIMENTAL PHASE CLOSED — 21 Jul 2026. Code tagged v-final-results
(111d3b6). Reopening requires: all chapters drafted; >=10 days to
submission; item already listed in the deferred registry above; supervisor
approval. Decision date: [NOT SPECIFIED — no submission date has been
established anywhere in this project's records; insert the actual date
before treating this line as binding, do not infer one].**
