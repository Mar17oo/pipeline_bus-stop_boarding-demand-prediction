# DISSERTATION HANDOFF — Paste this into any new chat to restore full context
# CE902 MSc Dissertation · University of Essex · Supervisor: Dr. Vishal K. Singh
# Student: Maria Bautista · Module: CE902

---

## WHAT THIS IS

**Goal:** Predict weekday boarding demand at London bus stops with **no historical
passenger data** (cold-start problem). A new bus stop or route has no travel history —
conventional models fail. This work predicts demand from the surrounding environment.

**Model:** Inductive GATv2 (Graph Attention Network v2, Brody et al. 2022) over a
spatial+route multigraph. "Inductive" means it generalises to stops it never saw during
training — following Hamilton et al. 2017 (GraphSAGE).

**Comparison paper:** Zheng et al. (2025) "MF-STGAT: Predicting Regional-Level Bus Stop
Passenger Flow." Uses GAT+LSTM, requires historical flow as node features, uses POI for
EDGE construction (Pearson ≥ 0.8 functional similarity). CANNOT handle cold-start.
Our contribution: same graph attention idea, made inductive, zero historical data needed.

**Evaluation:** Leave-borough-out spatial CV, 33 London boroughs. Each fold removes ALL
stops from one borough — model must predict them from zero. Mandatory design for spatial
data (Roberts et al. 2017, Ecography). Cannot reduce fold count without invalidating paper.

**Metric:** WMAPE (Weighted Mean Absolute Percentage Error). Lower = better.

---

## DATASETS

### Primary demand data — BUSTO 2023/24 (TfL Network Statistics)
17,943 London bus stops. Survey-weighted averages (decimal boardings, not raw counts).

Raw CSV columns per quarter-hour row:
  YEAR, DAY_TYPE, TIMEBAND, QHr (HH:MM:SS time slot), ROUTE, DIRECTION,
  STOPSEQUENCE, STOPCODE, STOPNAME, Boardings, Alightings, Load,
  Capacity, Seats, V/C (volume/capacity ratio)

Weekday, Saturday, Sunday files all available. Only Weekday used so far.
Currently target = total_boardings (sum across all route/direction/QH per stop).

### Node features used

| Feature group | Source | Count | Notes |
|---|---|---|---|
| AI23 accessibility | ONS/UBDC (Verduzco Torres 2024) | 8 | LSOA-level, 30-min PT access to employment, hospitals, GP, supermarkets, pharmacies, primary schools, secondary schools, built-up areas |
| OSM POI | OpenStreetMap via osmnx | 6 | Counts within 500m buffer: poi_residential, poi_shopping, poi_company, poi_education, poi_entertainment, poi_scenic |
| Service coverage | Derived from BUSTO | 1 | n_route_dir_qhr_rows = distinct (route×direction×QH) combinations per stop. Proxy for planned service frequency. For new stops: use GTFS. |
| Coordinates | TfL stop locations | 2 | lat, lon — essential: 96% of stops share an LSOA → identical AI23 values without coords |

### Feature files on disk
- `stops_features.csv` — AI23 + coords only (10 cols used as features)
- `stops_features_osm.csv` — AI23 + OSM + service_coverage + coords (17 cols)
- `busto_stop_level_boardings.csv` — aggregated demand (17,943 stops)
- `stops_with_coords.csv` — intermediate pipeline file (before LSOA join)
- `route_edges.csv` — precomputed route connectivity edges (43,220 directed edges)

### Graph structure (~124,000 directed edges total)
- KNN edges: K=5 nearest stops by Haversine distance (bidirectional)
- Route edges: consecutive stops on same route+direction from BUSTO (bidirectional)

---

## MODEL ARCHITECTURE (current, all bugs fixed)

```python
class GATv2Model(torch.nn.Module):
    def __init__(self, in_ch):
        self.c1   = GATv2Conv(in_ch, 64, heads=4, dropout=0.15, concat=True)
        self.c2   = GATv2Conv(256, 1, heads=1, dropout=0.15, concat=False)
        self.skip = torch.nn.Linear(in_ch, 256, bias=False)  # residual — critical

    def forward(self, x, edge_index):
        h = F.elu(self.c1(x, edge_index)) + self.skip(x)  # graph + residual
        h = F.dropout(h, p=0.15, training=self.training)
        return self.c2(h, edge_index).squeeze(-1)
```

**Training:**
- Loss: Huber(delta=0.5) on log1p(boardings) — robust to outlier high-demand stops
- Optimizer: Adam, LR=5e-4, weight_decay=1e-4
- EPOCHS=500, PATIENCE=20 checks every VAL_EVERY=5 epochs (effective patience = 100 epochs)
- Val split: stratified by 5 demand quantiles so early stopping sees the full demand range
- Scaler: StandardScaler fitted on training stops only (per fold, no leakage)

**Baselines:**
- HistAvg: mean boardings of training stops
- RF: RandomForest(150 trees, max_features=sqrt, min_samples_leaf=5, n_jobs=-1) on log1p
- MLP: 3-layer (in_ch → 256 → 128 → 1), same capacity, no graph, Huber loss

**Inductive inference:** Train on subgraph of training-borough stops only.
At test time, held-out borough stops are added; they aggregate features from
nearest training-set neighbours via the KNN+route multigraph. Labels never seen.

---

## ALL EXPERIMENTS COMPLETED — RESULTS

### Experiment 1: Feature Ablation (v1 — had validation-mode bug, GATv2 worse)
Config: MSE loss, DROPOUT=0.3, LR=1e-3, no skip connection, PATIENCE=40

| Feature set | Dims | HistAvg | RF | MLP | GATv2 |
|---|---|---|---|---|---|
| AI23-only | 10 | 1.0822 | 0.8121 | 0.8114 | 0.8241 |
| **AI23+OSM** | **16** | **1.0822** | **0.7975** | **0.7983** | **0.8058** |
| OSM-only | 8 | 1.0822 | 0.8062 | 0.8058 | 0.8259 |

Result files: results_summary_ai23_only.csv, results_summary_ai23_osm.csv,
              results_summary_osm_only.csv
Per-borough files: results_cv_ai23_only.csv, results_cv_ai23_osm.csv,
                   results_cv_osm_only.csv (33 boroughs × 4 models = 132 rows each)

### BEST RESULT SO FAR: RF with AI23+OSM = 0.7975 WMAPE (26% better than HistAvg)

### Pending experiments (v3 — all bugs fixed, Huber loss, stratified val, faster)
Run these overnight (~1.5h each):
```powershell
cd "C:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"
python step3d_add_service_coverage.py
python step4_model.py --osm-only --with-sc
copy results_cv_multigraph.csv results_cv_osm_sc.csv
copy results_summary_multigraph.csv results_summary_osm_sc.csv
python step4_model.py --with-sc
copy results_cv_multigraph.csv results_cv_ai23_osm_sc.csv
copy results_summary_multigraph.csv results_summary_ai23_osm_sc.csv
python step5b_vc_experiment.py
python step5a_borough_map.py
echo "ALL DONE"
```

Expected output files: results_summary_osm_sc.csv, results_summary_ai23_osm_sc.csv,
                        results_summary_vc_ai23_osm.csv, borough_wmape_map.png

---

## BUGS FOUND AND FIXED (chronological)

| # | Severity | Bug | Fix | Impact |
|---|---|---|---|---|
| 1 | Low | Dead global scaler computed but never used | Removed | Code clarity |
| 2 | Low | Pre-warm built dummy model with wrong feature count | Use len(ALL_FEAT_COLS) | Saved 2 min per run |
| 3 | **CRITICAL** | Validation loss used stochastic training-mode output (`out` from dropout-active pass) → noisy early stopping | Recompute forward pass in eval mode for val | GATv2 improved ~1.4pp WMAPE |
| 4 | Low | Terminal log printed wrong output filename | Fixed log message | Clarity |
| 5 | Medium | No residual skip connection → GATv2 forced through graph even when graph adds noise | Added `self.skip = Linear(in_ch, 256)` | GATv2 improved further |
| 6 | Medium | Random val split → majority low-demand stops dominated early stopping signal | Stratified 5-quantile val split | Better convergence signal |
| 7 | — | MSE loss sensitive to outlier high-demand stops | Replaced with Huber(delta=0.5) | More robust learning |
| 8 | — | Val checked every epoch → 2× compute per epoch wasted | Check val every 5 epochs, PATIENCE=20 | ~40% faster training |

---

## WHY GATv2 UNDERPERFORMS RF/MLP — ROOT CAUSE ANALYSIS

GATv2 is consistently 1–2pp worse than RF across all experiments. This is NOT a
model failure — it is a known limitation of GNNs in this spatial setting:

**The cross-borough aggregation problem:**
In leave-borough-out CV, ALL test stops belong to the held-out borough.
Their K=5 nearest training neighbours are stops from ADJACENT boroughs with
different land-use and demand profiles (e.g., Hillingdon test stops aggregate
from Ealing/Hounslow training stops — different suburban density).
GATv2 gets contaminated cross-borough signal. RF uses features directly — immune.

**Evidence:** Per-borough results show GATv2 loses worst in geographically
isolated outer boroughs (Hillingdon: GATv2=0.890, RF=0.857) but nearly matches
RF in well-connected inner boroughs (Camden: GATv2=0.756, RF=0.756).

**Academic framing:** Consistent with Grinsztajn et al. (2022, NeurIPS) and
Shwartz-Ziv & Tishby (2022) — GNNs don't reliably outperform tabular methods
when graph structure is noisy relative to feature signal.

**This IS a valid dissertation finding**, not a failure:
"For cross-sectional cold-start prediction, graph structure provides marginal
additional signal over tabular baselines when evaluated under strict spatial CV."

---

## IDEAS TO IMPROVE RESULTS — PRIORITISED

### High impact, low effort (try these first)
1. **service_coverage feature** (PENDING): n_route_dir_qhr_rows = service frequency
   proxy. Strongest missing predictor. Expected -3 to -6pp WMAPE. Run step3d then
   step4_model.py --with-sc to test.

2. **Time-period disaggregation**: Use TIMEBAND from raw BUSTO CSVs to create
   AM_peak_boardings as target instead of total_boardings. AM peak demand is more
   predictable (commuter patterns). Likely lower WMAPE than daily total.
   Requires: new aggregation script summing only TIMEBAND=AM rows per stop.

3. **V/C ratio as target** (PENDING, step5b): Predict peak overcrowding ratio
   (90th-pct V/C) instead of raw boardings. V/C is bounded (0–1.5), less skewed,
   more policy-relevant (TfL threshold = 0.85). Could give cleaner GATv2 performance.

### Medium impact, medium effort
4. **Saturday data**: Add Saturday BUSTO files. Saturday demand has different
   spatial pattern (shopping/leisure). Could train separate models or add
   day_type as a feature.

5. **Weighted edges**: Replace binary KNN edges with distance-weighted edges
   (weight = 1/haversine_distance). GATv2's attention would then have a stronger
   prior to upweight nearby stops. Currently all edges are equally weighted.

6. **Increase K from 5 to 10**: More neighbours = more aggregation context.
   Slightly increases memory/compute but may help GATv2 find useful neighbours
   even when some cross-borough ones are noisy.

7. **Add population density**: Census 2021 population per LSOA is free and open.
   High density → higher footfall → more demand. Simple to add via LSOA join.

### Lower impact / research-only
8. **Functional similarity edges** (like Zheng) — CORRECTED 17 Jul 2026: earlier
   text here said this needs demand correlation between stops (infeasible for
   cold-start test stops). That was a misreading of the source paper. Zheng
   et al. (2025) actually compute functional similarity from Pearson
   correlation of **POI count vectors** (residences, shopping, scenic,
   company, education, entertainment within a 500m buffer) — a static
   land-use feature, not demand. We already have those 6 counts
   (poi_residential/shopping/company/education/entertainment/scenic), so this
   IS computable for cold-start test stops. Re-evaluated and now being tried
   (see experiment_log.md Experiment Set 7) — main remaining concern is
   circularity (edges built from features already given directly to the
   model as node inputs), not cold-start infeasibility.

9. **3-layer GATv2**: Add one more attention layer for richer aggregation.
   Risk: over-smoothing (node reps become too similar). Marginal benefit expected.

10. **CatBoost/XGBoost instead of RF**: Gradient boosting typically outperforms RF
    on tabular data. Could lower baseline WMAPE by ~1pp with same features.

---

## PIPELINE SCRIPTS (in execution order)

| Script | Input | Output | Run time |
|---|---|---|---|
| step1_aggregate_busto.py | data/ BUSTO CSVs | busto_stop_level_boardings.csv | 2 min |
| step2_join_coordinates.py | busto + data/Bus_Stops.csv | stops_with_coords.csv | 1 min |
| step3_lsoa_features.py | stops_with_coords + data/access_*.csv | stops_features.csv | 2 min |
| step3b_osm_features.py | stops_features | stops_features_osm.csv (+5 OSM cols) | 30 min |
| step3c_add_scenic.py | stops_features_osm | stops_features_osm.csv (+scenic) | 10 min |
| step3d_add_service_coverage.py | stops_features_osm + busto | stops_features_osm.csv (+sc) | 5 sec |
| **step4_model.py** | stops_features*.csv + data/ | results_cv/summary_multigraph.csv | ~1.5h/run |
| step5a_borough_map.py | results_cv_ai23_osm.csv + ONS API | borough_wmape_map.png | 1 min |
| step5b_vc_experiment.py | stops_features_osm + data/ | results_cv/summary_vc_ai23_osm.csv | ~1.5h |

**step4_model.py flags:**
```
--ai23-only   Force AI23 features only (reads stops_features.csv)
--osm-only    OSM features only (no AI23)
--with-sc     Add service_coverage to any config (after step3d run)
--quick       5 folds only for fast testing (~15 min)
[no flags]    AI23 + OSM (default, reads stops_features_osm.csv)
```

---

## WHAT'S DONE vs PENDING

### Done ✅
- [x] Full data pipeline (steps 1-3d)
- [x] 33-fold ablation: AI23-only, AI23+OSM, OSM-only
- [x] All 5 bugs found and fixed
- [x] Architecture improved: skip connection, Huber loss, stratified val, faster training
- [x] Dataset evaluated: GTFS (schedule only, no demand), TTM2023 (travel times, no demand),
      Active Travel data (Glasgow only, not London), Oyster (not public for bus stops)
- [x] Comparison with Zheng et al. established (different task, not directly comparable)
- [x] experiment_log.md: complete record of all bugs + experiments
- [x] supervisor_comparison_results.xlsx: supervisor-ready Excel comparison

### Pending ⏳
- [ ] OSM + service_coverage experiment (--osm-only --with-sc)
- [ ] AI23 + OSM + service_coverage experiment (--with-sc) [MOST IMPORTANT]
- [ ] V/C overcrowding prediction (step5b)
- [ ] Borough choropleth map (step5a, 1 minute, professor requested this)
- [ ] Update experiment_log with v3 results
- [ ] Chapter 4 write-up (results + ablation table)
- [ ] Chapter 2 gap statement vs Zheng

---

## FILE INVENTORY (all current CSVs — clean, no duplicates)

| File | Rows | Description | Needed? |
|---|---|---|---|
| busto_stop_level_boardings.csv | 17,943 | Aggregated demand + service_coverage source | YES |
| stops_with_coords.csv | 19,614 | Intermediate: stops+coords before LSOA join | Pipeline only |
| stops_features.csv | 17,943 | AI23 + coords features | YES (--ai23-only) |
| stops_features_osm.csv | 17,943 | AI23 + OSM + service_coverage + coords | YES (main) |
| route_edges.csv | 43,220 | Precomputed route edges (src,dst) | Analysis |
| results_cv_ai23_only.csv | 132 | Per-borough: AI23-only, 33 boroughs × 4 models | YES |
| results_summary_ai23_only.csv | 4 | Summary stats: AI23-only | YES |
| results_cv_ai23_osm.csv | 132 | Per-borough: AI23+OSM ← BEST experiment | YES |
| results_summary_ai23_osm.csv | 4 | Summary stats: AI23+OSM ← BEST | YES |
| results_cv_osm_only.csv | 132 | Per-borough: OSM-only | YES |
| results_summary_osm_only.csv | 4 | Summary stats: OSM-only | YES |

6 junk files deleted: results_cv.csv (old buggy run), results_summary.csv (same),
results_cv_multigraph.csv (duplicate of osm_only), results_summary_multigraph.csv
(duplicate), results_cv_multigraph_quick.csv (5-fold test), results_summary_multigraph_quick.csv

---

## HONEST ASSESSMENT FOR THE PAPER

**What you CAN claim:**
- Cold-start bus stop demand prediction is feasible without historical data
- AI23 + OSM features together outperform either alone (complementary signal)
- All ML models beat naive baseline by 26% WMAPE under strict spatial CV
- GATv2 provides an inductive spatial framework competitive with tabular baselines
- OSM POI features improve prediction consistently across all model types

**What you CANNOT claim:**
- GATv2 outperforms RF/MLP (it doesn't — 1-2pp worse consistently)
- Results are comparable to Zheng et al. (different task: temporal vs cold-start)
- WMAPE of ~0.80 represents strong predictive accuracy (it is high — cold-start is hard)

**Viva concepts to know (2 sentences each):**
- Inductive vs transductive learning (GCN=transductive, GATv2/GraphSAGE=inductive)
- Message passing / neighbourhood aggregation
- Why leave-borough-out CV, not random splits (spatial autocorrelation, Tobler's Law)
- WMAPE not MAPE (zero-demand stops break MAPE)
- Over-smoothing in GNNs (too many layers collapse node representations)
- Cold-start from recommender systems applied to urban transport planning
