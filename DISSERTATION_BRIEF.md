# Dissertation Project Brief — Maria Bautista

> **HOW TO USE:** Paste this whole document at the start of any new chat with
> Claude to rebuild context instantly. Update it as decisions change.
> This is your single source of truth — your "save file."

---

## 1. One-line summary
Predict **bus-stop passenger demand for geographically unseen ("cold-start") areas**
using an **inductive GATv2 graph neural network** over a spatial+functional multigraph.
**SCOPE NOW NARROWED: one primary RQ + one ablation sub-question. No optimisation.**
Module: CE902. Supervisor: Vishal K. Singh (University of Essex). MSc (Data Science/AI).

## 2. The core idea
Conventional demand models are **transductive**: they need historical data at every
stop, so they fail when a city adds a new neighbourhood (the "Urban Cold-start").
This project builds an **inductive** model that infers demand from land-use / POIs /
accessibility of surrounding areas, so it can reason about stops with no travel history.

## 3. SCOPE DECISION (important)
- **DROPPED:** RQ3 / NSGA-II / route optimisation. It is a different discipline
  (operations research, not ML) and doubles the risk. → moved to "future work".
- **KEPT:** one focused predictive dissertation with a rigorous ablation.
- Rationale: an Essex MSc rewards DEPTH on one method over breadth. One narrow,
  well-executed question beats three shallow ones. The non-trivial part is the
  inductive, spatially-held-out setup (predicting for unseen places).

## 4. Research questions (FINAL — narrowed)
- **Primary RQ:** Can an inductive GATv2 predict bus-stop demand for stops in
  geographically unseen areas more accurately than (a) Historical Average and
  (b) a non-graph Random Forest?
- **Sub-question (the scientifically interesting bit):** How much of any improvement
  comes from the GRAPH STRUCTURE itself vs. the node FEATURES alone?
  → isolated by ablation: GATv2 vs RF (same features, no graph) vs features-with-edges-removed.

**Metrics:** WMAPE + RMSE + MAE. NEVER plain MAPE (zero-flow stops break it; both
reference papers reject it).
**Validation:** leave-borough-out spatial cross-validation (simulates unseen areas).

## 5. Datasets (ALL OPEN — confirmed)
- **BUSTO autumn 2023/24** (TfL flat-file downloads → Network Statistics): bus boardings/alightings/
  loadings per stop. Survey-derived TYPICAL-DAY estimate (not 15-min taps).
  → **PRIMARY DEMAND TARGET.**
- **PTAI22 / AI23** (UBDC, Verduzco Torres & McArthur 2024): accessibility indicators,
  41,729 LSOA/DZ. Zenodo 10.5281/zenodo.8037156. Already loaded in MA336. → FEATURES.
- **OSM POIs** (Overpass): functional features + functional graph edges. → FEATURES/EDGES.
- **Bus stop locations & routes** (TfL): node coords + connectivity edges.
- **Census 2021 / IMD**: population density, deprivation. → FEATURES.
- **RODS** (TfL, OGL): rail OD matrix — OPTIONAL secondary, only if time allows.

**NOTE:** TfL Unified API has NO historical per-stop demand (only real-time/supply).
Demand lives in the FLAT-FILE downloads (BUSTO/RODS), not the API.

## 5b. REFINED DATA DECISIONS (latest)
- Use **BUSTO autumn 2023/24** (NOT 2025/26) to align with AI23's early-2023 features.
- Use **AI23 only** as features; **DROP PTAI22** (only needed for temporal task, which
  we no longer do). Simplifies data + removes COVID-era comparability issue.
- Join key = **2011** LSOA geo_code (AI23 uses 2011 codes) -> use 2011-based ONS lookup.
- Use **employment_all** (not the 18 industry sub-types) + hospitals/GPs/schools/supermarkets.
- See LIMITATIONS_AND_CONSTRAINTS.md for the full defence list.

## 6. My prior work (assets)
- **MA336 project**: full PTAI22/AI23 load/clean/merge pipeline + DT/RF baselines.
  REUSE the pipeline. KNOWN FLAW to fix and FEATURE in the dissertation: random
  train/test split → spatial leakage → inflated R²=0.98. Fix = spatial CV.
- **CE902 proposal**: framing, multigraph formalism, GATv2, cold-start narrative.

## 7. Experiment ladder (Chapter 5)
Historical Average → Random Forest (no graph) → GAT (transductive) → GATv2 (inductive),
all under leave-borough-out spatial CV. The RF-vs-GATv2 gap answers the sub-question.

## 8. Chapter structure
1. Introduction (cold-start problem; target = BUSTO).
2. Literature (ST-GNN demand prediction → inductive learning → graph attention).
3. Data & multigraph construction (MA336 pipeline + BUSTO/OSM/census).
4. Methodology (GATv2; spatial-validation design — frame the MA336 leakage fix here).
5. Experiments & results (the ladder + ablation).
6. Discussion (what the graph adds; equity angle: high-demand + low-access zones).
7. Limitations (BUSTO = survey-averaged) + conclusions + future work (incl. NSGA-II).

## 9. CONCEPTS TO DEFEND IN VIVA (learn each to a 2-sentence explanation)
1. **Inductive vs transductive learning** — THE core concept. Why GCN is transductive,
   why GraphSAGE/GATv2 generalise to unseen nodes.
2. **Message passing / neighbourhood aggregation** — how a node's representation is
   built from neighbours; why this lets the model reason about a new node.
3. **Attention in GAT/GATv2** — learned per-neighbour weights vs fixed convolution;
   why GATv2 fixes GATv1's STATIC attention (Brody et al.).
4. **Spatial autocorrelation & data leakage** — Tobler's First Law; why random splits
   leak; why leave-borough-out is correct. (YOUR DEFENCE WEAPON — found own mistake.)
5. **Why WMAPE not MAPE** — MAPE explodes at zero-demand stops; WMAPE weights by volume.
6. **Cold-start problem** — from recommender systems; applied to urban geography.
7. **Over-smoothing** — too many GNN layers collapse node reps; why use few layers.
8. **Generalisation / bias-variance** — why 0.98 on a random split drops on a spatial
   split; the random split overestimates generalisation to new distributions.

## 10. Reading list (10 you understand DEEPLY > 50 you skim)
**CORE (method extends this):** Zheng et al. (2025), MF_STGAT — bus stops, POI
functional graph, GAT. My contribution = make it inductive.
**Foundational (read FIRST):**
- Hamilton, Ying & Leskovec (2017), GraphSAGE, NeurIPS — MOST IMPORTANT; defines inductivity.
- Brody, Alon & Yahav (2022), GATv2, ICLR — the model I use.
- Velickovic et al. (2018), GAT, ICLR — predecessor.
- Kipf & Welling (2017), GCN, ICLR — the transductive baseline I contrast against.
**Bridge concept:**
- Simini et al. (2021), Deep Gravity, Nature Communications — land-use predicts mobility
  (justifies my whole premise).
**Method/validation:**
- Roberts et al. (2017), spatial cross-validation, Ecography — defends my validation design.
**Supporting:**
- Xie et al. (2025), DMSTGCN — multigraph fusion + gating.
- Verduzco Torres & McArthur (2024a, 2024b) — PTAI22/AI23 data papers.

## 11. Resolved decisions
- [x] Cold-start hold-out = **leave-borough-out** (whole London boroughs held out
      from training; predict their stops). Cleaner + more defensible than the
      Elizabeth Line corridor. Elizabeth Line mentioned as real-world MOTIVATION
      in Ch.1 and as future validation, but NOT the main test set.

## 12. Immediate next actions
1. Register + download BUSTO from TfL open-data downloads (Network Statistics).
2. Confirm BUSTO column names/format → fixes exact target variable.
3. Swap synthetic load_stops() in dissertation_starter.py for real merge.
4. Read GraphSAGE + GATv2 (the two that define the thesis).
5. Rerun MA336 experiment with leave-region-out CV → first dissertation figure +
   empirical motivation for the cold-start chapter.

## 13. Files in my project folder
- DISSERTATION_BRIEF.md (this file — the save file)
- dissertation_starter.py (runnable scaffold: multigraph, spatial CV, baselines, GATv2)
- dissertation_main.tex (LaTeX skeleton to write into)
- references.bib (bibliography)
- PAPER_ANALYSIS_PLAN.md (how to read each paper deeply)
