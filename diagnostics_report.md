# Diagnostics Report — Read-Only Verification

Generated 19 Jul 2026. Read-only session: no retraining, no new experiments, no
modification of any existing file. D4 and D5 were not received (the task
message was truncated mid-D3, after "... or one poole[d]"; D3's second half is
inferred — see note) — this report covers D1–D3 only. Append D4–D5 once the
complete question text is available.

---

## D1 — Provenance of `service_coverage`

**D1.1 — exact lines, and confirm/refute the row-counting claim**

`service_coverage` is **not computed** in `step3d_add_service_coverage.py`. It
is computed in `step1_aggregate_busto.py` and merely renamed/merged in step3d.
Both are quoted:

`step1_aggregate_busto.py`, lines 46–54:
```python
stops = (
    raw.groupby("STOPCODE")
       .agg(
           stop_name=("STOPNAME", "first"),
           total_boardings=("Boardings", "sum"),
           n_route_dir_qhr_rows=("Boardings", "size"),  # how many rows contributed
       )
       .reset_index()
)
```
`raw` = `pd.concat()` of the 4 files matched by `glob.glob("data/*Weekday*QUARTER HOUR*.csv")`
(line 23), each row being one (ROUTE, DIRECTION, QHr, STOPCODE) combination.

`step3d_add_service_coverage.py`, lines 34–35, 38:
```python
sc = (busto[["STOPCODE", "n_route_dir_qhr_rows"]]
      .rename(columns={"n_route_dir_qhr_rows": "service_coverage"}))
...
merged["service_coverage"] = merged["service_coverage"].fillna(0).astype(int)
```

`.agg(n_route_dir_qhr_rows=("Boardings", "size"))` counts **rows**, not
boardings values — `"size"` counts group membership regardless of the value
in the aggregated column. Confirmed: it counts the number of distinct
(ROUTE, DIRECTION, QHr) rows per STOPCODE.

**D1.2 — raw BUSTO extract statistics**

Code:
```python
files = glob.glob("data/*Weekday*QUARTER HOUR*.csv")
frames = [pd.read_csv(f, dtype={"STOPCODE":"string"}) for f in files]
raw = pd.concat(frames, ignore_index=True)
b = raw["Boardings"]
```
Raw output:
```
Files found: 4 (Letter Prefix, 1-149, 150-299, 300-549)
Total rows loaded: 3,441,745
min boardings value: 0.0
1st percentile boardings: 0.0
rows with boardings==0: 584,829 / 3,441,745 = 16.9922%
Boardings dtype: float64, NaN count: 0
```
Schema (`sample.columns`): `YEAR, DAY_TYPE, TIMEBAND, QHr, ROUTE, DIRECTION,
STOPCODE, STOPNAME, STOPSEQUENCE, Boardings, Alightings, Load, Capacity,
Seats, V/C`.

`Capacity`/`Seats`/`STOPSEQUENCE` independence check, boardings==0 rows
(n=584,829) vs. boardings>0 rows (n=2,856,916):
```
                    boardings==0                  boardings>0
Capacity   NaN=0, min=2.0, mean=60.37    NaN=0, min=2.0, mean=98.55
Seats      NaN=0, min=1.0, mean=38.60    NaN=0, min=1.0, mean=64.84
STOPSEQUENCE NaN=0, min=1.0, mean=30.25  NaN=0, min=1.0, mean=21.24
```
`Capacity` and `Seats` are fully populated (zero NaNs) with sensible vehicle
values even when `Boardings==0` — e.g. a sampled zero-boarding row: `ROUTE=A10,
STOPCODE=BP1353, Boardings=0.0, Capacity=55.0, Seats=28.0, Load=1.136`. These
columns describe the scheduled vehicle/trip, populated independently of
whether anyone boarded.

**D1.3 — classification**

**Answer: (a) — BUSTO includes zero-boarding rows (16.99% of all rows, min
and 1st-percentile Boardings both exactly 0.0), and `Capacity`/`Seats`/
`STOPSEQUENCE` are fully populated on those same zero-boarding rows,
confirming they describe a scheduled service run independently of observed
demand → `service_coverage` is a schedule census, not demand-censored.**

---

## D2 — Dependence of `service_coverage` on the target

Code:
```python
df = pd.read_csv("stops_features_osm.csv")   # n = 17,943
sc, boardings = df["service_coverage"].values, df["total_boardings"].values
log_boardings = np.log1p(boardings)
pearsonr(sc, boardings); spearmanr(sc, boardings)
pearsonr(sc, log_boardings); spearmanr(sc, log_boardings)
Ridge(alpha=1.0).fit(np.log1p(sc).reshape(-1,1), log_boardings)   # descriptive, full sample, no CV
```
Raw output:
```
n stops: 17943
Pearson(SC, boardings)          = 0.6016  p=0
Spearman(SC, boardings)         = 0.7235  p=0
Pearson(SC, log1p(boardings))   = 0.6038  p=0
Spearman(SC, log1p(boardings))  = 0.7235  p=0
Univariate Ridge (log1p(SC) -> log1p(boardings)), full-sample R2 = 0.4736
Univariate Ridge (raw SC -> raw boardings), full-sample R2 = 0.3620
```

**Answer: Pearson r=0.6016 (raw) / 0.6038 (log1p); Spearman ρ=0.7235 (both
scales); univariate log1p-log1p Ridge R²=0.4736 (raw-scale R²=0.3620) — SC
alone explains ~47% of the variance in log1p(boardings) with no other
features, full-sample, descriptive (not cross-validated).**

---

## D3 — WMAPE aggregation method

Code (`step4_model.py`, lines 148–149, called once per fold inside `score()`):
```python
def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))
```
Code (`step4_model.py`, lines 552–558, `main()`, run once after all 33 folds):
```python
agg_mean = (results.groupby("model")[["WMAPE","RMSE","MAE"]]
            .agg(["mean","std","median"]).round(4))
agg_mean.columns = [f"{m}_{s}" for m, s in agg_mean.columns]
agg = agg_mean.reset_index()
```
`results` is a DataFrame with one row per (borough, model) pair, where each
row's `WMAPE` value was already computed by calling `wmape()` on that single
borough's `y_true`/`y_pred` only (inside `run_fold` → `score()`). The
reported `WMAPE_mean` in every `results_summary_*.csv` is the arithmetic mean
of these 33 pre-computed per-borough values — `groupby("model")["WMAPE"].agg("mean")`
never re-touches raw predictions or residuals; it averages already-aggregated
numbers.

**Note on inference:** the task message was cut off after "... or one
poole[d]". This is answered as the standard macro-vs-micro-WMAPE distinction
implied by the sentence structure (macro already named in the message; the
alternative is a single pooled/micro WMAPE computed once over all 17,943
stops' concatenated errors). If a different second option was intended,
disregard the phrasing below and re-ask.

**Answer: Macro-average. The headline WMAPE is the unweighted arithmetic mean
of 33 independently-computed per-borough WMAPEs — NOT a single pooled
(micro-average) WMAPE computed over all 17,943 stops' concatenated
errors/targets. Consequence: every borough contributes equally to the
headline number regardless of stop count (e.g. City of London, n=101, and
Bromley, n=1,179, are weighted identically), which is a real, checkable
methodological detail worth stating explicitly in the methods chapter.**

---

## D4 — Fold sizes

Code:
```python
df = pd.read_csv("stops_features_osm.csv")
counts = df.groupby("lad_name").size().sort_values()
```
Raw output — all 33 boroughs, ascending:
```
City of London                   101 <-- FLAG (<200)
Kensington and Chelsea           260
Hammersmith and Fulham           276
Islington                        349
Barking and Dagenham             364
Tower Hamlets                    381
Kingston upon Thames             391
Westminster                      419
Sutton                           421
Camden                           424
Harrow                           432
Hackney                          442
Merton                           457
Haringey                         461
Newham                           465
Wandsworth                       481
Richmond upon Thames             484
Waltham Forest                   492
Redbridge                        539
Brent                            575
Lambeth                          577
Southwark                        583
Lewisham                         609
Hounslow                         651
Greenwich                        659
Bexley                           672
Ealing                           683
Havering                         691
Hillingdon                       706
Enfield                          707
Barnet                           993
Croydon                         1019
Bromley                         1179
```
Total boroughs: 33. Total stops: 17,943. Only one borough falls below the
200-stop flag threshold.

3 smallest folds, fold size next to WMAPE (MLP/RF/GATv2), from
`results_cv_ai23_osm_sc.csv`:
```
model                   n_test   GATv2     MLP      RF
borough
City of London             101  0.7855  0.5702  0.5729
Kensington and Chelsea     260  0.6059  0.5803  0.5484
Hammersmith and Fulham     276  0.6958  0.6872  0.6730
```

**Answer: 1 of 33 boroughs falls below 200 stops — City of London, n=101 (the
smallest fold by a wide margin; the next-smallest, Kensington and Chelsea, is
260). City of London explicit: 101 stops. For the 3 smallest folds (101, 260,
276), GATv2's WMAPE (0.7855, 0.6059, 0.6958) is higher than both RF and MLP
in all three; City of London specifically shows the largest GATv2-vs-tabular
gap of the three (GATv2=0.7855 vs RF=0.5729, a 0.2126 gap).**

---

## D5 — LSOA / AI23 leakage figures (verification)

**D5.1 — LSOA sharing and stops-per-LSOA distribution**

Code:
```python
lsoa_counts = df.groupby("lsoa11cd").size()
n_shared = (df["lsoa11cd"].map(lsoa_counts) > 1).sum()
```
Raw output:
```
stops sharing LSOA with >=1 other stop: 17,290 / 17,943 = 96.36%
stops-per-LSOA distribution: median=3.0, max=62, mean=4.250, n_LSOAs=4222
```

**Answer: 96.36% of stops (17,290/17,943) share their LSOA with at least one
other stop. Stops-per-LSOA: median=3, max=62, mean=4.25, across 4,222 unique
LSOAs.**

**D5.2 — recompute two specific prose claims**

(i) Stop 1000 → LSOA E01004733, 3 of 5 KNN neighbours in that LSOA. Recomputed
independently via manual haversine (not a re-run of the project's own
`BallTree`-based `build_knn_edge_index` — a separate numpy implementation, to
cross-check rather than repeat):
```python
dlat = lat - lat[i0]; dlon = lon - lon[i0]
a = np.sin(dlat/2)**2 + np.cos(lat[i0])*np.cos(lat)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
dist_km = 6371.0088 * c
```
Raw output:
```
STOPCODE  stop_name                              lsoa11cd    dist_km
1645      WESTMINSTER STN / PARLIAMENT SQUARE     E01004731   0.029188
996       WESTMINSTER STN / PARLIAMENT SQUARE     E01004731   0.040300
BP3404    WESTMINSTER STN / WESTMINSTER PIER      E01004733   0.081906
BP5568    WESTMINSTER STN / PARLIAMENT SQUARE     E01004733   0.098007
BP4490    PARLIAMENT SQUARE / WESTMINSTER ABBEY   E01004733   0.171341
stop 1000 LSOA = E01004733; of 5 nearest neighbours, 3 share this LSOA
```

(ii) "LSOAs nest within boroughs in 4,219/4,222 cases (99.93%)":
```python
lsoa_to_boroughs = df.groupby("lsoa11cd")["lad_name"].nunique()
```
Raw output:
```
Total unique LSOAs: 4222
LSOAs mapping to exactly 1 borough: 4219
LSOAs mapping to >1 borough: 3
Percentage single-borough: 99.9289%
```

**Answer: (i) MATCH — independently recomputed, 3 of 5 nearest neighbours
(identical STOPCODEs: 1645, 996, BP3404, BP5568, BP4490) share stop 1000's
LSOA, exact match to the figure already in the draft prose. (ii) MATCH —
4,219/4,222 = 99.9289%, rounds to the "99.93%" already quoted; 3 LSOAs cross
a borough boundary in this dataset.**

---

## D6 — Assortativity of the target across graph edges

Run exactly as supplied. Note: this diagnostic builds its own edge set
independently for this check (one-directional K=5 KNN via `BallTree`, plus
route-consecutive pairs taken directly from each (ROUTE, DIRECTION) group's
observed STOPSEQUENCE order) — it is not a re-use of, and is not guaranteed
edge-for-edge identical to, `step4_model.py`'s own `build_knn_edge_index`/
`build_route_edges` (which explicitly symmetrise both edge types and dedup
via `torch.unique`). Stated as a factual construction difference, not a
correctness claim about either.

Code: as supplied (KNN via `BallTree(coords, metric="haversine").query(k=6)`,
route edges via consecutive `STOPCODE` pairs per `(ROUTE, DIRECTION)` group
sorted by `STOPSEQUENCE`, Pearson correlation of `log1p(total_boardings)`
across each edge subset).

Raw output:
```
TODAS las aristas                    n= 128685  r=0.5035
dentro de borough                    n= 123314  r=0.5088
CRUZAN borough (condicion de test)   n=   5371  r=0.3354

DAY_TYPE: ['Weekday']  (length 1 — single value)
filas de rutas escolares 6xx: 0
total filas raw: 3,441,745
n edges total (src): 128,685   n route-edge pairs (re_): 38,970
```
Follow-up check on the 6xx result (dtype/formatting artifact test): `ROUTE`
dtype is `object` (string); 602 unique route codes exist, including
single/double-digit codes starting with 6 (`"6"`, `"60"`...`"69"`); max route
string length is 4 characters; an anchored regex (`^6\d\d$`) also returns 0.
Not a dtype mismatch — no 3-digit "6XX"-coded route exists in this raw
extract.

**Answer: target assortativity (Pearson r of log1p(boardings) between
connected nodes) is 0.5088 within-borough vs. 0.3354 cross-borough (n=123,314
vs. n=5,371 edges) — a drop of 0.1734 (34.1% relative). Cross-borough edges,
which are the ONLY edges available to a held-out borough's stops under
leave-borough-out CV, connect nodes whose target values are measurably less
correlated than within-borough edges. DAY_TYPE has exactly one value
("Weekday") in the raw extract — no Saturday/Sunday contamination. Zero rows
match a 3-digit "6XX" route code pattern (verified not a dtype artifact) —
no school-route-coded services present in this extract under that pattern.**
