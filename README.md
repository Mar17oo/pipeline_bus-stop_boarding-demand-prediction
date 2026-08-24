# Cold-Start Bus Stop Demand Prediction under Spatial Extrapolation

Code, data pipeline, and results for a Master's dissertation predicting
weekday boarding demand at London bus stops that have no observed
boarding history — a cold-start problem — using open accessibility
indicators, OpenStreetMap points of interest, and planned service supply,
evaluated under 33-fold leave-borough-out spatial cross-validation.

**Headline result:** a plain 3-layer MLP (WMAPE = 0.6311) beats every graph
neural network variant tried, including GATv2 (0.7187) and GCN (0.7006).
The reason is diagnosed, not just observed: boundary-crossing graph edges
connect stops whose demand is only weakly correlated (see `RUNBOOK.md` §6).

This repository is **code, data pipeline, and results — not the manuscript
itself**, which is authored and maintained separately.

## Start here

**[`RUNBOOK.md`](RUNBOOK.md) is the canonical entry point** — full
reproduction instructions, the pipeline script table, the frozen results
table, and provenance notes for every number. Read that file, not this one,
for anything beyond a first orientation.

Two ways to check the results without necessarily reading all of RUNBOOK.md:

- **[`reproduce_tables_and_stats.ipynb`](reproduce_tables_and_stats.ipynb)**
  — regenerates every table/statistic from the CSVs already committed here.
  Runtime: well under a minute, no GPU, no raw data folder needed.
- **[`dissertation_colab.ipynb`](dissertation_colab.ipynb)** — re-runs the
  full 33-fold CV pipeline in Google Colab (needs the raw feature files
  uploaded to Drive first — see the notebook's own first cell).

## Data

Raw source data (~453MB) is not included in this repository (see
`RUNBOOK.md` §4.0 for exact download sources: TfL BUSTO, the AI23
accessibility dataset, ONS geography lookups, and OpenStreetMap via the
Overpass API). All three primary sources are openly licensed.

## Status

This project is frozen at commit `111d3b6` ("Freeze: code and results as
reported in dissertation"), with post-freeze verification and cleanup
tracked in subsequent commits. See `RUNBOOK.md` §8 for what every other
document in this repo is for, and whether it's still current.
