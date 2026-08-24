"""
MERGE RESULTS — consolidate per-config result CSVs into two files
====================================================================
step4_model.py each write one
results_cv_<config>.csv + results_summary_<config>.csv pair per run, which
adds up to a lot of same-schema file pairs cluttering the repo root. This
script concatenates all of them into all_results_cv.csv /
all_results_summary.csv (added 'config', 'seeds', 'n_seeds' columns),
verifies no rows were lost, and leaves the per-config source files untouched
(delete them yourself once you've checked the output — see
PROJECT_STRUCTURE.md §4).

'seeds'/'n_seeds' are static metadata attached per config below, not
recomputed from the source files. Every config here is a single, fixed-seed
run: SEED=42 is hardcoded in step4_model.py and step4c/e_*.py, and every
other step4*/step5b script that contributes a config either imports SEED
from step4_model.py directly or reassigns its own copy of the same value
(verified by grep across step4*.py) -- so n_seeds=1 everywhere
except idw, which has no stochastic component at all (BallTree distance
weighting is deterministic) and is marked seeds="deterministic" (not "n/a" --
that string is one of pandas' default NA sentinels and would silently become
NaN on the next pd.read_csv). The experiments that
DO use multiple seeds (Gated mixing, 3 seeds; standalone MLP, 5 seeds -- see
experiment_log.md Experiment Sets 8 and 12) are exactly the ones this script
already keeps out of the merge (next paragraph), because their per-seed
schema doesn't fit a single seeds/n_seeds pair per config row. If a config
below is ever re-run with multiple seeds, update its tuple here to match --
this metadata is not derived automatically.

step6_consolidated_table.py and step7_borough_extract.py both read from
all_results_cv.csv / all_results_summary.csv (they take the old per-config
filename as an argument/constant purely to name which 'config' slice to
pull out — see their load_mean()/load_cv() functions). Re-run this script
after generating fresh results_cv_<config>.csv files, before running step6
or step7, or they will not find your new run's rows.

Only merges files sharing the standard per-fold schema
(borough,n_test,model,WMAPE,RMSE,MAE) or summary schema
(model,WMAPE_mean,...). Deliberately excludes: results_cv_colab.csv (wide
format, one column per model), results_cv_multiseed_mlp.csv (seed-indexed,
no model/RMSE/MAE columns), and the results_cv_gated_seed*/all_seeds.csv
group (consumed directly by step4g_gated_analysis.py with its own per-seed
structure) -- these stay as standalone files.
"""

import os
import pandas as pd

# (config, n_seeds, seeds) -- see docstring above for how n_seeds/seeds was
# determined and why it's manual, static metadata rather than derived.
CONFIGS = [
    ("ai23_only",               1, "42"),
    ("ai23_only_fastbaselines", 1, "42"),
    ("ai23_osm",                1, "42"),
    ("ai23_osm_fastbaselines",  1, "42"),
    ("ai23_osm_sc",             1, "42"),
    # ai23_sc (see RUNBOOK.md Sec.5, "AI23+SC" note) was added after this
    # CONFIGS list was first written, so it was silently missing from every
    # merge until now. Re-run this script to pick it up in
    # all_results_cv.csv / all_results_summary.csv.
    ("ai23_sc",                 1, "42"),
    ("func_sim",                1, "42"),
    ("gatv2_edge_attrs",        1, "42"),
    ("gcn",                     1, "42"),
    ("gnn_fairness",            1, "42"),
    ("idw",                     1, "deterministic"),   # no random seed (BallTree distance weighting)
    ("osm_only",                1, "42"),
    ("osm_only_fastbaselines",  1, "42"),
    ("osm_sc",                  1, "42"),
    ("pca_ai23_osm_sc",         1, "42"),
    ("tuned",                   1, "42"),
    ("vc_ai23_osm",             1, "42"),
    ("zheng_fusion",            1, "42"),
    # alt_target_activity has no full run yet -- include the quick one so its
    # data isn't lost if/when it's superseded by a full run, just re-add the
    # non-quick tag here once step4m has been run to completion.
    ("alt_target_activity_quick", 1, "42"),
]


def merge(prefix, out_name):
    frames, counts, skipped = [], {}, []
    for cfg, n_seeds, seeds in CONFIGS:
        fn = f"{prefix}_{cfg}.csv"
        if not os.path.exists(fn):
            skipped.append(fn)
            continue
        df = pd.read_csv(fn)
        df.insert(0, "config", cfg)
        df.insert(1, "seeds", seeds)
        df.insert(2, "n_seeds", n_seeds)
        frames.append(df)
        counts[cfg] = len(df)
    if skipped:
        print(f"{out_name}: skipping {len(skipped)} config(s) with no source file -- "
              f"not yet run, or the file wasn't kept: {skipped}")
    merged = pd.concat(frames, ignore_index=True)
    expected = sum(counts.values())
    assert len(merged) == expected, f"{out_name}: row count mismatch ({len(merged)} vs {expected})"
    merged.to_csv(out_name, index=False)
    print(f"{out_name}: {len(merged)} rows from {len(CONFIGS)} configs -- OK")
    return merged


def main():
    merge("results_cv", "all_results_cv.csv")
    merge("results_summary", "all_results_summary.csv")


if __name__ == "__main__":
    main()
