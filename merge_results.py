"""
MERGE RESULTS — consolidate per-config result CSVs into two files
====================================================================
Each step4_model.py run writes one results_cv_<config>.csv +
results_summary_<config>.csv pair, which adds up fast. This script
concatenates them all into all_results_cv.csv / all_results_summary.csv
(adding 'config', 'seeds', 'n_seeds' columns) and leaves the per-config
files untouched.

'seeds'/'n_seeds' are static metadata, not recomputed from the source
files -- every config here is a single fixed-seed run (SEED=42), so
n_seeds=1 everywhere except idw, marked "deterministic" (not "n/a", which
pandas would silently read back as NaN). Configs that use multiple seeds
(gated mixing, multi-seed MLP) are deliberately excluded below -- their
per-seed schema doesn't fit a single seeds/n_seeds row. Update a config's
tuple here if it's ever re-run with multiple seeds.

step6_consolidated_table.py and step7_borough_extract.py read from the
merged files, not the per-config ones -- re-run this script after any new
results_cv_<config>.csv, before running those.

Only merges the standard per-fold/summary schema. Excludes
results_cv_colab.csv (wide format), results_cv_multiseed_mlp.csv
(seed-indexed), and the results_cv_gated_seed*/all_seeds.csv group (own
per-seed structure, consumed directly by step4g_gated_analysis.py) -- these
stay standalone.
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
