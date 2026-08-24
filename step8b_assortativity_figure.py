"""
STEP 8B -- Two-panel edge-assortativity scatter (within vs. across borough)
==============================================================================
Companion figure to the assortativity result already written up in
report.tex (Section~sec:boundary): log(1+y) at one edge endpoint plotted
against log(1+y) at the other, split by whether the edge crosses a borough
boundary, with Pearson r printed on each panel.

Reproduces D6's diagnostic graph from raw data (diagnostics_report.md, D6)
-- a one-directional, non-deduplicated K=5 KNN + route-consecutive graph,
built independently of (and not edge-for-edge identical to) step4_model.py's
own build_knn_edge_index/build_route_edges, which symmetrise and dedup. This
is a deliberate reproduction of D6's own construction, not the production
multigraph GATv2 trains on -- see D6's and this script's own notes.

D6 construction, one-directional / not deduplicated:
  - KNN: BallTree(haversine).query(k=6), edge i -> each of its 5 nearest
    neighbours (excluding self, no reverse edge added). Expect exactly
    17,943 * 5 = 89,715 edges.
  - Route: consecutive STOPCODE pairs per (ROUTE, DIRECTION) group, sorted
    by STOPSEQUENCE, one direction only (i -> i+1, no reverse). D6's raw
    output: 38,970 pairs, from the Weekday quarter-hour BUSTO CSVs.
  - Combined, no dedup: 89,715 + 38,970 = 128,685 edges -- matches D6's own
    raw output ("n edges total (src): 128,685") exactly, which is the
    reproduction check this script runs before trusting the plot.

Run with: python -u step8b_assortativity_figure.py
Output: fig7_assortativity_scatter.png (300 dpi)
"""
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import BallTree
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Palette (dataviz skill reference palette -- light mode, categorical slots 1/2,
# same as step8_dissertation_figures.py)
# ---------------------------------------------------------------------------
BLUE      = "#2a78d6"
ORANGE    = "#eb6834"
INK       = "#0b0b0b"
INK_SEC   = "#52514e"
AXIS      = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SEC,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "axes.grid": False,
})

DATA_FILE = "stops_features_osm.csv"
K = 5


def log(msg):
    print(msg, flush=True)


def build_knn_onedirectional(lats, lons, k=K):
    coords = np.radians(np.column_stack([lats, lons]))
    _, nbrs = BallTree(coords, metric="haversine").query(coords, k=k + 1)
    n = len(lats)
    src = np.repeat(np.arange(n), k)
    dst = nbrs[:, 1:].reshape(-1)
    return src, dst


def build_route_onedirectional(df_stops, data_folder="data"):
    files = glob.glob(f"{data_folder}/*Weekday*QUARTER HOUR*.csv")
    frames = [pd.read_csv(f, dtype={"STOPCODE": "string"},
                           usecols=["ROUTE", "DIRECTION", "STOPSEQUENCE", "STOPCODE"])
              for f in files]
    raw = pd.concat(frames, ignore_index=True)
    log(f"  raw BUSTO rows: {len(raw):,}  (D6 raw output: 3,441,745)")

    code_to_idx = {
        c: i for i, c in enumerate(
            df_stops["STOPCODE"].astype(str).str.strip().str.upper()
        )
    }
    raw["STOPCODE"] = raw["STOPCODE"].str.strip().str.upper()
    raw = raw[raw["STOPCODE"].isin(code_to_idx)].copy()
    raw["node_idx"] = raw["STOPCODE"].map(code_to_idx)

    src, dst = [], []
    for _, grp in raw.groupby(["ROUTE", "DIRECTION"]):
        idxs = (grp.drop_duplicates("STOPCODE")
                    .sort_values("STOPSEQUENCE")["node_idx"].tolist())
        for i in range(len(idxs) - 1):
            src.append(idxs[i]); dst.append(idxs[i + 1])
    return np.array(src), np.array(dst)


def main():
    df = pd.read_csv(DATA_FILE)
    n = len(df)
    lad = df["lad_name"].values
    logy = np.log1p(df["total_boardings"].values.astype(float))

    log(f"Loaded {n:,} stops")

    log("Building KNN edges (one-directional, K=5, not deduplicated)...")
    knn_src, knn_dst = build_knn_onedirectional(df["lat"].values, df["lon"].values)
    log(f"  KNN edges: {len(knn_src):,}  (expect {n * K:,})")

    log("Building route edges (one-directional, Weekday quarter-hour BUSTO)...")
    rt_src, rt_dst = build_route_onedirectional(df)
    log(f"  route-edge pairs: {len(rt_src):,}  (D6 raw output: 38,970)")

    src = np.concatenate([knn_src, rt_src])
    dst = np.concatenate([knn_dst, rt_dst])
    log(f"  combined, no dedup: {len(src):,}  (D6 raw output: 128,685)")

    within = lad[src] == lad[dst]
    log(f"  within: {within.sum():,}  cross: {(~within).sum():,}  "
        f"(D6 raw output: 123,314 / 5,371)")

    xi, yj = logy[src], logy[dst]
    r_within, _ = pearsonr(xi[within], yj[within])
    r_cross, _  = pearsonr(xi[~within], yj[~within])
    log(f"  r within = {r_within:.4f}  (D6: 0.5088)")
    log(f"  r cross  = {r_cross:.4f}  (D6: 0.3354)")

    cmap_blue   = LinearSegmentedColormap.from_list("blue_seq",   ["#ffffff", BLUE])
    cmap_orange = LinearSegmentedColormap.from_list("orange_seq", ["#ffffff", ORANGE])

    lims = (0, max(xi.max(), yj.max()) * 1.02)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)

    panels = [
        (axes[0], within,  cmap_blue,   "A. Within borough",          r_within, int(within.sum())),
        (axes[1], ~within, cmap_orange, "B. Across borough boundary", r_cross,  int((~within).sum())),
    ]
    for ax, mask, cmap, title, r, npts in panels:
        ax.plot(lims, lims, color=AXIS, lw=1, ls="--", zorder=1)
        ax.hexbin(xi[mask], yj[mask], gridsize=45, cmap=cmap, mincnt=1,
                   extent=(*lims, *lims), linewidths=0.1, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
        ax.set_xlabel(r"$\log(1+y_i)$")
        ax.text(0.06, 0.92, f"$r = {r:.2f}$\n$n = {npts:,}$",
                 transform=ax.transAxes, fontsize=10, color=INK,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=AXIS, linewidth=0.8, alpha=0.92))
        for spine in ax.spines.values():
            spine.set_edgecolor(AXIS)

    axes[0].set_ylabel(r"$\log(1+y_j)$")

    fig.tight_layout()
    fig.savefig("fig7_assortativity_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("\nSaved -> fig7_assortativity_scatter.png")


if __name__ == "__main__":
    main()
