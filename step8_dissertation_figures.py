"""
STEP 8 — Dissertation figures F1-F5
=====================================
Replots existing result CSVs into five dissertation-ready figures. No new
experiments are run; every number here already exists in
all_results_cv.csv / all_results_summary.csv / consolidated_results_table.csv
/ boundary_diagnostic_stops.csv / stops_features.csv.

  F1  WMAPE by model, AI23+OSM+SC (headline) — ranked horizontal bars
  F2  Grouped bars across 4 feature sets, all 14 models, v1-era cells hatched
  F3  Per-borough paired distribution, MLP vs GATv2 (33 folds)
  F4  Difference choropleth (GATv2 - MLP) per borough
  F5  Fold-structure schematic + real KNN-crosses-boundary example (Islington)

F4 and F5 download London borough boundaries from the ONS Open Geography
API (same endpoint as step5a_borough_map.py) — needs internet.

Output: fig1_wmape_by_model.png ... fig5_fold_structure.png (300 dpi)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.stats import wilcoxon
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------------------
# Palette (dataviz skill reference palette — light mode, categorical slots)
# ---------------------------------------------------------------------------
BLUE      = "#2a78d6"   # categorical slot 1 — tabular / MLP / training
ORANGE    = "#eb6834"   # categorical slot 2 — graph / GATv2
AQUA      = "#1baf7a"   # categorical slot 3
YELLOW    = "#eda100"   # categorical slot 4
RED       = "#e34948"   # diverging pole (worse / held-out fold)
INK       = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
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


# ===========================================================================
# F1 — WMAPE by model, AI23+OSM+SC (headline), ranked horizontal bars
# ===========================================================================
def fig1_wmape_by_model():
    table = pd.read_csv("consolidated_results_table.csv", index_col=0)
    col = "AI23+OSM+SC"

    models = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "RF-tuned",
              "GATv2", "GCN", "GATv2-Fusion (G1/G2/G3)"]
    tabular = {"HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "RF-tuned"}
    labels = {"GATv2-Fusion (G1/G2/G3)": "GATv2-Fusion"}

    vals = table.loc[models, col].astype(float)
    order = vals.sort_values(ascending=True).index.tolist()   # best (lowest) first
    order = order[::-1]                                       # best at TOP of barh

    y = np.arange(len(order))
    colors = [BLUE if m in tabular else ORANGE for m in order]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.barh(y, [vals[m] for m in order], color=colors,
                    edgecolor="white", linewidth=0.6, height=0.68, zorder=3)

    for yi, m in zip(y, order):
        v = vals[m]
        ax.text(v + 0.012, yi, f"{v:.4f}", va="center", ha="left",
                fontsize=8.5, color=INK_SEC)

    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(m, m) for m in order], fontsize=9.5)
    ax.set_xlabel("WMAPE (lower = better)")
    ax.set_title("Model comparison — AI23 + OSM + SC (headline feature set)\n"
                  "33-fold leave-borough-out spatial CV", fontsize=11.5, pad=12)

    hist_val = table.loc["HistAvg", col]
    ax.axvline(hist_val, color=INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(hist_val, len(order) - 0.3, f" HistAvg reference ({hist_val:.4f})",
            fontsize=8, color=INK_MUTED, va="bottom", ha="left", style="italic")

    # MLP / RF-tuned tie annotation
    mlp_y = order.index("MLP")
    rft_y = order.index("RF-tuned")
    lo, hi = sorted([mlp_y, rft_y])
    bx = max(vals["MLP"], vals["RF-tuned"]) + 0.19
    ax.plot([bx, bx], [lo, hi], color=INK_SEC, linewidth=0.9, zorder=4)
    ax.plot([bx - 0.012, bx], [lo, lo], color=INK_SEC, linewidth=0.9, zorder=4)
    ax.plot([bx - 0.012, bx], [hi, hi], color=INK_SEC, linewidth=0.9, zorder=4)
    ax.text(bx + 0.02, (lo + hi) / 2, "statistically tied\n(p = 0.292)",
            fontsize=8, color=INK_SEC, va="center", ha="left")

    ax.set_xlim(0, hist_val + 0.32)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    legend_handles = [mpatches.Patch(color=BLUE, label="Tabular"),
                       mpatches.Patch(color=ORANGE, label="Graph")]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig("fig1_wmape_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> fig1_wmape_by_model.png")


# ===========================================================================
# F2 — Grouped bars across 4 feature sets, all 14 models
# ===========================================================================
def fig2_grouped_feature_sets():
    table = pd.read_csv("consolidated_results_table.csv", index_col=0)
    cols = ["AI23-only", "OSM-only", "AI23+OSM", "AI23+OSM+SC"]
    rows = ["HistAvg", "IDW", "MLR", "RF", "XGBoost", "MLP", "GATv2", "GCN",
            "GATv2-fairness (K=10)", "GATv2-func-sim", "GATv2-Fusion (G1/G2/G3)",
            "MLR-tuned", "RF-tuned", "XGBoost-tuned"]
    short = {"GATv2-fairness (K=10)": "GATv2\n(K=10)",
              "GATv2-func-sim": "GATv2\n(func-sim)",
              "GATv2-Fusion (G1/G2/G3)": "GATv2\n-Fusion"}

    colors = {"AI23-only": BLUE, "OSM-only": ORANGE,
              "AI23+OSM": AQUA, "AI23+OSM+SC": YELLOW}
    V1_ERA_MODELS = {"MLP", "GATv2"}
    V1_ERA_COLS = {"AI23-only", "OSM-only", "AI23+OSM"}

    n_rows, n_cols = len(rows), len(cols)
    width = 0.8 / n_cols
    x = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(17, 6.5))
    for j, c in enumerate(cols):
        xs = x + (j - (n_cols - 1) / 2) * width
        for i, m in enumerate(rows):
            v = table.loc[m, c]
            if pd.isna(v):
                continue
            hatched = (m in V1_ERA_MODELS) and (c in V1_ERA_COLS)
            ax.bar(xs[i], v, width=width * 0.92, color=colors[c],
                   edgecolor="white", linewidth=0.5,
                   hatch="///" if hatched else None, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short.get(m, m) for m in rows], fontsize=8.5)
    ax.set_ylabel("WMAPE (lower = better)")
    ax.set_title("WMAPE by model across feature sets — grid cells with no run are left empty",
                  fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    feature_handles = [mpatches.Patch(color=colors[c], label=c) for c in cols]
    hatch_handle = mpatches.Patch(facecolor="white", edgecolor=INK_SEC,
                                   hatch="///", label="v1-era MLP/GATv2 (pre-fix)ᵉ")
    ax.legend(handles=feature_handles + [hatch_handle], loc="upper right",
              frameon=False, fontsize=8.5, ncol=1)

    fig.text(0.01, -0.02,
              "(e) MLP/GATv2 cells for AI23-only, OSM-only and AI23+OSM predate the "
              "validation-loss fix (Bug 3); HistAvg/MLR/RF/XGBoost in those same columns "
              "were re-run under the current protocol. AI23+OSM+SC (headline) and all "
              "GNN-variant/tuned rows are current-protocol throughout.",
              fontsize=8, color=INK_MUTED, ha="left", va="top", wrap=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("fig2_grouped_feature_sets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> fig2_grouped_feature_sets.png")


# ===========================================================================
# F3 — Per-borough paired distribution, MLP vs GATv2
# ===========================================================================
def fig3_paired_borough_boxplot():
    df = pd.read_csv("all_results_cv.csv")
    sub = df[df["config"] == "ai23_osm_sc"]
    piv = sub.pivot(index="borough", columns="model", values="WMAPE")
    mlp, gatv2 = piv["MLP"].values, piv["GATv2"].values

    stat, p = wilcoxon(mlp, gatv2)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for m, g in zip(mlp, gatv2):
        ax.plot([1, 2], [m, g], color=INK_MUTED, alpha=0.35, linewidth=0.9, zorder=2)
    ax.scatter(np.full(len(mlp), 1), mlp, color=BLUE, s=18, alpha=0.7, zorder=3)
    ax.scatter(np.full(len(gatv2), 2), gatv2, color=ORANGE, s=18, alpha=0.7, zorder=3)

    bp = ax.boxplot([mlp, gatv2], positions=[1, 2], widths=0.32, patch_artist=True,
                     showmeans=True, zorder=4,
                     medianprops=dict(color=INK, linewidth=1.4),
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor=INK, markersize=6),
                     whiskerprops=dict(color=INK_SEC), capprops=dict(color=INK_SEC),
                     flierprops=dict(markeredgecolor=INK_SEC, markersize=4))
    for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MLP", "GATv2"], fontsize=12)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel("WMAPE (lower = better)")
    ax.set_title("Per-borough WMAPE, MLP vs GATv2 (n = 33 boroughs)\n"
                  "AI23 + OSM + SC, leave-borough-out CV", fontsize=11.5, pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(0.5, 0.02,
            f"Wilcoxon signed-rank (paired by borough): p = {p:.2e}\n"
            f"GATv2 worse in {int((gatv2 > mlp).sum())}/33 boroughs "
            f"(mean gap {np.mean(gatv2 - mlp):+.4f})",
            transform=ax.transAxes, fontsize=8.5, color=INK_SEC,
            ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig("fig3_paired_borough_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> fig3_paired_borough_boxplot.png  (Wilcoxon p={p:.3e})")


# ---------------------------------------------------------------------------
# Shared: fetch London borough boundaries (ONS Open Geography, same as step5a)
# ---------------------------------------------------------------------------
def fetch_borough_boundaries():
    import geopandas as gpd
    ONS_URL = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Local_Authority_Districts_May_2023_UK_BFC_V2/FeatureServer/0/query"
        "?where=LAD23CD+LIKE+'E09%25'"
        "&outFields=LAD23NM&returnGeometry=true"
        "&geometryPrecision=5&outSR=4326&f=geojson"
    )
    print("Downloading London borough boundaries from ONS...")
    gdf = gpd.read_file(ONS_URL)
    gdf = gdf.rename(columns={"LAD23NM": "borough"})
    gdf["borough"] = gdf["borough"].str.strip()
    print(f"  Downloaded {len(gdf)} borough boundaries")
    return gdf


# ===========================================================================
# F4 — Difference choropleth (GATv2 - MLP) per borough
# ===========================================================================
def fig4_difference_map(gdf):
    df = pd.read_csv("all_results_cv.csv")
    sub = df[df["config"] == "ai23_osm_sc"]
    piv = sub.pivot(index="borough", columns="model", values="WMAPE").reset_index()
    piv["diff"] = piv["GATv2"] - piv["MLP"]
    piv["borough"] = piv["borough"].str.strip()

    merged = gdf.merge(piv[["borough", "diff", "GATv2", "MLP"]], on="borough", how="inner")
    if len(merged) < 30:
        print(f"  WARNING — only {len(merged)}/33 boroughs matched")

    vmax = merged["diff"].abs().max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", RED])

    fig, ax = plt.subplots(figsize=(9, 9))
    merged.plot(column="diff", ax=ax, cmap=cmap, norm=norm,
                edgecolor="white", linewidth=0.5)

    # Hammersmith and Fulham / Kensington and Chelsea are small, adjacent
    # boroughs whose centroids sit close enough that both labels land on
    # top of each other at this map scale. Keyed by borough name (not row
    # index or centroid coords) so the nudge survives a re-sort of the
    # underlying CSV. Small, deterministic offset + hairline leader line —
    # not adjustText — so the figure renders identically on every regen.
    LABEL_OFFSETS = {
        "Hammersmith and Fulham": (-24, -20),   # down-left, into H&F's own area
        "Kensington and Chelsea": (10, -26),     # down, toward the river — Westminster
                                                  # sits NE of Kensington, so an up-right
                                                  # nudge collides with it; south is clear
    }

    for _, row in merged.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        if row["diff"] >= merged["diff"].quantile(0.85) or row["diff"] <= merged["diff"].quantile(0.15):
            label = f"{row['borough'].split()[0]}\n{row['diff']:+.3f}"
            color = "white" if row["diff"] > vmax * 0.3 else INK
            bbox = dict(boxstyle="round,pad=0.12", fc="black", alpha=0.0, ec="none")
            offset = LABEL_OFFSETS.get(row["borough"])
            if offset:
                ax.annotate(label, xy=(cx, cy), xytext=offset, textcoords="offset points",
                            ha="center", va="center", fontsize=6, color=color, bbox=bbox,
                            arrowprops=dict(arrowstyle="-", color=INK_SEC, lw=0.5,
                                             shrinkA=0, shrinkB=3))
            else:
                ax.annotate(label, xy=(cx, cy), ha="center", va="center",
                            fontsize=6, color=color, bbox=bbox)

    ax.set_title("Where GATv2 costs the most relative to MLP\n"
                  "(GATv2 WMAPE − MLP WMAPE) per borough, AI23+OSM+SC",
                  fontsize=12, pad=12)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, shrink=0.75)
    cbar.set_label("GATv2 − MLP  (WMAPE, positive = GATv2 worse)", fontsize=9.5)

    # "All 33 / range" summary now lives in the LaTeX caption (3dp, matching
    # the per-borough labels) — the burned-in text duplicated it at 4dp.
    # ax.text(0.02, 0.02,
    #         f"GATv2 is worse than MLP in all {len(merged)}/{len(merged)} boroughs "
    #         f"(range {merged['diff'].min():+.4f} to {merged['diff'].max():+.4f}) —\n"
    #         "no borough crosses zero, so only the red half of the scale is populated.",
    #         transform=ax.transAxes, fontsize=8, color=INK_SEC, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig("fig4_difference_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> fig4_difference_map.png")


# ===========================================================================
# F5 — Fold structure diagram: held-out borough + real cross-boundary KNN example
# ===========================================================================
def fig5_fold_structure(gdf):
    HELD_OUT = "Islington"

    stops = pd.read_csv("stops_features.csv")
    target = stops[stops["STOPCODE"].astype(str) == "8562"].iloc[0]   # SNOW HILL, Islington

    coords = np.radians(stops[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    tq = np.radians([[target["lat"], target["lon"]]])
    dist, idx = tree.query(tq, k=6)                    # self + 5 neighbours
    dist_km = dist[0] * 6371.0088
    nbrs = stops.iloc[idx[0][1:]].copy()
    nbrs["dist_km"] = dist_km[1:]

    # Print-sized figure: dimension in the units it will actually appear at
    # (6.3in ~ a dissertation text width) rather than saving oversized and
    # letting LaTeX shrink it — shrinking is what makes labels unreadable.
    # No mechanism claim and no caption-length prose belong IN the image:
    # this stop illustrates the fold geometry only; the assortativity
    # result belongs in the results chapter, not baked into a Ch.3 figure.
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 8}):
        fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(6.3, 3.1))

        # ---- panel (a): all 33 boroughs, Islington highlighted as held-out ----
        is_held = gdf["borough"] == HELD_OUT
        gdf[~is_held].plot(ax=ax_main, color=GRID, edgecolor="white", linewidth=0.3)
        gdf[is_held].plot(ax=ax_main, color=RED, alpha=0.55, edgecolor=RED,
                           linewidth=0.7, hatch="///")

        ax_main.plot(target["lon"], target["lat"], marker="*", color=RED,
                     markersize=9, zorder=5, markeredgecolor=INK, markeredgewidth=0.5)
        ax_main.annotate(f"{target['stop_name'].title()} [{target['STOPCODE']}]",
                          xy=(target["lon"], target["lat"]),
                          xytext=(target["lon"] + 0.05, target["lat"] + 0.04),
                          fontsize=6.5, color=INK,
                          arrowprops=dict(arrowstyle="-", color=INK_SEC, lw=0.6))
        ax_main.set_title(f"(a) {HELD_OUT} held out (1 of 33 folds)")
        ax_main.axis("off")
        legend_handles = [mpatches.Patch(facecolor=RED, alpha=0.55, hatch="///",
                                          edgecolor=RED, label="Held-out (test) borough"),
                           mpatches.Patch(color=GRID, label="Training boroughs")]
        ax_main.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=6.5)

        # ---- panel (b): K=5 neighbourhood as a schematic radial network
        # (true GPS layout puts all 6 points within ~160m of each other —
        # illegible as a map at this scale; the network keeps the real
        # facts (borough, distance, stop code) in a layout built for
        # label clarity, not bearing accuracy) ----
        nbrs_sorted = nbrs.sort_values("dist_km").reset_index(drop=True)
        borough_colors = {HELD_OUT: RED, "Camden": BLUE, "City of London": AQUA}
        n = len(nbrs_sorted)
        angles = np.linspace(90, 90 - 360, n, endpoint=False)   # clockwise from top
        radius = 1.0
        # edges start just outside the star, not at the exact centre, so no
        # label placed near the star can be crossed by a radial line
        inner_r = 0.16

        ax_inset.set_xlim(-1.75, 1.75)
        ax_inset.set_ylim(-1.35, 2.3)   # bottom trimmed to content; top keeps room for the legend
        ax_inset.set_aspect("equal")

        for i, row in nbrs_sorted.iterrows():
            theta = np.radians(angles[i])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            nx, ny = radius * cos_t, radius * sin_t
            c = borough_colors.get(row["lad_name"], INK_MUTED)

            ax_inset.plot([inner_r * cos_t, nx], [inner_r * sin_t, ny], color=INK_SEC,
                          linewidth=0.9, alpha=0.75, zorder=2, linestyle=(0, (1, 1.5)))

            ax_inset.scatter(nx, ny, color=c, s=110, zorder=4, edgecolor="white", linewidth=0.8)
            # label outward along the same spoke, so it can never cross a
            # different spoke's line; alignment follows which side of the
            # centre the node falls on
            lx, ly = nx * 1.2, ny * 1.2
            ha = "center" if abs(cos_t) < 0.3 else ("left" if cos_t > 0 else "right")
            va = "bottom" if sin_t > 0.3 else ("top" if sin_t < -0.3 else "center")
            ax_inset.annotate(f"{row['stop_name'].title()} [{row['STOPCODE']}]\n"
                              f"{row['lad_name']} · {row['dist_km']*1000:.0f} m",
                              xy=(nx, ny), xytext=(lx, ly), fontsize=6.5, color=INK,
                              ha=ha, va=va)

        ax_inset.scatter(0, 0, color=RED, s=210, zorder=5, marker="*",
                         edgecolor=INK, linewidth=0.9)

        ax_inset.set_title(f"(b) K=5 neighbours of {target['stop_name'].title()} "
                            f"[{target['STOPCODE']}]")
        ax_inset.axis("off")

        inset_legend = [Line2D([0], [0], marker="*", color="none", markerfacecolor=RED,
                                markeredgecolor=INK, markersize=10,
                                label=f"{target['stop_name'].title()} [{target['STOPCODE']}] — test stop"),
                         Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                                markeredgecolor="white", markersize=7, label="Neighbour in Camden"),
                         Line2D([0], [0], marker="o", color="none", markerfacecolor=AQUA,
                                markeredgecolor="white", markersize=7, label="Neighbour in City of London")]
        ax_inset.legend(handles=inset_legend, loc="upper left", frameon=False, fontsize=6.5, ncol=1)

        fig.savefig("fig5_fold_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved -> fig5_fold_structure.pdf")


if __name__ == "__main__":
    fig1_wmape_by_model()
    fig2_grouped_feature_sets()
    fig3_paired_borough_boxplot()
    gdf = fetch_borough_boundaries()
    fig4_difference_map(gdf)
    fig5_fold_structure(gdf)
    print("\nAll 5 figures saved.")

