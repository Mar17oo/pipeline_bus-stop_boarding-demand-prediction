"""
STEP 8B -- Figure 6.3: effect of adding service_coverage (Ridge + RF only)
=============================================================================
Slope chart, AI23+OSM -> AI23+OSM+SC, for Ridge and RF only. MLP is
deliberately excluded: its AI23+OSM value (0.7983) is v1-era (pre skip-
connection/leakage-fix, see RUNBOOK.md Sec.5) while its AI23+OSM+SC value
(0.6311) is current-protocol -- plotting them together would silently mix
two different code versions inside one line. Ridge and RF are both
current-protocol on both ends (RF/MLR/XGBoost were never affected by the
GATv2-specific bugs), so the slope is a clean, single-version comparison.

Run with: python step8b_fig6_service_coverage.py
Output: fig6_service_coverage.png (300 dpi)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Same palette as step8_dissertation_figures.py (dataviz skill reference
# palette, light mode, categorical slots) -- BLUE/ORANGE are already loaded
# with "tabular vs graph" meaning in fig1/fig2, so this figure (two tabular
# models only) uses BLUE + AQUA instead to avoid implying either model is
# a graph model.
BLUE      = "#2a78d6"
AQUA      = "#1baf7a"
INK       = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SEC,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "axes.grid": False,
})

table = pd.read_csv("consolidated_results_table.csv", index_col=0)
# Ridge labelled ABOVE its line, RF labelled BELOW -- the two series are close
# enough (0.7982 vs 0.7970, 0.6404 vs 0.6428) that same-side labels collide;
# this was caught by actually rendering and looking, not assumed away.
series = {
    "Ridge":         (table.loc["MLR", "AI23+OSM"], table.loc["MLR", "AI23+OSM+SC"], BLUE,  1),
    "Random Forest": (table.loc["RF",  "AI23+OSM"], table.loc["RF",  "AI23+OSM+SC"], AQUA, -1),
}

fig, ax = plt.subplots(figsize=(6.5, 6))
x = [0, 1]

for name, (v0, v1, color, side) in series.items():
    ax.plot(x, [v0, v1], color=color, linewidth=2.4, marker="o", markersize=8,
             markeredgecolor="white", markeredgewidth=1.2, zorder=3, label=name)
    delta = v1 - v0
    dy = 11 * side
    va = "bottom" if side > 0 else "top"
    ax.annotate(f"{v0:.4f}", xy=(0, v0), xytext=(-8, dy), textcoords="offset points",
                ha="right", va=va, fontsize=9.5, color=color)
    ax.annotate(f"{v1:.4f}", xy=(1, v1), xytext=(8, dy), textcoords="offset points",
                ha="left", va=va, fontsize=9.5, color=color)
    mid_y = (v0 + v1) / 2
    ax.annotate(f"{name}: {delta:+.4f} ({delta/v0*100:+.1f}%)", xy=(0.5, mid_y),
                xytext=(0, 14 * side), textcoords="offset points",
                ha="center", va=va, fontsize=9, color=color, fontweight="bold")

ax.set_xlim(-0.35, 1.35)
ax.set_xticks(x)
ax.set_xticklabels(["AI23+OSM", "AI23+OSM+SC"], fontsize=11.5)
ax.set_ylabel("WMAPE (lower = better)")
ax.set_title("Effect of adding service coverage\n(Ridge and RF only -- see note on MLP exclusion)",
              fontsize=11.5, pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.5))
ax.margins(y=0.14)
ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)

fig.tight_layout()
fig.savefig("fig6_service_coverage.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved -> fig6_service_coverage.png")
for name, (v0, v1, _, _) in series.items():
    print(f"  {name}: {v0:.4f} -> {v1:.4f}  (delta {v1-v0:+.4f}, {(v1-v0)/v0*100:+.1f}%)")
