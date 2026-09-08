"""
STEP 8C -- Target distribution before and after log1p
=============================================================================
Chapter 3 (Dataset description) is the only substantive chapter with no
figure -- the skewness claim in Section `target` (5.64 raw, -0.55 after
log1p) is stated as numbers only. This is the visual for it: two histograms
of total_boardings, raw and log1p-transformed, side by side.

Run with: python step8c_target_distribution.py
Output: fig8_target_distribution.png (300 dpi)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

BLUE      = "#2a78d6"
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

df = pd.read_csv("stops_features_osm.csv")
y = df["total_boardings"].values.astype(float)
y_log = np.log1p(y)

skew_raw = skew(y)
skew_log = skew(y_log)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

axes[0].hist(y, bins=80, color=BLUE, edgecolor="white", linewidth=0.3, zorder=3)
axes[0].set_title("A. total_boardings (raw)", fontsize=11, loc="left")
axes[0].set_xlabel("Average weekday boardings")
axes[0].set_ylabel("Number of stops")
axes[0].text(0.97, 0.94, f"skewness = {skew_raw:.2f}", transform=axes[0].transAxes,
             fontsize=9.5, color=INK_SEC, ha="right", va="top")

axes[1].hist(y_log, bins=80, color=BLUE, edgecolor="white", linewidth=0.3, zorder=3)
axes[1].set_title("B. log1p(total_boardings)", fontsize=11, loc="left")
axes[1].set_xlabel(r"$\log(1+y)$")
axes[1].text(0.03, 0.94, f"skewness = {skew_log:.2f}", transform=axes[1].transAxes,
             fontsize=9.5, color=INK_SEC, ha="left", va="top")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)

fig.suptitle(f"Target distribution: n={len(y):,} stops, log1p reduces skewness "
             f"from {skew_raw:.2f} to {skew_log:.2f}", fontsize=11.5, y=1.02)

fig.tight_layout()
fig.savefig("fig8_target_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved -> fig8_target_distribution.png")
print(f"  raw skew    = {skew_raw:.4f}")
print(f"  log1p skew  = {skew_log:.4f}")
