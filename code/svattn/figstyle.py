"""Shared figure style for the SV-Attention paper, matched to the author's
prior work (dissertation figures): periwinkle-blue / salmon-red palette, bold
sans-serif labels, ALL-CAPS bold subplot titles, blue--white--red diverging
heatmaps, black pipeline arrows and a black model box.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Palette lifted from the dissertation's Study-1/Study-2 bar charts.
BLUE = "#5b6bd6"     # periwinkle (margin / SV gate / "ours")
RED = "#ee7b7b"      # salmon (error set / decay / H2O)
GRAY = "#9aa0a6"     # reserve / neutral
DARKGRAY = "#54585d"
GREEN = "#5fae7e"    # full-context reference
INK = "#1a1a1a"      # boxes, arrows, text

# Blue--white--red diverging map (covariance-heatmap look).
BWR = LinearSegmentedColormap.from_list("bwr_diss", [BLUE, "#ffffff", RED])


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 160,
    })


def caps_title(ax, text):
    ax.set_title(text.upper(), fontweight="bold")
