"""Shared plotting style, so every figure in the report looks like part of one study.

Colours come from the Okabe-Ito palette, which is designed to stay distinguishable
under the common forms of colour-vision deficiency (Okabe & Ito, 2008). Series are
assigned colours in a fixed order, so an area keeps the same colour in every figure.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # no interactive window; figures are written straight to disk

FIGURES = Path("figures")

# Okabe-Ito, in fixed assignment order.
PALETTE = ["#0072B2",  # blue
           "#D55E00",  # vermillion
           "#009E73",  # bluish green
           "#CC79A7",  # reddish purple
           "#E69F00",  # orange
           "#56B4E9",  # sky blue
           "#F0E442"]  # yellow
INK = "#222222"
MUTED = "#6B6B6B"
GRID = "#D9D9D9"

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str, figures: Path = FIGURES) -> Path:
    """Write a figure to figures/<name>.png and return the path."""
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")
    return path
