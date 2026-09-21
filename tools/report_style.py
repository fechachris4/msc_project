"""Shared look for report figures (Imperial MSc guide + Scientific
Communication workshop): no titles inside figures (the LaTeX caption carries
the message), one style per condition in every figure, no gridlines, text
legible at true print size, vector PDF next to each PNG.

Figures are drawn at their printed width, so font sizes are real points.
"""

from pathlib import Path

import matplotlib.pyplot as plt

FULL_WIDTH_IN = 6.3        # A4 text width; include with width=\\linewidth

# One style per condition, used by every figure.
NO_CONTROL = dict(color="0.35", linestyle="--", linewidth=1.2)
REACTIVE = dict(color="#0072B2", linestyle="-", linewidth=1.6)
FEEDFORWARD = dict(color="#D55E00", linestyle="-", linewidth=1.6)
NO_CONTROL_LABEL = "no control (arm rigid on mount)"
REACTIVE_LABEL = "reactive control"
FEEDFORWARD_LABEL = "reactive control + mount-velocity feedforward"


def apply():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.markersize": 4.5,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letters(axes, x=-0.02, y=1.04):
    for letter, ax in zip("abcdefgh", axes):
        ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9,
                fontweight="bold", ha="right", va="bottom")


def save(fig, path):
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)
