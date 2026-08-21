"""Generate figures/fig00_activations.{pdf,png} -- the activation taxonomy.

The distinction the whole paper turns on is not "linear vs nonlinear" but
**double-saturating** (bounded above *and* below) versus single-sided or
unbounded.  This figure states that distinction visually, before any
information-theoretic machinery appears.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "figure.facecolor": "white",
})

x = np.linspace(-5, 5, 800)

BOUNDED = [
    ("tanh",     np.tanh(x),                      "#1f77b4"),
    ("softsign", x / (1 + np.abs(x)),             "#17becf"),
    ("sigmoid",  1 / (1 + np.exp(-x)),            "#9467bd"),
]
UNBOUNDED = [
    ("linear",   x,                                                       "0.65"),
    ("softplus", np.log1p(np.exp(x)),                                     "#ff7f0e"),
    ("GELU",     x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
                                                                          "#8c564b"),
    ("ReLU",     np.maximum(x, 0),                                        "#d62728"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

# ---- A: double-saturating -------------------------------------------------- #
ax = axes[0]
for b in (-1.0, 0.0, 1.0):
    ax.axhline(b, ls="--", lw=1.0, color="0.6", zorder=1)
for name, y, c in BOUNDED:
    ax.plot(x, y, color=c, label=name, zorder=2)
ax.set_ylim(-1.75, 1.75)
ax.set_xlabel("net input  $w x$")
ax.set_ylabel("activation  $h$")
ax.set_title("A: double-saturating — a ceiling on BOTH sides")
ax.legend(loc="upper left", ncol=3, columnspacing=1.0)
ax.text(0.0, -1.62, "grow $w$: almost every input is pushed onto an asymptote,\n"
                    "so distinct inputs become indistinguishable",
        ha="center", va="bottom", fontsize=8.5, color="#1f77b4")

# ---- B: single-sided / unbounded ------------------------------------------- #
ax = axes[1]
ax.axhline(0, ls="--", lw=1.0, color="0.6")
for name, y, c in UNBOUNDED:
    ls = (0, (4, 2)) if name == "linear" else "-"
    lw = 1.4 if name == "linear" else 2.0
    ax.plot(x, y, color=c, label=name, ls=ls, lw=lw)
ax.set_ylim(-1.75, 5.4)
ax.set_xlabel("net input  $w x$")
ax.set_title("B: single-sided or unbounded — no ceiling at all")
ax.legend(loc="upper left", ncol=2, columnspacing=1.0)
ax.text(0.0, -1.62, "grow $w$: the activations simply spread further apart,\n"
                    "so distinct inputs stay distinguishable",
        ha="center", va="bottom", fontsize=8.5, color="#d62728")

fig.tight_layout()
FIGDIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig00_activations.{ext}")
print("wrote", FIGDIR / "fig00_activations.pdf")
