"""Figures in the style of the paper.

The signature plot is the *information plane*: each layer traces a curve with
:math:`I(T;X)` on the x-axis and :math:`I(T;Y)` on the y-axis, coloured by
training epoch, with layers at the same epoch joined by a faint line.  Reading
it: movement **right** is fitting (the layer becomes more informative about the
input), movement **left** is compression, movement **up** is fitting the label.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

__all__ = [
    "use_paper_style",
    "information_plane",
    "learning_curves",
    "activation_histograms",
    "gradient_snr_panel",
    "layer_curves",
    "save",
]

FIGDIR = "figures"


def use_paper_style() -> None:
    """Consistent, print-friendly defaults for every figure in the report."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "figure.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str, directory: str = FIGDIR) -> str:
    """Save as both PDF (for LaTeX) and PNG (for quick viewing)."""
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    pdf = d / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(d / f"{name}.png")
    return str(pdf)


# --------------------------------------------------------------------------- #
# information plane
# --------------------------------------------------------------------------- #
def information_plane(
    epochs: Sequence[int],
    I_XT: np.ndarray,
    I_TY: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    cmap: str = "viridis",
    layer_names: Sequence[str] | None = None,
    connect_layers: bool = True,
    connect_every: int | None = None,
    log_color: bool = True,
    colorbar: bool = True,
    xlabel: str = r"$I(X;T)$  [bits]",
    ylabel: str = r"$I(T;Y)$  [bits]",
    markersize: float = 22,
    annotate_layers: bool = False,
) -> plt.Axes:
    """Plot information-plane trajectories.

    Parameters
    ----------
    I_XT, I_TY
        Arrays of shape ``(n_checkpoints, n_layers)``.
    connect_layers
        Draw the thin lines joining layers at the same epoch, as the paper does;
        they make the layer *ordering* visible (input layer far right, output
        layer far left).
    connect_every
        Draw a connector for every ``connect_every``-th checkpoint.  Defaults to
        a spacing that yields about 25 connectors, which keeps the ordering
        readable without the lines swamping the trajectories.
    log_color
        Colour by :math:`\\log(\\text{epoch})`, matching the paper's colour bar.
    """
    I_XT = np.asarray(I_XT)
    I_TY = np.asarray(I_TY)
    epochs = np.asarray(epochs)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.6))

    e = np.where(epochs == 0, 1, epochs) if log_color else epochs
    norm = LogNorm(vmin=max(e.min(), 1), vmax=e.max()) if log_color else Normalize(
        e.min(), e.max()
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    if connect_layers and I_XT.shape[1] > 1:
        step = connect_every or max(1, len(epochs) // 25)
        for i in range(0, len(epochs), step):
            ax.plot(
                I_XT[i], I_TY[i], "-", color=sm.to_rgba(e[i]), alpha=0.35, lw=0.5,
                zorder=1,
            )

    n_layers = I_XT.shape[1]
    for j in range(n_layers):
        ax.scatter(
            I_XT[:, j], I_TY[:, j], c=e, cmap=cmap, norm=norm, s=markersize,
            edgecolors="none", zorder=2,
        )
        if annotate_layers and layer_names is not None:
            # Nudge the label back into the axes: the final point of the
            # right-most layer otherwise collides with the colour bar.
            frac = (I_XT[-1, j] - I_XT.min()) / max(np.ptp(I_XT), 1e-12)
            dx, ha = (-5, "right") if frac > 0.75 else (5, "left")
            ax.annotate(
                layer_names[j],
                (I_XT[-1, j], I_TY[-1, j]),
                textcoords="offset points",
                xytext=(dx, 5),
                ha=ha,
                fontsize=7,
                color="0.25",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("epoch", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return ax


def layer_curves(
    epochs: Sequence[int],
    values: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    layer_names: Sequence[str] | None = None,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "epoch",
    logx: bool = True,
    logy: bool = False,
    cmap: str = "plasma",
    hline: float | None = None,
    hline_label: str = "",
) -> plt.Axes:
    """One curve per layer against training time.

    Used for :math:`I(X;T)` versus epoch (which shows compression as a *falling*
    curve much more directly than the information plane does), weight norms, and
    gradient SNR.
    """
    values = np.asarray(values)
    epochs = np.asarray(epochs)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.2))

    n = values.shape[1]
    colors = plt.get_cmap(cmap)(np.linspace(0.05, 0.85, n))
    x = np.where(epochs == 0, 0.5, epochs) if logx else epochs
    for j in range(n):
        label = layer_names[j] if layer_names is not None else f"layer {j + 1}"
        ax.plot(x, values[:, j], color=colors[j], label=label)

    if hline is not None:
        ax.axhline(hline, ls="--", color="0.4", lw=1.0, label=hline_label or None)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(ncol=2)
    return ax


# --------------------------------------------------------------------------- #
# learning curves
# --------------------------------------------------------------------------- #
def learning_curves(
    epochs: Sequence[int],
    train: np.ndarray,
    test: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    ylabel: str = "loss",
    title: str = "",
    logx: bool = True,
    logy: bool = False,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 3.0))
    epochs = np.asarray(epochs)
    x = np.where(epochs == 0, 0.5, epochs) if logx else epochs
    ax.plot(x, train, label="train", color="#1f77b4")
    ax.plot(x, test, label="test", color="#d62728")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


# --------------------------------------------------------------------------- #
# activation histograms (appendix E)
# --------------------------------------------------------------------------- #
def activation_histograms(
    epochs: Sequence[int],
    activations: Sequence[Sequence[np.ndarray]],
    *,
    layer_names: Sequence[str] | None = None,
    n_bins: int = 40,
    bounds: tuple[float, float] | None = (-1.0, 1.0),
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
    title: str = "",
) -> plt.Figure:
    """Density of hidden activity per layer over training (figures E1/E2).

    This is the direct evidence for the paper's mechanism: in a tanh network the
    mass migrates into the top and bottom bins (saturation), collapsing
    :math:`H(T)`.  In a ReLU network a fixed fraction sits at exactly zero while
    the rest disperses without bound, so :math:`H(T)` grows.
    """
    n_layers = len(activations[0])
    figsize = figsize or (2.3 * n_layers, 2.6)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize, squeeze=False)
    axes = axes[0]
    epochs = np.asarray(epochs)

    for j, ax in enumerate(axes):
        if bounds is None:
            lo = min(float(a[j].min()) for a in activations)
            hi = max(float(a[j].max()) for a in activations)
        else:
            lo, hi = bounds
        edges = np.linspace(lo, hi, n_bins + 1)
        H = np.stack(
            [np.histogram(a[j].ravel(), bins=edges, density=False)[0] for a in activations]
        ).T.astype(float)
        H /= np.maximum(H.sum(axis=0, keepdims=True), 1)

        x = np.where(epochs == 0, 0.5, epochs)
        ax.pcolormesh(x, 0.5 * (edges[:-1] + edges[1:]), H, cmap=cmap, shading="nearest")
        ax.set_xscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(layer_names[j] if layer_names is not None else f"layer {j + 1}")
        ax.grid(False)
        if j == 0:
            ax.set_ylabel("activation")
    if title:
        fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# gradient SNR (appendix I)
# --------------------------------------------------------------------------- #
def gradient_snr_panel(
    epochs: Sequence[int],
    grad_mean: np.ndarray,
    grad_std: np.ndarray,
    weight_norms: np.ndarray,
    *,
    layer_names: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "",
) -> plt.Figure:
    """One panel per layer: mean/std gradient norms, their ratio, and ||W||.

    Reproduces the fourth row of figures B2/B3 and figure I1.  The step-like
    drop in the red SNR curve is the "drift -> diffusion" transition; the point
    of the figure is that it appears whether or not the network compresses.
    """
    n_layers = grad_mean.shape[1]
    figsize = figsize or (2.6 * n_layers, 2.8)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize, squeeze=False, sharex=True)
    axes = axes[0]
    epochs = np.asarray(epochs)
    x = np.where(epochs == 0, 0.5, epochs)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = grad_mean / grad_std

    for j, ax in enumerate(axes):
        ax.plot(x, grad_mean[:, j], color="#1f77b4", label=r"$\|$mean grad$\|$")
        ax.plot(x, grad_std[:, j], color="#ff7f0e", label=r"$\|$std grad$\|$")
        ax.plot(x, snr[:, j], color="#d62728", label="SNR")
        ax.plot(x, weight_norms[:, j], color="#2ca02c", label=r"$\|W\|_F$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(layer_names[j] if layer_names is not None else f"layer {j + 1}")
        if j == 0:
            ax.set_ylabel("magnitude")
    axes[-1].legend(loc="best", fontsize=7)
    if title:
        fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return fig


def compression_legend(ax: plt.Axes) -> None:
    """Annotate the reading of the information plane."""
    handles = [
        Line2D([], [], color="0.3", marker=r"$\rightarrow$", ls="none", label="fitting"),
        Line2D([], [], color="0.3", marker=r"$\leftarrow$", ls="none", label="compression"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
