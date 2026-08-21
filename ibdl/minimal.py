"""The minimal three-neuron model of paper section 2 (figure 2).

A scalar Gaussian input :math:`X \\sim \\mathcal{N}(0,1)` is scaled by a single
weight, passed through a nonlinearity, and binned:

.. math::
    h = f(w_1 X), \\qquad T = \\mathrm{bin}(h).

Because :math:`T` is a deterministic function of :math:`X`,
:math:`H(T\\mid X) = 0` and

.. math::
    I(T;X) = H(T) = -\\sum_i p_i \\log p_i .

For a monotone :math:`f` the bin probabilities are available in closed form
from the Gaussian CDF (paper equation 4):

.. math::
    p_i = P\\bigl(f^{-1}(b_i)/w_1 \\le X < f^{-1}(b_{i+1})/w_1\\bigr)
        = \\Phi\\!\\bigl(f^{-1}(b_{i+1})/w_1\\bigr) - \\Phi\\!\\bigl(f^{-1}(b_i)/w_1\\bigr).

No sampling, no estimator error -- this is the exact value of the quantity that
the information-plane plots are estimating.

What it shows
-------------
* **tanh**: :math:`I(T;X)` rises, peaks, then *falls to 1 bit* as :math:`w_1`
  grows.  Large weights push almost every input into one of the two saturation
  bins, so :math:`T` degenerates to a coin flip.  Compression.
* **ReLU**: :math:`I(T;X)` grows without bound.  Half the inputs land in the
  zero bin (contributing a fixed 1 bit at most), while the positive half spreads
  over ever more bins as the weight grows.  No compression, ever.

Since networks start with small weights and must grow them to compute anything
nonlinear (appendix D), a tanh network traverses this curve left to right during
training -- and that traversal *is* the "compression phase".
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

__all__ = [
    "INVERSES",
    "bin_probabilities",
    "mi_minimal_model",
    "sweep_weight",
]

def _softplus_inverse(y: np.ndarray) -> np.ndarray:
    """:math:`f^{-1}(y) = \\log(e^y - 1)` for :math:`f(x) = \\log(1+e^x)`.

    Evaluated as ``y + log1p(-exp(-y))`` for large ``y``, since ``expm1(y)``
    overflows past ``y ~ 709``.  Softplus is single-sided saturating, so its
    curve must rise monotonically; an overflow here would fabricate a spurious
    compression phase and reverse the paper's conclusion for this activation.
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, -np.inf)
    small = (y > 0) & (y < 30.0)
    large = y >= 30.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[small] = np.log(np.expm1(y[small]))
        out[large] = y[large] + np.log1p(-np.exp(-y[large]))
    return out


#: Inverse of each nonlinearity, used to pull bin edges back to input space.
#: ``None`` marks a nonlinearity handled by a special case.
INVERSES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.arctanh,
    "linear": lambda y: y,
    "sigmoid": lambda y: np.log(y / (1.0 - y)),
    "softsign": lambda y: y / (1.0 - np.abs(y)),
    "softplus": lambda y: _softplus_inverse(y),
    "relu": lambda y: y,  # valid for y > 0; y <= 0 handled separately
}

BOUNDED: dict[str, tuple[float, float]] = {
    "tanh": (-1.0, 1.0),
    "sigmoid": (0.0, 1.0),
    "softsign": (-1.0, 1.0),
}


def _bin_edges(
    activation: str,
    n_bins: int,
    bin_width: float,
    max_edge: float,
    net_input_binning: bool,
    net_input_range: tuple[float, float],
) -> np.ndarray:
    """Bin edges in *activation* space."""
    if net_input_binning:
        # Appendix C: evenly spaced in net input, mapped through f.  For tanh
        # this crowds bins into the saturation region instead of spreading them
        # uniformly over [-1, 1].
        grid = np.linspace(net_input_range[0], net_input_range[1], n_bins + 1)
        if activation == "tanh":
            return np.tanh(grid)
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-grid))
        if activation == "softsign":
            return grid / (1.0 + np.abs(grid))
        raise ValueError(f"net-input binning undefined for {activation!r}")

    if activation in BOUNDED:
        lo, hi = BOUNDED[activation]
        return np.linspace(lo, hi, n_bins + 1)

    # Unbounded activations: fixed-width bins tiling the whole range.  The paper
    # bins between the min and max activity ever observed, which it notes gives
    # the same answer as infinitely many equally spaced bins -- so we tile out
    # to ``max_edge`` and let the tail probability fall into the final bin.
    if activation in ("relu", "softplus"):
        return np.arange(0.0, max_edge + bin_width, bin_width)
    return np.arange(-max_edge, max_edge + bin_width, bin_width)


def bin_probabilities(
    w1: float,
    activation: str = "tanh",
    n_bins: int = 30,
    bin_width: float = 2.0 / 30.0,
    max_edge: float | None = None,
    n_sigma: float = 8.0,
    net_input_binning: bool = False,
    net_input_range: tuple[float, float] = (-50.0, 50.0),
) -> np.ndarray:
    """Exact probability of each bin under :math:`X \\sim \\mathcal{N}(0,1)`.

    Parameters
    ----------
    w1
        The single first-layer weight.
    n_bins
        Number of bins for bounded activations (30 in the paper for tanh).
    bin_width
        Bin width for unbounded activations.  Defaults to the tanh bin width
        (:math:`2/30`) so the comparison between panels is like for like.
    max_edge
        How far to tile bins for unbounded activations.  ``None`` (default)
        auto-scales to ``n_sigma * w1``, since the activity of an unbounded unit
        is :math:`w_1 X` with :math:`X` standard normal.  A fixed value would
        silently truncate the distribution at large ``w1`` and turn the
        monotonically rising ReLU curve into a spurious falling one.
    n_sigma
        Number of standard deviations of activity to tile when auto-scaling.
    """
    w1 = float(w1)
    if w1 == 0.0:
        return np.array([1.0])
    if max_edge is None:
        max_edge = max(n_sigma * abs(w1), 10.0 * bin_width)

    edges = _bin_edges(
        activation, n_bins, bin_width, max_edge, net_input_binning, net_input_range
    )

    if activation == "relu":
        # h = 0 for all x <= 0, so the first bin absorbs half the mass plus the
        # inputs mapping into [0, edges[1]).
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = norm.cdf(edges[1:] / w1)
        p = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
        return p[p > 0]

    if activation == "softplus":
        # f(x) = log(1+e^x) > 0 everywhere; f^{-1}(y) -> -inf as y -> 0, so the
        # lowest edge maps to -inf and the CDF starts at 0.
        inv = _softplus_inverse(edges)
        cdf = norm.cdf(inv / w1)
        p = np.diff(np.concatenate([cdf, [1.0]]))
        return p[p > 0]

    inv = INVERSES[activation]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_edges = inv(edges.astype(np.float64))
    cdf = norm.cdf(x_edges / w1)
    # Clamp the outermost edges to the full range so probabilities sum to one.
    cdf[0], cdf[-1] = 0.0, 1.0
    p = np.diff(cdf)
    return p[p > 0]


def mi_minimal_model(w1: float, base: float = 2.0, **kwargs) -> float:
    """:math:`I(T;X) = H(T)` for the minimal model, exactly."""
    p = bin_probabilities(w1, **kwargs)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def sweep_weight(
    weights: np.ndarray,
    activation: str = "tanh",
    base: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """:math:`I(T;X)` as a function of the weight (figure 2C/2D)."""
    return np.array(
        [mi_minimal_model(w, base=base, activation=activation, **kwargs) for w in weights]
    )
