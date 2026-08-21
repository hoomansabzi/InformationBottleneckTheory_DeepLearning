"""Binning ("simple discretisation") estimator of mutual information.

This is the estimator used by Shwartz-Ziv & Tishby (2017) and replicated in
section 2 of Saxe et al.

The setting
-----------
The input alphabet is finite and fully enumerated (the 4096 twelve-bit
patterns), and taken uniform, so :math:`H(X) = \\log_2 4096 = 12` bits.  A
hidden layer computes a *deterministic* continuous activity :math:`h = f(X)`.
Discretising it,

.. math::
    T = \\mathrm{bin}(h),

makes :math:`T` a deterministic function of :math:`X`, hence
:math:`H(T \\mid X) = 0` and

.. math::
    I(T; X) = H(T) - H(T \\mid X) = H(T) = -\\sum_i p_i \\log p_i,

which is paper equations (1)-(3).  Two consequences worth internalising:

* :math:`I(T;X)` is just the *entropy of the discretised representation*.  Any
  "compression" observed is a drop in :math:`H(T)`, i.e. the layer's activity
  patterns collapsing onto fewer distinct bins.
* :math:`I(T;X) \\le \\log_2 P` always.  With fine enough bins every input gets
  its own code and the curve pins to :math:`\\log_2 P` forever (appendix C,
  figure C3).  The estimator's *resolution* therefore sets the whole scale of
  the y-axis, which is the paper's central complaint.

For the label, :math:`Y` is genuinely discrete, so

.. math::
    I(T; Y) = H(T) - \\sum_y p(y) H(T \\mid Y = y)

is computed directly with no assumptions beyond the binning itself.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

__all__ = [
    "discretize_uniform",
    "discretize_by_size",
    "discretize_net_input",
    "entropy_of_rows",
    "mi_binned",
    "information_plane_binned",
]

_LOG2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# discretisation schemes
# --------------------------------------------------------------------------- #
def discretize_uniform(
    activity: np.ndarray, n_bins: int, lo: float, hi: float
) -> np.ndarray:
    """Bin into ``n_bins`` equal intervals spanning ``[lo, hi]``.

    Used for tanh (30 bins on :math:`[-1, 1]`, paper section 2) and for ReLU
    (100 bins between the min and max activity seen over all of training).
    Values outside the range are clamped into the end bins, which for the ReLU
    protocol is exactly right: the range is defined as the observed extremes,
    so this is equivalent to infinitely many equally spaced bins.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(activity, edges[1:-1], right=False)
    return idx.astype(np.int32)


def discretize_by_size(activity: np.ndarray, bin_size: float) -> np.ndarray:
    """Bin with a fixed bin *width*, as ``floor(h / bin_size)``.

    This is the scheme of the authors' ``simplebinmi.bin_calc_information2``
    and is what produces the MNIST binning row of figures B2/B3 (bin size 0.5).
    """
    return np.floor(activity / bin_size).astype(np.int64)


def discretize_net_input(
    activity: np.ndarray,
    n_bins: int,
    activation: str = "tanh",
    net_input_range: tuple[float, float] = (-50.0, 50.0),
) -> np.ndarray:
    """Bin with edges evenly spaced in *net input* rather than in activity.

    Appendix C: choosing bin edges :math:`b_i \\in \\tanh(\\mathrm{linspace}
    (-50, 50, N))` places bins tightly inside the saturation region instead of
    uniformly across the output range.  Measured this way the tanh network no
    longer compresses (figure C2), demonstrating that the information-plane
    trajectory is a property of the arbitrary binning choice and not of the
    network.
    """
    grid = np.linspace(net_input_range[0], net_input_range[1], n_bins + 1)
    if activation == "tanh":
        edges = np.tanh(grid)
    elif activation == "sigmoid":
        edges = 1.0 / (1.0 + np.exp(-grid))
    elif activation == "softsign":
        edges = grid / (1.0 + np.abs(grid))
    else:  # pragma: no cover - user error path
        raise ValueError(f"net-input binning not defined for {activation!r}")
    edges = np.unique(edges)
    return np.digitize(activity, edges[1:-1], right=False).astype(np.int32)


# --------------------------------------------------------------------------- #
# entropy of a set of discrete vectors
# --------------------------------------------------------------------------- #
def _row_ids(codes: np.ndarray) -> np.ndarray:
    """Map each row of an integer matrix to a unique id.

    Every neuron in the layer is binned separately; the layer's symbol is the
    *joint* pattern across neurons, so rows must be treated as atomic.
    """
    codes = np.ascontiguousarray(codes)
    if codes.ndim == 1:
        codes = codes[:, None]
    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    return inverse.ravel()


def entropy_of_rows(codes: np.ndarray, base: float = 2.0) -> float:
    """Plug-in entropy :math:`-\\sum_i p_i \\log p_i` of the row distribution."""
    ids = _row_ids(codes)
    counts = np.bincount(ids)
    p = counts[counts > 0] / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(base))


# --------------------------------------------------------------------------- #
# mutual information
# --------------------------------------------------------------------------- #
def mi_binned(
    activity: np.ndarray,
    labels: np.ndarray,
    *,
    scheme: Literal["uniform", "size", "net_input", "exact"] = "uniform",
    n_bins: int = 30,
    bounds: tuple[float, float] = (-1.0, 1.0),
    bin_size: float = 0.5,
    activation: str = "tanh",
    base: float = 2.0,
) -> tuple[float, float]:
    """Return ``(I(T;X), I(T;Y))`` in ``base``-logarithm units.

    Parameters
    ----------
    activity
        ``(P, n_units)`` hidden activity, one row per input pattern.  The rows
        are assumed to enumerate the input distribution (uniform over ``P``).
    labels
        ``(P,)`` integer class labels.
    scheme
        ``"uniform"`` -> :func:`discretize_uniform`,
        ``"size"``    -> :func:`discretize_by_size`,
        ``"net_input"`` -> :func:`discretize_net_input`,
        ``"exact"``   -> no binning at all: the float32 activations are already a
        finite, very fine discretisation (~:math:`2^{32}` levels per unit), which
        is what a real network on real hardware actually computes.  This is the
        "binning at full machine precision" of appendix C / figure C3.

    Notes
    -----
    ``I(T;X) = H(T)`` exactly, because the map :math:`X \\to T` is
    deterministic *and* the ``P`` rows are the whole input distribution.  This
    is only legitimate when every input pattern appears exactly once; for a
    resampled dataset it would be an approximation.
    """
    activity = np.asarray(activity, dtype=np.float64)
    if scheme == "uniform":
        codes = discretize_uniform(activity, n_bins, *bounds)
    elif scheme == "size":
        codes = discretize_by_size(activity, bin_size)
    elif scheme == "net_input":
        codes = discretize_net_input(activity, n_bins, activation=activation)
    elif scheme == "exact":
        # Reinterpret the float32 bit patterns as integers: two activities share
        # a symbol only if they are bit-identical.
        codes = activity.astype(np.float32).view(np.int32)
    else:  # pragma: no cover - user error path
        raise ValueError(f"unknown scheme {scheme!r}")

    h_t = entropy_of_rows(codes, base=base)  # == I(T; X)

    labels = np.asarray(labels).ravel()
    h_t_given_y = 0.0
    for y in np.unique(labels):
        mask = labels == y
        h_t_given_y += mask.mean() * entropy_of_rows(codes[mask], base=base)

    return h_t, h_t - h_t_given_y


def information_plane_binned(
    activities: Sequence[np.ndarray],
    labels: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`mi_binned` to every layer.

    Returns ``(I_XT, I_TY)`` arrays of length ``len(activities)``.
    """
    out = [mi_binned(a, labels, **kwargs) for a in activities]
    ixt = np.array([o[0] for o in out])
    ity = np.array([o[1] for o in out])
    return ixt, ity
