"""Kernel-density (mixture-of-Gaussians) mutual information bounds.

Estimator of Kolchinsky & Tracey (2017), *Estimating mixture entropy with
pairwise distances*, as used in appendix B of Saxe et al.

Setup
-----
Take the empirical input distribution to be uniform over the :math:`P` samples.
The hidden activity :math:`h_i = f(x_i)` is deterministic, so for the analysis
we posit additive noise

.. math::
    T = h + \\epsilon, \\qquad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I).

Then :math:`p(T)` is *exactly* a uniform mixture of :math:`P` Gaussians with
means :math:`h_i` and covariance :math:`\\sigma^2 I`.  Mixture entropy has no
closed form, but pairwise-distance bounds do.  Since
:math:`H(T \\mid X) = H(\\epsilon)` is the entropy of a single Gaussian,

.. math::
    I(T;X) = H(T) - \\tfrac{d}{2}\\bigl(\\log 2\\pi\\sigma^2 + 1\\bigr),

and substituting the KL-based upper bound on :math:`H(T)` gives paper equation
(B.1):

.. math::
    I(T;X) \\le -\\frac{1}{P}\\sum_i \\log \\frac{1}{P} \\sum_j
        \\exp\\Bigl(-\\frac{1}{2}\\frac{\\|h_i-h_j\\|^2}{\\sigma^2}\\Bigr).

The Bhattacharyya-based lower bound replaces :math:`\\sigma^2` by
:math:`4\\sigma^2` inside the exponent (equation B.5); the normalising terms
cancel exactly, which is a pleasant check on the algebra.

For the label, :math:`I(T;Y) = H(T) - \\sum_y p(y) H(T \\mid Y=y)`, where each
conditional is the entropy of the sub-mixture over samples with that label
(equations B.2-B.4).

Everything here is a *bound*, not a point estimate, and both bounds are
reported in figures B2/B3.  The bounds are tight when the mixture components
barely overlap and loose when they do.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "pairwise_sq_dists",
    "entropy_upper_kl",
    "entropy_lower_bhattacharyya",
    "gaussian_entropy",
    "mi_kde",
    "information_plane_kde",
]

_LOG2 = np.log(2.0)


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, clipped at zero for stability."""
    X = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    d = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d, 0.0, out=d)
    return d


def _logsumexp_rows(A: np.ndarray) -> np.ndarray:
    m = A.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=1, keepdims=True))).ravel()


def entropy_upper_kl(X: np.ndarray, var: float, chunk: int = 2048) -> float:
    """KL-based *upper* bound on the entropy of the Gaussian mixture, in nats.

    Kolchinsky & Tracey (2017) section 4; the authors' ``kde.entropy_estimator_kl``.

    The distance matrix is formed in row blocks of ``chunk`` so that memory stays
    :math:`O(\\text{chunk} \\times P)` rather than :math:`O(P^2)`.  For MNIST's
    10 000 test points a dense float64 matrix would be 800 MB; blocked, it is a
    few tens of MB, with identical arithmetic.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    sq = np.einsum("ij,ij->i", X, X)
    normconst = (d / 2.0) * np.log(2.0 * np.pi * var)

    total = 0.0
    for start in range(0, n, chunk):
        block = X[start : start + chunk]
        dists = sq[start : start + chunk, None] + sq[None, :] - 2.0 * (block @ X.T)
        np.maximum(dists, 0.0, out=dists)
        dists /= 2.0 * var
        total += float(_logsumexp_rows(-dists).sum())

    lprob_mean = total / n - np.log(n) - normconst
    return float(d / 2.0 - lprob_mean)


def entropy_lower_bhattacharyya(X: np.ndarray, var: float, chunk: int = 2048) -> float:
    """Bhattacharyya-based *lower* bound on the mixture entropy, in nats."""
    d = np.asarray(X).shape[1]
    return entropy_upper_kl(X, 4.0 * var, chunk=chunk) + np.log(0.25) * d / 2.0


def gaussian_entropy(dim: int, var: float) -> float:
    """:math:`H(\\epsilon)` for :math:`\\epsilon \\sim \\mathcal{N}(0,\\sigma^2 I_d)`, in nats.

    This is :math:`H(T \\mid X)`: the only reason the mutual information is
    finite at all.  Without it, :math:`H(T\\mid X) = -\\infty` and
    :math:`I(T;X) = \\infty` (paper appendix C).
    """
    return (dim / 2.0) * (np.log(2.0 * np.pi * var) + 1.0)


def mi_kde(
    activity: np.ndarray,
    labels: np.ndarray,
    var: float = 0.1,
    bound: str = "upper",
    base: float = 2.0,
) -> tuple[float, float]:
    """Return ``(I(T;X), I(T;Y))`` for one layer.

    Parameters
    ----------
    activity
        ``(P, d)`` hidden activity.
    labels
        ``(P,)`` integer class labels.
    var
        Analysis noise variance :math:`\\sigma^2`.  The paper uses 0.1.
    bound
        ``"upper"`` (equations B.1-B.4) or ``"lower"`` (B.5-B.6).
    """
    activity = np.asarray(activity, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n, d = activity.shape

    h_est = entropy_upper_kl if bound == "upper" else entropy_lower_bhattacharyya
    h_cond_noise = gaussian_entropy(d, var)

    h_t = h_est(activity, var)
    i_xt = h_t - h_cond_noise

    h_t_given_y = 0.0
    for y in np.unique(labels):
        mask = labels == y
        h_t_given_y += mask.mean() * h_est(activity[mask], var)
    i_ty = h_t - h_t_given_y

    scale = np.log(base)
    return float(i_xt / scale), float(i_ty / scale)


def information_plane_kde(
    activities: Sequence[np.ndarray],
    labels: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`mi_kde` to every layer."""
    out = [mi_kde(a, labels, **kwargs) for a in activities]
    return (
        np.array([o[0] for o in out]),
        np.array([o[1] for o in out]),
    )
