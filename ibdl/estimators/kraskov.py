"""k-nearest-neighbour (Kraskov) entropy estimator.

Appendix B.3 of Saxe et al.  Rather than commit to a noise variance, this route
estimates the *differential entropy* of the hidden representation directly from
nearest-neighbour distances and exploits the following observation.

If the analysis variable is :math:`T = h + Z` with :math:`Z` independent of
:math:`X` (homoscedastic noise), then

.. math::
    I(T;X) = H(T) - H(T \\mid X) = H(T) - H(Z) = H(T) - c

for a constant :math:`c`.  So whether the layer *compresses* -- the only
question at issue -- can be answered by watching :math:`H(T)` alone, without
ever fixing the noise level.  Decreasing entropy means compression.

The estimator itself (paper equation B.10) is the Kozachenko-Leonenko /
Kraskov et al. (2004) form:

.. math::
    \\hat{H} = \\frac{d}{P}\\sum_{i=1}^{P} \\log(r_i + \\epsilon)
             + \\frac{d}{2}\\log \\pi - \\log\\Gamma(d/2 + 1)
             + \\psi(P) - \\psi(k),

with :math:`r_i` the distance from sample :math:`i` to its :math:`k`-th nearest
neighbour and :math:`\\epsilon = 10^{-16}` guarding against duplicate points
(which do occur once a tanh layer saturates hard).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

__all__ = [
    "entropy_kraskov",
    "entropy_kraskov_layers",
    "duplicate_fraction",
    "zero_neighbour_fraction",
]

_EPS = 1e-16


def duplicate_fraction(activity: np.ndarray) -> float:
    """Fraction of rows that are exact duplicates of some other row.

    This is a *validity check*, not a nicety.  The Kozachenko-Leonenko estimator
    assumes an absolutely continuous distribution.  A ReLU layer violates that
    badly: its output has an **atom at zero** (every unit that is off contributes
    an exact 0), so many activation vectors are bit-identical and the
    distribution is mixed discrete-continuous, for which differential entropy is
    not even defined.

    When duplicates are present the :math:`\\varepsilon` guard in equation
    (B.10) -- there to stop :math:`\\log r_i` diverging -- silently takes over:
    each duplicated point contributes :math:`(d/P)\\log \\varepsilon`, which for
    :math:`\\varepsilon = 10^{-16}` is about :math:`-36.8` nats per dimension.
    The resulting number tracks *how many points coincide*, not how spread out
    the representation is.  Appendix B.3 of the paper does not flag this.
    """
    X = np.asarray(activity)
    if X.ndim == 1:
        X = X[:, None]
    return 1.0 - len(np.unique(X, axis=0)) / len(X)


def zero_neighbour_fraction(activity: np.ndarray, k: int = 2) -> float:
    """Fraction of points whose k-th nearest neighbour is at distance zero."""
    X = np.asarray(activity, dtype=np.float64)
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    return float(np.mean(dists[:, k] <= 0.0))


def entropy_kraskov(
    activity: np.ndarray,
    k: int = 2,
    eps: float = _EPS,
    base: float = 2.0,
    warn_duplicates: float | None = None,
) -> float:
    """Kozachenko-Leonenko entropy estimate (paper equation B.10).

    Parameters
    ----------
    activity
        ``(P, d)`` samples.
    k
        Neighbour rank.  The paper uses ``k = 2``.
    eps
        Additive guard so that duplicated points do not send the estimate to
        :math:`-\\infty`.  With a saturated tanh layer many activity vectors
        become byte-identical, so this term is doing real work.
    base
        Logarithm base of the result (2 -> bits).

    warn_duplicates
        Emit a warning when a substantial fraction of points have a zero-distance
        neighbour, which makes the estimate a function of ``eps`` rather than of
        the data.  See :func:`duplicate_fraction`.

    Returns
    -------
    float
        Estimated differential entropy.  Only *changes* in this quantity are
        meaningful here; the absolute value carries the arbitrary offset
        :math:`c = H(Z)`.
    """
    X = np.asarray(activity, dtype=np.float64)
    P, d = X.shape
    if P <= k:  # pragma: no cover - guard
        raise ValueError(f"need more than k={k} samples, got {P}")

    tree = cKDTree(X)
    # query k+1 because the first neighbour of a point is the point itself
    dists, _ = tree.query(X, k=k + 1)
    r = dists[:, k]

    if warn_duplicates is not None:
        frac = float(np.mean(r <= 0.0))
        if frac > warn_duplicates:
            import warnings

            warnings.warn(
                f"{100 * frac:.1f}% of points have a zero-distance {k}-th "
                "neighbour; the Kraskov estimate is dominated by the eps guard "
                "and is not a meaningful entropy",
                RuntimeWarning,
                stacklevel=2,
            )

    h = (
        (d / P) * np.sum(np.log(r + eps))
        + (d / 2.0) * np.log(np.pi)
        - gammaln(d / 2.0 + 1.0)
        + digamma(P)
        - digamma(k)
    )
    return float(h / np.log(base))


def entropy_kraskov_layers(
    activities: Sequence[np.ndarray], **kwargs
) -> np.ndarray:
    """Apply :func:`entropy_kraskov` to every layer."""
    return np.array([entropy_kraskov(a, **kwargs) for a in activities])
