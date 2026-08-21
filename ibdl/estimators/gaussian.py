"""Exact mutual information for linear-Gaussian networks.

Section 3 and appendix G of Saxe et al.  This is the most important estimator
in the paper's argument, because it removes estimation error from the picture
entirely: in a deep *linear* network with Gaussian inputs every hidden layer is
jointly Gaussian with the input and the output, so :math:`I(T;X)` and
:math:`I(T;Y)` are available in closed form.  Whatever the information plane
does here, it is not an artefact of a poor estimator -- it follows from the
noise assumption and nothing else.

Model
-----
Inputs :math:`X \\sim \\mathcal{N}(0, \\Sigma_X)`, teacher
:math:`Y = W_o X + \\epsilon_o` with :math:`\\epsilon_o \\sim \\mathcal{N}(0,
\\sigma_o^2 I)`.  A hidden layer applies the cumulative map :math:`\\bar{W} =
W_l \\cdots W_1`, and *for analysis only* we add
:math:`\\epsilon_{MI} \\sim \\mathcal{N}(0, \\sigma_{MI}^2 I)`:

.. math::
    T = \\bar{W} X + \\epsilon_{MI}.

Then, with :math:`\\Sigma_T = \\bar{W}\\Sigma_X\\bar{W}^\\top +
\\sigma_{MI}^2 I`,

.. math::
    I(T;X) = \\tfrac12 \\log|\\Sigma_T| - \\tfrac12\\log|\\sigma_{MI}^2 I|,

which is paper equation (6), and :math:`I(T;Y)` follows from equations
(G.1)-(G.4) via :math:`I = H(Y) + H(T) - H(Y,T)`.

A note on the paper's equations
-------------------------------
The paper states :math:`X \\sim \\mathcal{N}(0, \\tfrac{1}{N_i} I_{N_i})` in
the text but writes equations (5), (6) and (G.1)-(G.3) with :math:`\\Sigma_X`
suppressed (i.e. as if :math:`\\Sigma_X = I`), and equation (6) additionally
drops the factor :math:`\\tfrac12` that appears correctly in (G.2).  This
module carries :math:`\\Sigma_X` explicitly and keeps the :math:`\\tfrac12`.
The difference is a constant offset and a global scale on the axes -- it does
not touch any qualitative conclusion -- but the numbers here are the correct
ones for the stated generative model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "log_det",
    "gaussian_mi_input",
    "gaussian_mi_output",
    "gaussian_mi_subspace",
    "generalization_error",
    "information_plane_linear",
]

_LOG2 = np.log(2.0)


def log_det(A: np.ndarray) -> float:
    """:math:`\\log|A|` for a symmetric positive definite matrix."""
    sign, val = np.linalg.slogdet(np.asarray(A, dtype=np.float64))
    if sign <= 0:  # pragma: no cover - would signal a broken covariance
        raise np.linalg.LinAlgError(f"non-positive determinant (sign={sign})")
    return float(val)


def _sigma_x(n_inputs: int, sigma_x2: float | np.ndarray | None) -> np.ndarray:
    """Input covariance; defaults to :math:`I/N_i` as in the paper's text."""
    if sigma_x2 is None:
        return np.eye(n_inputs) / n_inputs
    if np.isscalar(sigma_x2):
        return np.eye(n_inputs) * float(sigma_x2)
    return np.asarray(sigma_x2, dtype=np.float64)


def gaussian_mi_input(
    W_bar: np.ndarray,
    sigma_mi2: float = 1.0,
    sigma_x2: float | np.ndarray | None = None,
    base: float = 2.0,
) -> float:
    """:math:`I(T;X)` for :math:`T = \\bar{W}X + \\epsilon_{MI}` (equation 6).

    Parameters
    ----------
    W_bar
        ``(n_hidden, n_inputs)`` cumulative map from input to this layer.
    sigma_mi2
        Analysis noise variance :math:`\\sigma_{MI}^2`.  The paper uses 1.0.
        Its role is to set the *resolution* at which the representation is
        read out; it is not part of the network.
    """
    W_bar = np.asarray(W_bar, dtype=np.float64)
    nh, ni = W_bar.shape
    Sx = _sigma_x(ni, sigma_x2)

    Sigma_T = W_bar @ Sx @ W_bar.T + sigma_mi2 * np.eye(nh)
    mi = 0.5 * (log_det(Sigma_T) - nh * np.log(sigma_mi2))
    return float(mi / np.log(base))


def gaussian_mi_output(
    W_bar: np.ndarray,
    W_o: np.ndarray,
    sigma_o: float,
    sigma_mi2: float = 1.0,
    sigma_x2: float | np.ndarray | None = None,
    base: float = 2.0,
) -> float:
    """:math:`I(T;Y)` via equations (G.1)-(G.4).

    :math:`(T, Y)` are jointly Gaussian, so
    :math:`I(T;Y) = H(T) + H(Y) - H(T,Y)` and the :math:`\\log(2\\pi e)` terms
    cancel, leaving only log-determinants of the three covariances.
    """
    W_bar = np.asarray(W_bar, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    nh, ni = W_bar.shape
    no = W_o.shape[0]
    Sx = _sigma_x(ni, sigma_x2)

    S_T = W_bar @ Sx @ W_bar.T + sigma_mi2 * np.eye(nh)
    S_Y = W_o @ Sx @ W_o.T + (sigma_o**2) * np.eye(no)
    S_TY = W_bar @ Sx @ W_o.T  # (nh, no) cross-covariance

    joint = np.block([[S_T, S_TY], [S_TY.T, S_Y]])
    mi = 0.5 * (log_det(S_T) + log_det(S_Y) - log_det(joint))
    return float(mi / np.log(base))


def gaussian_mi_subspace(
    W_bar: np.ndarray,
    subspace: slice | np.ndarray,
    sigma_mi2: float = 1.0,
    sigma_x2: float | np.ndarray | None = None,
    base: float = 2.0,
) -> float:
    """:math:`I(T; X_S)` for a subset :math:`S` of input coordinates.

    Needed for figure 6, where the input splits into a task-relevant and a
    task-irrelevant block.  Writing
    :math:`T = \\bar{W}_S X_S + \\bar{W}_{\\bar S} X_{\\bar S} + \\epsilon`,
    conditioning on :math:`X_S` leaves the complement acting as extra noise:

    .. math::
        I(T; X_S) = \\tfrac12\\log|\\Sigma_T|
                  - \\tfrac12\\log\\bigl|\\bar{W}_{\\bar S}\\Sigma_{\\bar S}
                    \\bar{W}_{\\bar S}^\\top + \\sigma_{MI}^2 I\\bigr|.

    So the "information about the irrelevant subspace" is a genuine mutual
    information, not a heuristic -- which is what makes the figure 6 claim
    (irrelevant information *does* compress, concurrently with fitting) sharp.
    """
    W_bar = np.asarray(W_bar, dtype=np.float64)
    nh, ni = W_bar.shape
    Sx = _sigma_x(ni, sigma_x2)

    idx = np.zeros(ni, dtype=bool)
    idx[subspace] = True
    comp = ~idx

    S_T = W_bar @ Sx @ W_bar.T + sigma_mi2 * np.eye(nh)
    Wc = W_bar[:, comp]
    S_cond = Wc @ Sx[np.ix_(comp, comp)] @ Wc.T + sigma_mi2 * np.eye(nh)

    mi = 0.5 * (log_det(S_T) - log_det(S_cond))
    return float(mi / np.log(base))


def generalization_error(
    W_tot: np.ndarray,
    W_o: np.ndarray,
    sigma_o: float,
    sigma_x2: float | np.ndarray | None = None,
) -> float:
    """True generalisation error, paper equation (5).

    .. math::
        E_g = \\mathbb{E}\\|Y - \\hat{Y}\\|^2
            = \\mathrm{tr}\\bigl[(W_o - W_{tot})\\Sigma_X(W_o - W_{tot})^\\top\\bigr]
              + \\sigma_o^2 N_o.

    With :math:`\\Sigma_X = I` this reduces to the paper's
    :math:`\\|W_o - W_{tot}\\|_F^2 + \\sigma_o^2`; with the stated
    :math:`\\Sigma_X = I/N_i` there is an extra :math:`1/N_i`.  Note this is
    the *exact* expected error, not an estimate from a finite test set.
    """
    W_tot = np.asarray(W_tot, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    ni = W_o.shape[1]
    Sx = _sigma_x(ni, sigma_x2)
    D = W_o - W_tot
    return float(np.trace(D @ Sx @ D.T) + (sigma_o**2) * W_o.shape[0])


def information_plane_linear(
    layer_maps: Sequence[np.ndarray],
    W_o: np.ndarray,
    sigma_o: float,
    sigma_mi2: float = 1.0,
    sigma_x2: float | np.ndarray | None = None,
    base: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(I(T;X), I(T;Y))`` for every layer of a deep linear network."""
    ixt = np.array(
        [gaussian_mi_input(W, sigma_mi2, sigma_x2, base) for W in layer_maps]
    )
    ity = np.array(
        [
            gaussian_mi_output(W, W_o, sigma_o, sigma_mi2, sigma_x2, base)
            for W in layer_maps
        ]
    )
    return ixt, ity
