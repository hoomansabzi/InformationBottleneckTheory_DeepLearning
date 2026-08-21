"""Validation of the mutual-information estimators.

Two kinds of check:

* **Analytic** -- cases where the true value is known in closed form
  (a uniform code, a Gaussian channel, the data processing inequality).
* **Reference** -- agreement with the authors' own implementations in
  ``reference/`` (``simplebinmi.py``, ``kde.py``), reimplemented here in NumPy
  since the originals import Keras.

Run with::

    ./.venv/bin/python tests/test_estimators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ibdl.estimators import binning, gaussian, kde, kraskov  # noqa: E402

RNG = np.random.default_rng(0)
_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return bool(abs(a - b) <= tol * max(1.0, abs(a), abs(b)))


# --------------------------------------------------------------------------- #
# reference reimplementations (authors' code, Keras stripped out)
# --------------------------------------------------------------------------- #
def ref_get_unique_probs(x: np.ndarray):
    """Verbatim logic of ``reference/simplebinmi.py:get_unique_probs``."""
    uniqueids = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1]))
    )
    _, unique_inverse, unique_counts = np.unique(
        uniqueids, return_index=False, return_inverse=True, return_counts=True
    )
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse.ravel()


def ref_bin_calc_information2(labelixs, layerdata, binsize):
    """Verbatim logic of ``reference/simplebinmi.py:bin_calc_information2``."""

    def get_h(d):
        digitized = np.floor(d / binsize).astype("int")
        p_ts, _ = ref_get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0.0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT


def ref_entropy_estimator_kl(x, var):
    """Verbatim logic of ``reference/kde.py:entropy_estimator_kl``."""
    N, dims = x.shape
    x2 = np.sum(np.square(x), axis=1, keepdims=True)
    dists = x2 + x2.T - 2 * (x @ x.T)
    dists2 = dists / (2 * var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    m = dists2.min(axis=1, keepdims=True)
    lprobs = (
        np.log(np.sum(np.exp(-(dists2 - m)), axis=1)) - m.ravel() - np.log(N) - normconst
    )
    h = -np.mean(lprobs)
    return dims / 2 + h


def ref_entropy_estimator_bd(x, var):
    dims = x.shape[1]
    return ref_entropy_estimator_kl(x, 4 * var) + np.log(0.25) * dims / 2


def ref_kde_condentropy(output, var):
    dims = output.shape[1]
    return (dims / 2.0) * (np.log(2 * np.pi * var) + 1)


# --------------------------------------------------------------------------- #
# binning
# --------------------------------------------------------------------------- #
def test_binning_uniform_code() -> None:
    """P distinct patterns, fine bins -> I(T;X) = log2(P) exactly."""
    P = 512
    act = np.linspace(-0.9, 0.9, P)[:, None]
    labels = np.zeros(P, dtype=int)
    ixt, _ = binning.mi_binned(act, labels, n_bins=4096, bounds=(-1, 1))
    check("binning: distinct patterns give log2(P)", close(ixt, np.log2(P), 1e-12),
          f"{ixt:.6f} vs {np.log2(P):.6f}")


def test_binning_collapse() -> None:
    """All activity in one bin -> zero information."""
    act = np.full((256, 3), 0.999)
    labels = RNG.integers(0, 2, 256)
    ixt, ity = binning.mi_binned(act, labels, n_bins=30, bounds=(-1, 1))
    check("binning: collapsed representation gives 0 bits",
          close(ixt, 0.0, 1e-12) and close(ity, 0.0, 1e-12), f"I(T;X)={ixt:g}")


def test_binning_perfect_label_code() -> None:
    """T = Y exactly -> I(T;Y) = H(Y)."""
    labels = RNG.integers(0, 2, 4096)
    act = np.where(labels == 1, 0.9, -0.9)[:, None]
    _, ity = binning.mi_binned(act, labels, n_bins=30, bounds=(-1, 1))
    p = labels.mean()
    hy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    check("binning: T=Y gives I(T;Y)=H(Y)", close(ity, hy, 1e-12),
          f"{ity:.6f} vs {hy:.6f}")


def test_binning_matches_reference() -> None:
    """Fixed-width binning agrees with the authors' bin_calc_information2."""
    P, d = 800, 4
    act = RNG.normal(size=(P, d))
    labels = RNG.integers(0, 3, P)
    binsize = 0.5

    ixt, ity = binning.mi_binned(act, labels, scheme="size", bin_size=binsize,
                                 base=np.e)
    labelixs = {y: labels == y for y in np.unique(labels)}
    ref_h, ref_ity = ref_bin_calc_information2(labelixs, act, binsize)

    check("binning: I(T;X) matches reference (nats)", close(ixt, ref_h, 1e-10),
          f"{ixt:.9f} vs {ref_h:.9f}")
    check("binning: I(T;Y) matches reference (nats)", close(ity, ref_ity, 1e-10),
          f"{ity:.9f} vs {ref_ity:.9f}")


def test_binning_dpi_within_estimator() -> None:
    """Coarsening bins can only reduce H(T)."""
    act = RNG.uniform(-1, 1, size=(2000, 2))
    labels = RNG.integers(0, 2, 2000)
    fine, _ = binning.mi_binned(act, labels, n_bins=256, bounds=(-1, 1))
    coarse, _ = binning.mi_binned(act, labels, n_bins=8, bounds=(-1, 1))
    check("binning: coarser bins give less information", coarse <= fine + 1e-12,
          f"coarse={coarse:.4f} <= fine={fine:.4f}")


def test_net_input_binning_captures_saturation() -> None:
    """Net-input bin edges resolve the saturation region that uniform bins lose."""
    w = 20.0
    x = RNG.normal(size=20000)
    h = np.tanh(w * x)[:, None]
    labels = np.zeros(len(h), dtype=int)
    uni, _ = binning.mi_binned(h, labels, scheme="uniform", n_bins=30, bounds=(-1, 1))
    net, _ = binning.mi_binned(h, labels, scheme="net_input", n_bins=30,
                               activation="tanh")
    check("binning: net-input edges beat uniform edges under saturation", net > uni,
          f"net_input={net:.3f} > uniform={uni:.3f}")


# --------------------------------------------------------------------------- #
# KDE
# --------------------------------------------------------------------------- #
def test_kde_matches_reference() -> None:
    P, d = 400, 5
    act = RNG.normal(size=(P, d))
    labels = RNG.integers(0, 2, P)
    var = 0.1

    ixt, ity = kde.mi_kde(act, labels, var=var, bound="upper", base=np.e)

    ref_h = ref_entropy_estimator_kl(act, var)
    ref_ixt = ref_h - ref_kde_condentropy(act, var)
    ref_hcond = sum(
        (labels == y).mean() * ref_entropy_estimator_kl(act[labels == y], var)
        for y in np.unique(labels)
    )
    ref_ity = ref_h - ref_hcond

    check("kde: I(T;X) matches reference", close(ixt, ref_ixt, 1e-10),
          f"{ixt:.9f} vs {ref_ixt:.9f}")
    check("kde: I(T;Y) matches reference", close(ity, ref_ity, 1e-10),
          f"{ity:.9f} vs {ref_ity:.9f}")


def test_kde_bounds_ordered() -> None:
    act = RNG.normal(size=(300, 4))
    labels = RNG.integers(0, 2, 300)
    up, _ = kde.mi_kde(act, labels, var=0.1, bound="upper")
    lo, _ = kde.mi_kde(act, labels, var=0.1, bound="lower")
    check("kde: lower bound <= upper bound", lo <= up + 1e-10,
          f"{lo:.4f} <= {up:.4f}")


def test_kde_bounded_by_log_P() -> None:
    """I(T;X) <= H(X) = log2(P) for a uniform empirical input distribution."""
    P = 500
    act = RNG.normal(size=(P, 3)) * 10.0  # widely separated -> bound near log2 P
    labels = np.zeros(P, dtype=int)
    up, _ = kde.mi_kde(act, labels, var=0.1, bound="upper")
    check("kde: I(T;X) <= log2(P)", up <= np.log2(P) + 1e-9,
          f"{up:.4f} <= {np.log2(P):.4f}")


def test_kde_identical_points_give_zero() -> None:
    """All samples identical -> the representation carries no information."""
    act = np.zeros((200, 3))
    labels = RNG.integers(0, 2, 200)
    up, _ = kde.mi_kde(act, labels, var=0.1, bound="upper")
    check("kde: identical activity gives 0 bits", close(up, 0.0, 1e-9), f"{up:.2e}")


def test_kde_gaussian_channel() -> None:
    """The bounds bracket the analytic Gaussian-channel value.

    For ``h ~ N(0, s^2 I)`` and ``T = h + N(0, var)``, the mixture density
    converges as ``P -> inf`` to ``N(0, s^2 + var)``, so the true information
    tends to ``d/2 * log2(1 + s^2/var)``.  The KL bound is an *upper* bound on
    ``H(T)`` and so must sit above that value, the Bhattacharyya bound below.
    """
    d, s2, var, P = 1, 25.0, 0.1, 6000
    act = RNG.normal(0, np.sqrt(s2), size=(P, d))
    labels = np.zeros(P, dtype=int)
    up, _ = kde.mi_kde(act, labels, var=var, bound="upper")
    lo, _ = kde.mi_kde(act, labels, var=var, bound="lower")
    analytic = 0.5 * d * np.log2(1 + s2 / var)
    check("kde: bounds bracket the Gaussian channel value",
          lo <= analytic <= up,
          f"lower={lo:.3f} <= analytic={analytic:.3f} <= upper={up:.3f}")


# --------------------------------------------------------------------------- #
# Kraskov
# --------------------------------------------------------------------------- #
def test_kraskov_uniform_entropy() -> None:
    """H of U[0,1]^d is 0; the estimator should be close."""
    for d in (1, 2, 3):
        X = RNG.uniform(0, 1, size=(20000, d))
        h = kraskov.entropy_kraskov(X, k=2, base=np.e)
        check(f"kraskov: H(U[0,1]^{d}) ~ 0", abs(h) < 0.06, f"{h:+.4f} nats")


def test_kraskov_gaussian_entropy() -> None:
    """H of N(0, s^2 I_d) = d/2 log(2 pi e s^2)."""
    d, s = 2, 2.0
    X = RNG.normal(0, s, size=(20000, d))
    h = kraskov.entropy_kraskov(X, k=2, base=np.e)
    true = d / 2 * np.log(2 * np.pi * np.e * s**2)
    check("kraskov: Gaussian entropy", abs(h - true) < 0.06,
          f"{h:.4f} vs {true:.4f} nats")


def test_kraskov_scaling_law() -> None:
    """H(cX) = H(X) + d log c."""
    d, c = 2, 3.0
    X = RNG.normal(size=(10000, d))
    h1 = kraskov.entropy_kraskov(X, k=2, base=np.e)
    h2 = kraskov.entropy_kraskov(c * X, k=2, base=np.e)
    check("kraskov: H(cX) = H(X) + d log c",
          abs((h2 - h1) - d * np.log(c)) < 1e-6,
          f"delta={h2 - h1:.6f} vs {d * np.log(c):.6f}")


def test_kraskov_detects_saturation() -> None:
    """Driving a tanh unit into saturation must lower the estimated entropy."""
    x = RNG.normal(size=(8000, 1))
    h_small = kraskov.entropy_kraskov(np.tanh(0.5 * x), k=2)
    h_large = kraskov.entropy_kraskov(np.tanh(50.0 * x), k=2)
    check("kraskov: saturation lowers entropy", h_large < h_small,
          f"w=50 -> {h_large:.3f} < w=0.5 -> {h_small:.3f} bits")


# --------------------------------------------------------------------------- #
# exact linear-Gaussian
# --------------------------------------------------------------------------- #
def test_gaussian_scalar_channel() -> None:
    """Scalar case: I = 1/2 log2(1 + w^2 sx2 / var)."""
    w, sx2, var = 2.0, 1.0, 0.5
    mi = gaussian.gaussian_mi_input(np.array([[w]]), sigma_mi2=var, sigma_x2=sx2)
    analytic = 0.5 * np.log2(1 + w**2 * sx2 / var)
    check("gaussian: scalar channel matches analytic", close(mi, analytic, 1e-12),
          f"{mi:.9f} vs {analytic:.9f}")


def test_gaussian_diagonal_channel() -> None:
    """Parallel independent channels add."""
    ws = np.array([0.5, 2.0, 5.0])
    W = np.diag(ws)
    var, sx2 = 1.0, 1.0
    mi = gaussian.gaussian_mi_input(W, sigma_mi2=var, sigma_x2=sx2)
    analytic = np.sum(0.5 * np.log2(1 + ws**2 * sx2 / var))
    check("gaussian: parallel channels add", close(mi, analytic, 1e-12),
          f"{mi:.9f} vs {analytic:.9f}")


def test_gaussian_zero_map() -> None:
    W = np.zeros((5, 8))
    check("gaussian: zero map gives 0 bits",
          close(gaussian.gaussian_mi_input(W), 0.0, 1e-12))


def test_gaussian_rescaling_breaks_invariance() -> None:
    """Appendix C, equation (C.5): identical input-output map, different I(T;X).

    Networks w1, w2 and w1/c, c*w2 compute the same function and generalise
    identically, yet the noise assumption makes their measured I(T;X) differ.
    This is the paper's sharpest illustration that the metric is not a property
    of the function computed.
    """
    w1, var = 2.0, 1.0
    mis = [
        gaussian.gaussian_mi_input(np.array([[w1 / c]]), sigma_mi2=var, sigma_x2=1.0)
        for c in (0.25, 1.0, 4.0)
    ]
    check("gaussian: rescaling changes I(T;X) despite identical function",
          mis[0] > mis[1] > mis[2],
          f"c=0.25 -> {mis[0]:.3f}, c=1 -> {mis[1]:.3f}, c=4 -> {mis[2]:.3f} bits")


def test_gaussian_subspace_decomposition() -> None:
    """I(T;X_rel) and I(T;X_irrel) each lie below I(T;X)."""
    W = RNG.normal(size=(10, 100))
    kw = dict(sigma_mi2=1.0, sigma_x2=None)
    full = gaussian.gaussian_mi_input(W, **kw)
    rel = gaussian.gaussian_mi_subspace(W, slice(0, 30), **kw)
    irr = gaussian.gaussian_mi_subspace(W, slice(30, 100), **kw)
    check("gaussian: subspace MI <= total MI", rel <= full + 1e-9 and irr <= full + 1e-9,
          f"full={full:.3f}, rel={rel:.3f}, irrel={irr:.3f}")
    # Chain rule with independent blocks:
    #   I(T;X) = I(T;X_rel) + I(T;X_irrel) + I(X_rel; X_irrel | T),
    # and the synergy term I(X_rel; X_irrel | T) >= 0 because observing T makes
    # the two a-priori-independent blocks dependent ("explaining away").  The
    # parts are therefore SUB-additive.
    synergy = full - (rel + irr)
    check("gaussian: subspaces are sub-additive (non-negative synergy)",
          synergy >= -1e-9,
          f"rel+irrel={rel + irr:.3f} <= full={full:.3f} (synergy {synergy:.3f} bits)")


def test_gaussian_subspace_zero_weights() -> None:
    """If the map ignores a subspace, MI with that subspace is exactly 0."""
    W = RNG.normal(size=(6, 50))
    W[:, 30:] = 0.0
    mi = gaussian.gaussian_mi_subspace(W, slice(30, 50))
    check("gaussian: ignored subspace gives 0 bits", close(mi, 0.0, 1e-9), f"{mi:.2e}")


def test_gaussian_dpi_for_composed_maps() -> None:
    """I(T2;X) <= I(T1;X) when T2 = W2 T1 and noise is added at both stages.

    The paper stresses (appendix C) that the DPI does *not* hold for the
    analysis variables as usually plotted.  Here we build the chain the way the
    DPI actually requires -- propagating the noise -- and confirm it holds.
    """
    W1 = RNG.normal(size=(20, 40)) / np.sqrt(40)
    W2 = RNG.normal(size=(10, 20)) / np.sqrt(20)
    var = 1.0
    i1 = gaussian.gaussian_mi_input(W1, sigma_mi2=var, sigma_x2=1.0)
    # T2 = W2 (W1 X + eps1) + eps2 -> effective noise cov W2 W2' var + var I
    S_eff = var * (W2 @ W2.T) + var * np.eye(10)
    Wtot = W2 @ W1
    S_T2 = Wtot @ Wtot.T + S_eff
    i2 = 0.5 * (gaussian.log_det(S_T2) - gaussian.log_det(S_eff)) / np.log(2)
    check("gaussian: DPI holds when noise is propagated", i2 <= i1 + 1e-9,
          f"I(T2;X)={i2:.3f} <= I(T1;X)={i1:.3f}")


def test_gaussian_mi_output_sanity() -> None:
    """A layer that reproduces the teacher recovers I(T;Y) close to H(Y)-H(noise)."""
    ni, no = 20, 1
    W_o = RNG.normal(size=(no, ni))
    sigma_o = 0.1
    # very large gain -> analysis noise negligible -> I(T;Y) -> I(X;Y)
    W_bar = 1e4 * W_o
    mi = gaussian.gaussian_mi_output(W_bar, W_o, sigma_o, sigma_mi2=1.0, sigma_x2=1.0)
    sy2 = float((W_o @ W_o.T).item())
    analytic = 0.5 * np.log2(1 + sy2 / sigma_o**2)
    check("gaussian: I(T;Y) approaches I(X;Y) for a faithful layer",
          abs(mi - analytic) < 1e-3, f"{mi:.5f} vs {analytic:.5f}")


def test_generalization_error_formula() -> None:
    """E_g at the teacher's own weights equals the irreducible noise floor."""
    W_o = RNG.normal(size=(1, 50))
    sigma_o = 0.3
    eg = gaussian.generalization_error(W_o.copy(), W_o, sigma_o)
    check("gaussian: E_g at W_tot=W_o equals sigma_o^2",
          close(eg, sigma_o**2, 1e-12), f"{eg:.9f} vs {sigma_o**2:.9f}")


# --------------------------------------------------------------------------- #
# cross-estimator consistency
# --------------------------------------------------------------------------- #
def test_kde_vs_exact_gaussian() -> None:
    """KDE upper bound on a linear-Gaussian layer brackets the exact value."""
    ni, nh, P, var = 8, 4, 4000, 1.0
    W = RNG.normal(size=(nh, ni)) / np.sqrt(ni)
    X = RNG.normal(size=(P, ni))
    h = X @ W.T
    exact = gaussian.gaussian_mi_input(W, sigma_mi2=var, sigma_x2=1.0)
    up, _ = kde.mi_kde(h, np.zeros(P, dtype=int), var=var, bound="upper")
    lo, _ = kde.mi_kde(h, np.zeros(P, dtype=int), var=var, bound="lower")
    check("cross: exact value lies within KDE bounds",
          lo - 0.05 <= exact <= up + 0.05,
          f"lower={lo:.3f} <= exact={exact:.3f} <= upper={up:.3f}")


def test_kraskov_vs_exact_gaussian() -> None:
    """Kraskov entropy of a noisy linear layer recovers the exact H(T)."""
    ni, nh, P, var = 6, 3, 30000, 1.0
    W = RNG.normal(size=(nh, ni)) / np.sqrt(ni)
    X = RNG.normal(size=(P, ni))
    T = X @ W.T + RNG.normal(0, np.sqrt(var), size=(P, nh))
    S_T = W @ W.T + var * np.eye(nh)
    exact_h = 0.5 * (nh * np.log(2 * np.pi * np.e) + gaussian.log_det(S_T)) / np.log(2)
    est = kraskov.entropy_kraskov(T, k=3, base=2.0)
    check("cross: Kraskov H(T) matches exact Gaussian H(T)",
          abs(est - exact_h) < 0.1, f"{est:.4f} vs {exact_h:.4f} bits")


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} estimator checks\n" + "=" * 70)
    for fn in tests:
        print(f"\n-- {fn.__name__}")
        fn()
    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): " + ", ".join(_FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
