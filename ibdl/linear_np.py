"""Fast NumPy trainer for deep *linear* networks.

Sections 3 and 5 of the paper run many long SGD experiments on tiny linear
networks (100 inputs, batch size 5, thousands of epochs).  PyTorch's per-step
overhead dominates completely at that scale -- a 1-1-1 network costs roughly
400 microseconds per optimiser step in torch, essentially all of it framework
bookkeeping.  Writing the linear case out by hand makes it two orders of
magnitude faster and, because a deep linear network's gradients are three lines
of algebra, it stays fully transparent.

Forward and backward
--------------------
With :math:`h_0 = x` and :math:`h_l = W_l h_{l-1}`, and per-sample loss
:math:`E_n = \\|y_n - \\hat{y}_n\\|^2`,

.. math::
    \\delta_L = -2(y - \\hat{y}), \\qquad
    \\delta_{l-1} = W_l^\\top \\delta_l, \\qquad
    \\frac{\\partial E}{\\partial W_l} = \\delta_l h_{l-1}^\\top .

Per-sample gradient statistics
------------------------------
Equations (I.1)-(I.2) need the mean and the element-wise standard deviation of
:math:`\\partial E_n/\\partial W_l` across samples.  Both are matrix products,
never requiring the :math:`P \\times n_l \\times n_{l-1}` stack to be formed:

.. math::
    \\mathbb{E}[G]_{ij} = \\tfrac1P \\sum_n \\delta_{ni} h_{nj}, \\qquad
    \\mathbb{E}[G^2]_{ij} = \\tfrac1P \\sum_n \\delta_{ni}^2 h_{nj}^2 ,

i.e. ``delta.T @ h / P`` and ``(delta**2).T @ (h**2) / P``.
"""

from __future__ import annotations

import time
from typing import Callable, Literal, Sequence

import numpy as np

from .train import TrainingLog, log_spaced_epochs

__all__ = ["LinearNetNP", "train_linear"]

Array = np.ndarray


class LinearNetNP:
    """Deep linear network :math:`\\hat{Y} = W_L \\cdots W_1 X`, in NumPy.

    Mirrors the interface of :class:`ibdl.models.DeepLinear` (``layer_maps``,
    ``total_map``, ``weight_norms``, ``layer_names``) so the same analysis code
    works with either.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        init_std: float | str | None = "fan_in",
        seed: int = 0,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.weights: list[Array] = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if init_std is None or init_std == "fan_in":
                std = 1.0 / np.sqrt(n_in)
            else:
                std = float(init_std)
            self.weights.append(rng.normal(0.0, std, size=(n_out, n_in)))

    # -- inference ---------------------------------------------------------- #
    def forward(self, X: Array) -> tuple[Array, list[Array]]:
        """Return ``(output, [h_1, ..., h_L])`` for a batch of rows."""
        acts: list[Array] = []
        h = X
        for W in self.weights:
            h = h @ W.T
            acts.append(h)
        return h, acts

    # -- exact linear maps -------------------------------------------------- #
    def layer_maps(self) -> list[Array]:
        """Cumulative maps :math:`\\bar{W}_l = W_l \\cdots W_1`."""
        maps: list[Array] = []
        W = None
        for Wl in self.weights:
            W = Wl.copy() if W is None else Wl @ W
            maps.append(W.copy())
        return maps

    def total_map(self) -> Array:
        return self.layer_maps()[-1]

    def weight_norms(self) -> list[float]:
        return [float(np.linalg.norm(W)) for W in self.weights]

    def layer_names(self) -> list[str]:
        return [f"L{i + 1} ({n})" for i, n in enumerate(self.layer_sizes[1:])]

    @property
    def n_layers(self) -> int:
        return len(self.weights)


# --------------------------------------------------------------------------- #
def _grads(net: LinearNetNP, X: Array, Y: Array) -> list[Array]:
    """Mean-over-batch gradients of the summed-squared-error loss."""
    out, acts = net.forward(X)
    B = X.shape[0]
    delta = -2.0 * (Y - out)  # (B, n_L)
    grads: list[Array | None] = [None] * net.n_layers
    for l in range(net.n_layers - 1, -1, -1):
        h_prev = X if l == 0 else acts[l - 1]
        grads[l] = delta.T @ h_prev / B
        if l > 0:
            delta = delta @ net.weights[l]
    return grads  # type: ignore[return-value]


def _grad_stats(net: LinearNetNP, X: Array, Y: Array) -> tuple[list[float], list[float]]:
    """Frobenius norms of mean and element-wise std of per-sample gradients."""
    out, acts = net.forward(X)
    P = X.shape[0]
    delta = -2.0 * (Y - out)

    means: list[Array | None] = [None] * net.n_layers
    seconds: list[Array | None] = [None] * net.n_layers
    for l in range(net.n_layers - 1, -1, -1):
        h_prev = X if l == 0 else acts[l - 1]
        means[l] = delta.T @ h_prev / P
        seconds[l] = (delta**2).T @ (h_prev**2) / P
        if l > 0:
            delta = delta @ net.weights[l]

    mean_norms, std_norms = [], []
    for m, s in zip(means, seconds):
        var = np.maximum(s - m**2, 0.0)
        mean_norms.append(float(np.linalg.norm(m)))
        std_norms.append(float(np.linalg.norm(np.sqrt(var))))
    return mean_norms, std_norms


def _mse(net: LinearNetNP, X: Array, Y: Array) -> float:
    out, _ = net.forward(X)
    return float(((Y - out) ** 2).sum(axis=-1).mean())


def train_linear(
    net: LinearNetNP,
    X_train: Array,
    Y_train: Array,
    *,
    X_test: Array | None = None,
    Y_test: Array | None = None,
    X_analysis: Array | None = None,
    n_epochs: int = 10_000,
    batch_size: int | None = None,
    lr: float = 0.001,
    checkpoints: Sequence[int] | None = None,
    n_checkpoints: int = 100,
    store_activations: bool = False,
    store_layer_maps: bool = True,
    measure_fn: Callable[[int, LinearNetNP, list[Array]], dict] | None = None,
    track_grad_snr: bool = True,
    seed: int = 0,
    progress: bool = False,
) -> TrainingLog:
    """Train a :class:`LinearNetNP` with (stochastic) gradient descent on MSE.

    ``batch_size=None`` gives full-batch gradient descent -- the setting used in
    section 4 to show that compression does not require SGD's stochasticity.

    Returns the same :class:`~ibdl.train.TrainingLog` as the torch trainer, so
    downstream analysis and plotting are shared.
    """
    X = np.asarray(X_train, dtype=np.float64)
    Y = np.asarray(Y_train, dtype=np.float64)
    Xte = None if X_test is None else np.asarray(X_test, dtype=np.float64)
    Yte = None if Y_test is None else np.asarray(Y_test, dtype=np.float64)
    Xan = X if X_analysis is None else np.asarray(X_analysis, dtype=np.float64)

    P = X.shape[0]
    bs = P if batch_size is None else int(batch_size)
    rng = np.random.default_rng(seed)

    ckpts = (
        np.asarray(sorted({int(c) for c in checkpoints}))
        if checkpoints is not None
        else log_spaced_epochs(n_epochs, n_checkpoints)
    )
    ckpt_set = {int(c) for c in ckpts}

    log = TrainingLog(layer_names=net.layer_names())
    recorded: list[int] = []
    tr_loss, te_loss, wnorms, gmean, gstd = [], [], [], [], []

    def record(epoch: int) -> None:
        recorded.append(epoch)
        tr_loss.append(_mse(net, X, Y))
        te_loss.append(float("nan") if Xte is None else _mse(net, Xte, Yte))
        wnorms.append(net.weight_norms())
        if track_grad_snr:
            m, s = _grad_stats(net, X, Y)
            gmean.append(m)
            gstd.append(s)
        if store_layer_maps:
            log.layer_maps.append(net.layer_maps())
        acts = None
        if store_activations or measure_fn is not None:
            _, acts = net.forward(Xan)
        if store_activations:
            log.activations.append([a.copy() for a in acts])
        if measure_fn is not None:
            log.measurements.append(measure_fn(epoch, net, acts))

    t0 = time.time()
    if 0 in ckpt_set:
        record(0)

    for epoch in range(1, n_epochs + 1):
        if bs >= P:
            for l, g in enumerate(_grads(net, X, Y)):
                net.weights[l] -= lr * g
        else:
            perm = rng.permutation(P)
            for start in range(0, P, bs):
                idx = perm[start : start + bs]
                for l, g in enumerate(_grads(net, X[idx], Y[idx])):
                    net.weights[l] -= lr * g

        if epoch in ckpt_set:
            record(epoch)
            if progress and len(recorded) % 25 == 0:
                print(
                    f"  epoch {epoch:6d}  train {tr_loss[-1]:.4f}  "
                    f"test {te_loss[-1]:.4f}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    log.epochs = np.array(recorded)
    log.train_loss = np.array(tr_loss)
    log.test_loss = np.array(te_loss)
    log.train_acc = np.full(len(recorded), np.nan)
    log.test_acc = np.full(len(recorded), np.nan)
    log.weight_norms = np.array(wnorms)
    if track_grad_snr:
        log.grad_mean_norm = np.array(gmean)
        log.grad_std_norm = np.array(gstd)
        with np.errstate(divide="ignore", invalid="ignore"):
            log.grad_snr = log.grad_mean_norm / log.grad_std_norm
    log.wall_time = time.time() - t0
    return log
