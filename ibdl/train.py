"""Training loops instrumented for information-plane analysis.

Three things must be recorded over training, at log-spaced checkpoints:

1. the post-nonlinearity activity of every layer, evaluated on the *whole*
   analysis set (for the Tishby data this is all 4096 patterns, i.e. the entire
   input distribution);
2. learning curves;
3. the gradient signal-to-noise ratio and weight norms of each layer
   (paper equations I.1-I.2 and appendix D).

Checkpoints are log-spaced because the interesting dynamics span four orders of
magnitude in epoch number; the paper's figures use a log-coloured epoch axis
for the same reason.

Gradient SNR
------------
For layer :math:`l`,

.. math::
    m_l = \\bigl\\| \\langle \\partial E/\\partial W_l \\rangle \\bigr\\|_F,
    \\qquad
    s_l = \\bigl\\| \\mathrm{STD}(\\partial E/\\partial W_l) \\bigr\\|_F,
    \\qquad \\mathrm{SNR}_l = m_l / s_l,

where the mean and the *element-wise* standard deviation run over training
examples (``snr_mode="per_sample"``, used for the small networks) or over SGD
minibatches (``snr_mode="per_minibatch"``, used for MNIST, where per-sample
gradients over 825k parameters will not fit in memory).

Shwartz-Ziv & Tishby read the drop in this ratio as a transition from a "drift"
to a "diffusion" phase and attribute compression to it.  The point of logging
it here is that the transition shows up in ReLU networks too, which never
compress -- so it cannot be what causes compression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

__all__ = [
    "log_spaced_epochs",
    "TrainingLog",
    "train",
    "LOSSES",
]

Array = np.ndarray


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def _bce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on independent sigmoid outputs.

    The paper's replication network ends in *two sigmoidal neurons*; treating
    them as independent Bernoulli units is what makes that output layer
    double-saturating, and hence the only layer that compresses in the ReLU
    network of figure 1B.
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def _ce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy against one-hot targets."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error, summed over output units (paper equation 5)."""
    return ((pred - target) ** 2).sum(-1).mean()


LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "bce": _bce_from_logits,
    "ce": _ce_from_logits,
    "mse": _mse,
}


# --------------------------------------------------------------------------- #
# checkpoint schedule
# --------------------------------------------------------------------------- #
def log_spaced_epochs(n_epochs: int, n_points: int = 100, n_dense: int = 20) -> Array:
    """Epoch indices: every epoch up to ``n_dense``, then geometric spacing.

    Early training moves fast (weights are small, the tanh network is still
    linear), so the first epochs need dense sampling; later dynamics are slow
    and log-spaced sampling suffices.
    """
    dense = np.arange(0, min(n_dense, n_epochs) + 1)
    if n_epochs <= n_dense:
        return dense
    sparse = np.geomspace(max(n_dense, 1), n_epochs, n_points).round().astype(int)
    return np.unique(np.concatenate([dense, sparse, [n_epochs]]))


# --------------------------------------------------------------------------- #
# log container
# --------------------------------------------------------------------------- #
@dataclass
class TrainingLog:
    """Everything recorded during a run."""

    epochs: Array = field(default_factory=lambda: np.array([], dtype=int))
    layer_names: list[str] = field(default_factory=list)

    # learning curves (one entry per checkpoint)
    train_loss: Array = field(default_factory=lambda: np.array([]))
    test_loss: Array = field(default_factory=lambda: np.array([]))
    train_acc: Array = field(default_factory=lambda: np.array([]))
    test_acc: Array = field(default_factory=lambda: np.array([]))

    # diagnostics, shape (n_checkpoints, n_layers)
    weight_norms: Array = field(default_factory=lambda: np.zeros((0, 0)))
    grad_mean_norm: Array = field(default_factory=lambda: np.zeros((0, 0)))
    grad_std_norm: Array = field(default_factory=lambda: np.zeros((0, 0)))
    grad_snr: Array = field(default_factory=lambda: np.zeros((0, 0)))

    # optional heavy payloads
    activations: list[list[Array]] = field(default_factory=list)
    layer_maps: list[list[Array]] = field(default_factory=list)
    measurements: list[dict] = field(default_factory=list)

    wall_time: float = 0.0

    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    def measurement_array(self, key: str) -> Array:
        """Stack ``measurements[i][key]`` across checkpoints."""
        return np.array([m[key] for m in self.measurements])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"TrainingLog(checkpoints={len(self.epochs)}, "
            f"layers={self.n_layers}, "
            f"activations={'yes' if self.activations else 'no'}, "
            f"{self.wall_time:.1f}s)"
        )


# --------------------------------------------------------------------------- #
# gradient statistics
# --------------------------------------------------------------------------- #
def _per_sample_grad_stats(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: Callable,
    chunk_size: int = 512,
) -> tuple[list[float], list[float]]:
    """Frobenius norms of the mean and element-wise std of per-sample gradients.

    Uses ``torch.func.vmap(grad(...))`` so that all per-example gradients are
    obtained in one vectorised pass rather than a Python loop.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def single_loss(p, b, x, y):
        out = functional_call(model, (p, b), (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))

    grad_fn = vmap(grad(single_loss), in_dims=(None, None, 0, 0))

    weight_keys = [k for k in params if k.endswith("weight")]
    # Welford-style accumulation so memory stays O(n_params), not O(P n_params).
    total = {k: torch.zeros_like(params[k]) for k in weight_keys}
    total_sq = {k: torch.zeros_like(params[k]) for k in weight_keys}
    n = 0

    for start in range(0, X.shape[0], chunk_size):
        xb = X[start : start + chunk_size]
        yb = Y[start : start + chunk_size]
        g = grad_fn(params, buffers, xb, yb)
        for k in weight_keys:
            total[k] += g[k].sum(dim=0)
            total_sq[k] += (g[k] ** 2).sum(dim=0)
        n += xb.shape[0]

    mean_norms, std_norms = [], []
    for k in weight_keys:
        mean = total[k] / n
        var = torch.clamp(total_sq[k] / n - mean**2, min=0.0)
        mean_norms.append(float(mean.norm()))
        std_norms.append(float(var.sqrt().norm()))
    return mean_norms, std_norms


def _minibatch_grad_stats(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: Callable,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    """Same statistics, but across SGD *minibatch* gradients.

    This is the protocol the paper uses for the MNIST networks (figures B2/B3,
    row 4): the vector of mean updates across minibatches versus the vector of
    their standard deviations.
    """
    P = X.shape[0]
    perm = torch.randperm(P, generator=generator)
    weights = [p for n_, p in model.named_parameters() if n_.endswith("weight")]

    totals = [torch.zeros_like(w) for w in weights]
    total_sqs = [torch.zeros_like(w) for w in weights]
    n_batches = 0

    for start in range(0, P, batch_size):
        idx = perm[start : start + batch_size]
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(X[idx]), Y[idx])
        grads = torch.autograd.grad(loss, weights)
        for i, g in enumerate(grads):
            totals[i] += g
            total_sqs[i] += g**2
        n_batches += 1

    mean_norms, std_norms = [], []
    for i in range(len(weights)):
        mean = totals[i] / n_batches
        var = torch.clamp(total_sqs[i] / n_batches - mean**2, min=0.0)
        mean_norms.append(float(mean.norm()))
        std_norms.append(float(var.sqrt().norm()))
    model.zero_grad(set_to_none=True)
    return mean_norms, std_norms


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: Callable,
    classification: bool,
) -> tuple[float, float]:
    out = model(X)
    loss = float(loss_fn(out, Y))
    if not classification:
        return loss, float("nan")
    acc = float((out.argmax(-1) == Y.argmax(-1)).float().mean())
    return loss, acc


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    X_train: Array,
    Y_train: Array,
    *,
    X_test: Array | None = None,
    Y_test: Array | None = None,
    X_analysis: Array | None = None,
    n_epochs: int = 8000,
    batch_size: int | None = 256,
    lr: float = 0.0004,
    optimizer: Literal["sgd", "adam"] = "sgd",
    momentum: float = 0.0,
    loss: str = "bce",
    classification: bool = True,
    checkpoints: Sequence[int] | None = None,
    n_checkpoints: int = 100,
    store_activations: bool = True,
    store_layer_maps: bool = False,
    measure_fn: Callable[[int, nn.Module, list[Array]], dict] | None = None,
    track_grad_snr: bool = True,
    snr_mode: Literal["per_sample", "per_minibatch"] = "per_sample",
    seed: int = 0,
    progress: bool = True,
) -> TrainingLog:
    """Train ``model`` while logging everything the paper's figures need.

    Parameters
    ----------
    X_analysis
        Inputs on which activations are recorded.  For the Tishby dataset this
        should be **all 4096 patterns**: they *are* the input distribution, so
        binning-based mutual information computed on them is exact.  Defaults
        to ``X_train``.
    batch_size
        ``None`` means full-batch gradient descent (BGD).  Section 4 turns this
        knob to show that compression survives the removal of all stochasticity.
    measure_fn
        Called at each checkpoint as ``measure_fn(epoch, model, activations)``;
        its returned dict is appended to ``log.measurements``.  Use this for
        MNIST, where storing raw activations for every checkpoint would need
        gigabytes.
    snr_mode
        ``"per_sample"`` follows equations I.1-I.2 literally; ``"per_minibatch"``
        follows the MNIST protocol of appendix B.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    loss_fn = LOSSES[loss]
    Xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    Ytr = torch.as_tensor(np.asarray(Y_train, dtype=np.float32))
    Xte = None if X_test is None else torch.as_tensor(np.asarray(X_test, dtype=np.float32))
    Yte = None if Y_test is None else torch.as_tensor(np.asarray(Y_test, dtype=np.float32))
    Xan = Xtr if X_analysis is None else torch.as_tensor(
        np.asarray(X_analysis, dtype=np.float32)
    )

    P = Xtr.shape[0]
    full_batch = batch_size is None
    bs = P if full_batch else int(batch_size)

    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:  # pragma: no cover - user error path
        raise ValueError(f"unknown optimizer {optimizer!r}")

    ckpts = (
        np.asarray(sorted(set(int(c) for c in checkpoints)))
        if checkpoints is not None
        else log_spaced_epochs(n_epochs, n_checkpoints)
    )
    ckpt_set = set(int(c) for c in ckpts)

    log = TrainingLog(layer_names=list(model.layer_names()))
    recorded: list[int] = []
    tr_loss, te_loss, tr_acc, te_acc = [], [], [], []
    wnorms, gmean, gstd = [], [], []

    def record(epoch: int) -> None:
        model.eval()
        with torch.no_grad():
            _, acts = model.forward_with_acts(Xan)
            acts_np = [a.detach().cpu().numpy() for a in acts]

        recorded.append(epoch)
        l, a = _evaluate(model, Xtr, Ytr, loss_fn, classification)
        tr_loss.append(l)
        tr_acc.append(a)
        if Xte is not None:
            l, a = _evaluate(model, Xte, Yte, loss_fn, classification)
            te_loss.append(l)
            te_acc.append(a)
        else:
            te_loss.append(float("nan"))
            te_acc.append(float("nan"))

        wnorms.append(model.weight_norms())

        if track_grad_snr:
            if snr_mode == "per_sample":
                m, s = _per_sample_grad_stats(model, Xtr, Ytr, loss_fn)
            else:
                m, s = _minibatch_grad_stats(model, Xtr, Ytr, loss_fn, bs, generator)
            gmean.append(m)
            gstd.append(s)

        if store_activations:
            log.activations.append(acts_np)
        if store_layer_maps:
            log.layer_maps.append(model.layer_maps())
        if measure_fn is not None:
            log.measurements.append(measure_fn(epoch, model, acts_np))
        model.train()

    t0 = time.time()
    if 0 in ckpt_set:
        record(0)

    for epoch in range(1, n_epochs + 1):
        model.train()
        if full_batch:
            opt.zero_grad(set_to_none=True)
            loss_fn(model(Xtr), Ytr).backward()
            opt.step()
        else:
            perm = torch.randperm(P, generator=generator)
            for start in range(0, P, bs):
                idx = perm[start : start + bs]
                opt.zero_grad(set_to_none=True)
                loss_fn(model(Xtr[idx]), Ytr[idx]).backward()
                opt.step()

        if epoch in ckpt_set:
            record(epoch)
            if progress and len(recorded) % 20 == 0:
                print(
                    f"  epoch {epoch:6d}/{n_epochs}  "
                    f"train {tr_loss[-1]:.4f}  test {te_loss[-1]:.4f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    log.epochs = np.array(recorded)
    log.train_loss = np.array(tr_loss)
    log.test_loss = np.array(te_loss)
    log.train_acc = np.array(tr_acc)
    log.test_acc = np.array(te_acc)
    log.weight_norms = np.array(wnorms)
    if track_grad_snr:
        log.grad_mean_norm = np.array(gmean)
        log.grad_std_norm = np.array(gstd)
        with np.errstate(divide="ignore", invalid="ignore"):
            log.grad_snr = log.grad_mean_norm / log.grad_std_norm
    log.wall_time = time.time() - t0
    return log
