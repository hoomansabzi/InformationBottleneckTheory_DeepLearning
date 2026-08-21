"""Pre-computed information planes, cached separately from the training runs.

Estimating a KDE information plane is not cheap: every checkpoint needs a
:math:`P \\times P` pairwise-distance matrix per layer, plus one per class label.
For the Tishby dataset that is a 4096 x 4096 matrix, and a single four-activation
by five-seed sweep needs several thousand of them.

Training and measuring are therefore cached separately.  Notebooks load a
:class:`PlaneSet`, which holds the information plane already estimated under one
choice of estimator, so the analysis and plotting stay interactive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cache import cached
from .estimators.kraskov import entropy_kraskov_layers
from .experiments import (
    TishbyConfig,
    information_plane_from_log,
    relu_binning_bounds,
    run_tishby,
)

__all__ = ["PlaneSet", "tishby_plane", "tishby_plane_averaged", "kraskov_entropy_curve"]


@dataclass
class PlaneSet:
    """An information plane plus the metadata needed to plot it."""

    epochs: np.ndarray
    I_XT: np.ndarray  # (n_checkpoints, n_layers)
    I_TY: np.ndarray
    layer_names: list[str]
    train_acc: np.ndarray = field(default_factory=lambda: np.array([]))
    test_acc: np.ndarray = field(default_factory=lambda: np.array([]))
    n_runs: int = 1
    estimator: str = "binning"

    @property
    def compression(self) -> np.ndarray:
        """Bits lost from each layer's peak :math:`I(X;T)` by the end of training."""
        return self.I_XT.max(axis=0) - self.I_XT[-1]

    def summary(self) -> str:
        drops = ", ".join(
            f"{n}: {d:+.2f}" for n, d in zip(self.layer_names, self.compression)
        )
        return f"[{self.estimator}, n={self.n_runs}] compression per layer -> {drops}"


def _binning_kwargs(cfg: TishbyConfig, log) -> dict:
    """Binning protocol appropriate to the activation function.

    Bounded activations get evenly spaced bins over their range (30 on
    :math:`[-1,1]` for tanh, per section 2).  Unbounded ones follow appendix C:
    train first, then use 100 equally spaced bins between the extreme activities
    ever observed.
    """
    from .models import ACTIVATION_INFO

    info = ACTIVATION_INFO[cfg.activation]
    if info["bounds"] is not None:
        return dict(n_bins=30, bounds=info["bounds"])
    lo, hi = relu_binning_bounds(log)
    return dict(n_bins=100, bounds=(lo, hi))


def tishby_plane(
    cfg: TishbyConfig,
    labels: np.ndarray,
    *,
    estimator: str = "binning",
    var: float = 0.1,
    scheme: str = "uniform",
    force: bool = False,
    verbose: bool = False,
) -> PlaneSet:
    """Information plane for one Tishby run, cached."""
    suffix = f"{estimator}_{scheme}" + (f"_v{var:g}" if estimator == "kde" else "")
    tag = f"plane_{cfg.tag}_{suffix}"

    def compute() -> PlaneSet:
        log = cached(cfg.tag, lambda: run_tishby(cfg), verbose=False)
        kw = _binning_kwargs(cfg, log) if estimator == "binning" else {}
        if scheme in ("net_input", "exact"):
            kw = dict(n_bins=30) if scheme == "net_input" else {}
        ixt, ity = information_plane_from_log(
            log, labels, estimator=estimator, scheme=scheme,
            activation=cfg.activation, var=var, **kw,
        )
        return PlaneSet(
            epochs=log.epochs, I_XT=ixt, I_TY=ity, layer_names=log.layer_names,
            train_acc=log.train_acc, test_acc=log.test_acc, estimator=estimator,
        )

    return cached(tag, compute, force=force, verbose=verbose)


def tishby_plane_averaged(
    activation: str,
    labels: np.ndarray,
    n_seeds: int = 5,
    *,
    estimator: str = "kde",
    var: float = 0.1,
    **cfg_kwargs,
) -> PlaneSet:
    """Average the information plane over several random seeds.

    The paper averages figure B1 over 50 repetitions; five is enough to show the
    effect is not seed-specific, and is what this project uses.
    """
    planes = []
    for seed in range(n_seeds):
        cfg = TishbyConfig(
            activation=activation,
            seed=seed,
            track_grad_snr=(activation in ("tanh", "relu") and seed == 0),
            **cfg_kwargs,
        )
        planes.append(tishby_plane(cfg, labels, estimator=estimator, var=var))
    return PlaneSet(
        epochs=planes[0].epochs,
        I_XT=np.mean([p.I_XT for p in planes], axis=0),
        I_TY=np.mean([p.I_TY for p in planes], axis=0),
        layer_names=planes[0].layer_names,
        train_acc=np.mean([p.train_acc for p in planes], axis=0),
        test_acc=np.mean([p.test_acc for p in planes], axis=0),
        n_runs=len(planes),
        estimator=estimator,
    )


def kraskov_entropy_curve(
    activation: str,
    n_seeds: int = 5,
    k: int = 2,
    **cfg_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Kraskov entropy of each hidden layer over training (figure B5).

    Returns ``(epochs, H)`` with ``H`` of shape ``(n_checkpoints, n_hidden)``.
    The output layer is excluded, matching the paper's figure B5, which shows
    hidden layers 0-4 of sizes 10-7-5-4-3.
    """
    tag = f"kraskov_{activation}_k{k}_n{n_seeds}"

    def compute():
        ents, epochs = [], None
        for seed in range(n_seeds):
            cfg = TishbyConfig(
                activation=activation,
                seed=seed,
                track_grad_snr=(activation in ("tanh", "relu") and seed == 0),
                **cfg_kwargs,
            )
            log = cached(cfg.tag, lambda c=cfg: run_tishby(c), verbose=False)
            ents.append(
                np.array([entropy_kraskov_layers(a[:-1], k=k) for a in log.activations])
            )
            epochs = log.epochs
        return epochs, np.mean(ents, axis=0)

    return cached(tag, compute, verbose=False)
