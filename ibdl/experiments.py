"""Experiment definitions shared by the notebooks.

Each experiment is a module-level function taking a frozen config dataclass, so
that it can be shipped to a worker process by :func:`ibdl.cache.parallel_map`.
Keeping them here rather than inline in the notebooks means the notebooks read
as analysis rather than as plumbing, and the exact settings behind every figure
live in one auditable place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .data import load_mnist, load_tishby, make_relevant_irrelevant, make_student_teacher
from .estimators import binning, gaussian, kde
from .linear_np import LinearNetNP, train_linear
from .models import mnist_net, tishby_net
from .train import TrainingLog, train

__all__ = [
    "TishbyConfig",
    "run_tishby",
    "LinearConfig",
    "run_linear",
    "RelevanceConfig",
    "run_relevance",
    "MnistConfig",
    "run_mnist",
    "information_plane_from_log",
    "relu_binning_bounds",
]


# --------------------------------------------------------------------------- #
# Shwartz-Ziv & Tishby replication (sections 2, 4; figures 1, 5, E1-E2, I1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TishbyConfig:
    """Settings for one run of the 12-10-7-5-4-3-2 network.

    Defaults follow the paper's replication: SGD with 256 samples per batch on
    80% of the 4096 patterns.  The learning rate is the one value not stated in
    the paper; 0.004 was selected here because it reaches the accuracy the paper
    reports (~97% test) within 10 000 epochs.
    """

    activation: str = "tanh"
    batch_size: int | None = 256  # None -> full-batch gradient descent
    lr: float = 0.004
    n_epochs: int = 10_000
    n_checkpoints: int = 60
    train_fraction: float = 0.8
    seed: int = 0
    track_grad_snr: bool = False
    output_activation: str = "sigmoid"

    @property
    def tag(self) -> str:
        opt = "bgd" if self.batch_size is None else f"sgd{self.batch_size}"
        return (
            f"tishby_{self.activation}_{opt}_lr{self.lr:g}"
            f"_e{self.n_epochs}_f{self.train_fraction:g}_s{self.seed}"
        )


def run_tishby(cfg: TishbyConfig) -> TrainingLog:
    """Train one network on the Tishby dataset and log everything."""
    tr, te, full = load_tishby(train_fraction=cfg.train_fraction, seed=cfg.seed)
    net = tishby_net(
        cfg.activation, seed=cfg.seed, output_activation=cfg.output_activation
    )
    log = train(
        net,
        tr.X,
        tr.Y,
        X_test=te.X,
        Y_test=te.Y,
        X_analysis=full.X,  # all 4096 patterns == the full input distribution
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        loss="bce",
        classification=True,
        n_checkpoints=cfg.n_checkpoints,
        store_activations=True,
        track_grad_snr=cfg.track_grad_snr,
        snr_mode="per_sample",
        seed=cfg.seed,
        progress=False,
    )
    return log


def relu_binning_bounds(log: TrainingLog) -> tuple[float, float]:
    """Min and max activity over all units and all epochs.

    The paper's protocol for unbounded activations (appendix C): train first,
    then bin into equally spaced bins between the extremes ever observed.  It
    notes this gives the same estimate as infinitely many equally spaced bins,
    because bins beyond the observed maximum are never occupied.
    """
    lo = min(float(a.min()) for acts in log.activations for a in acts)
    hi = max(float(a.max()) for acts in log.activations for a in acts)
    return lo, hi


def information_plane_from_log(
    log: TrainingLog,
    labels: np.ndarray,
    *,
    estimator: Literal["binning", "kde"] = "binning",
    n_bins: int = 30,
    bounds: tuple[float, float] = (-1.0, 1.0),
    scheme: str = "uniform",
    bin_size: float = 0.5,
    activation: str = "tanh",
    var: float = 0.1,
    bound: str = "upper",
) -> tuple[np.ndarray, np.ndarray]:
    """``(I_XT, I_TY)`` of shape ``(n_checkpoints, n_layers)``."""
    ixt, ity = [], []
    for acts in log.activations:
        if estimator == "binning":
            a, b = binning.information_plane_binned(
                acts, labels, scheme=scheme, n_bins=n_bins, bounds=bounds,
                bin_size=bin_size, activation=activation,
            )
        else:
            a, b = kde.information_plane_kde(acts, labels, var=var, bound=bound)
        ixt.append(a)
        ity.append(b)
    return np.array(ixt), np.array(ity)


# --------------------------------------------------------------------------- #
# deep linear student/teacher (section 3; figures 3, 4A-B, F1, H1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LinearConfig:
    """Settings for a deep linear student/teacher run."""

    hidden: tuple[int, ...] = (100,)
    n_inputs: int = 100
    n_train: int = 100
    n_test: int = 10_000
    snr: float = 1.0
    batch_size: int | None = None  # None -> BGD
    lr: float = 0.05
    n_epochs: int = 10_000
    n_checkpoints: int = 80
    init_std: float | str = "fan_in"
    sigma_mi2: float = 1.0
    seed: int = 0

    @property
    def tag(self) -> str:
        opt = "bgd" if self.batch_size is None else f"sgd{self.batch_size}"
        h = "-".join(str(h) for h in self.hidden)
        return (
            f"linear_{self.n_inputs}-{h}-1_P{self.n_train}_snr{self.snr:g}"
            f"_{opt}_lr{self.lr:g}_e{self.n_epochs}_s{self.seed}"
        )


@dataclass
class LinearResult:
    """A linear run plus its exact information plane."""

    log: TrainingLog
    I_XT: np.ndarray
    I_TY: np.ndarray
    E_g: np.ndarray
    W_o: np.ndarray
    sigma_o: float
    cfg: LinearConfig


def run_linear(cfg: LinearConfig) -> LinearResult:
    """Train a deep linear student and compute its *exact* information plane."""
    prob = make_student_teacher(
        n_inputs=cfg.n_inputs,
        n_outputs=1,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        snr=cfg.snr,
        seed=cfg.seed,
    )
    sizes = [cfg.n_inputs, *cfg.hidden, 1]
    net = LinearNetNP(sizes, init_std=cfg.init_std, seed=cfg.seed)
    log = train_linear(
        net,
        prob.X_train,
        prob.Y_train,
        X_test=prob.X_test,
        Y_test=prob.Y_test,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        n_checkpoints=cfg.n_checkpoints,
        store_layer_maps=True,
        track_grad_snr=True,
        seed=cfg.seed,
    )

    ixt, ity, eg = [], [], []
    for maps in log.layer_maps:
        a, b = gaussian.information_plane_linear(
            maps, prob.W_o, prob.sigma_o, sigma_mi2=cfg.sigma_mi2
        )
        ixt.append(a)
        ity.append(b)
        eg.append(gaussian.generalization_error(maps[-1], prob.W_o, prob.sigma_o))

    return LinearResult(
        log=log,
        I_XT=np.array(ixt),
        I_TY=np.array(ity),
        E_g=np.array(eg),
        W_o=prob.W_o,
        sigma_o=prob.sigma_o,
        cfg=cfg,
    )


# --------------------------------------------------------------------------- #
# task-relevant / task-irrelevant split (section 5; figure 6)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelevanceConfig:
    """Section 5: 30 task-relevant and 70 task-irrelevant input dimensions."""

    n_relevant: int = 30
    n_irrelevant: int = 70
    hidden: tuple[int, ...] = (100,)
    n_train: int = 1_000
    n_test: int = 10_000
    snr: float = 5.0
    batch_size: int | None = 5
    lr: float = 0.05
    n_epochs: int = 3_000
    n_checkpoints: int = 60
    init_std: float | str = 0.03
    sigma_mi2: float = 1.0
    seed: int = 0

    @property
    def tag(self) -> str:
        return (
            f"relevance_{self.n_relevant}rel_{self.n_irrelevant}irr"
            f"_P{self.n_train}_snr{self.snr:g}_lr{self.lr:g}"
            f"_e{self.n_epochs}_i{self.init_std:g}_s{self.seed}"
        )


@dataclass
class RelevanceResult:
    log: TrainingLog
    I_XT: np.ndarray  # information with the whole input
    I_XrelT: np.ndarray  # information with the task-relevant subspace
    I_XirrT: np.ndarray  # information with the task-irrelevant subspace
    I_TY: np.ndarray
    E_g: np.ndarray
    cfg: RelevanceConfig


def run_relevance(cfg: RelevanceConfig) -> RelevanceResult:
    """Train on a task where part of the input is pure nuisance."""
    prob = make_relevant_irrelevant(
        n_relevant=cfg.n_relevant,
        n_irrelevant=cfg.n_irrelevant,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        snr=cfg.snr,
        seed=cfg.seed,
    )
    n_in = cfg.n_relevant + cfg.n_irrelevant
    net = LinearNetNP([n_in, *cfg.hidden, 1], init_std=cfg.init_std, seed=cfg.seed)
    log = train_linear(
        net,
        prob.X_train,
        prob.Y_train,
        X_test=prob.X_test,
        Y_test=prob.Y_test,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        n_checkpoints=cfg.n_checkpoints,
        store_layer_maps=True,
        track_grad_snr=True,
        seed=cfg.seed,
    )

    rel = slice(0, cfg.n_relevant)
    irr = slice(cfg.n_relevant, n_in)
    ixt, irel, iirr, ity, eg = [], [], [], [], []
    for maps in log.layer_maps:
        ixt.append([gaussian.gaussian_mi_input(W, cfg.sigma_mi2) for W in maps])
        irel.append([gaussian.gaussian_mi_subspace(W, rel, cfg.sigma_mi2) for W in maps])
        iirr.append([gaussian.gaussian_mi_subspace(W, irr, cfg.sigma_mi2) for W in maps])
        ity.append(
            [
                gaussian.gaussian_mi_output(W, prob.W_o, prob.sigma_o, cfg.sigma_mi2)
                for W in maps
            ]
        )
        eg.append(gaussian.generalization_error(maps[-1], prob.W_o, prob.sigma_o))

    return RelevanceResult(
        log=log,
        I_XT=np.array(ixt),
        I_XrelT=np.array(irel),
        I_XirrT=np.array(iirr),
        I_TY=np.array(ity),
        E_g=np.array(eg),
        cfg=cfg,
    )


# --------------------------------------------------------------------------- #
# MNIST (section 2 / appendix B; figures 1C-D, B2-B3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MnistConfig:
    """784-1024-20-20-20-10 on MNIST, as in figure 1C/D.

    The three narrow layers are the paper's choice, made so the kernel density
    estimator stays well conditioned: KDE accuracy degrades quickly with
    dimension and the estimate is built from test-set points only.
    """

    activation: str = "tanh"
    batch_size: int = 128
    lr: float = 0.05
    n_epochs: int = 60
    n_checkpoints: int = 40
    kde_var: float = 0.1
    kde_samples: int = 4_000  # subsample of the test set used for the KDE bounds
    bin_size: float = 0.5  # binning-based estimate (figures B2/B3, row 3)
    seed: int = 0

    @property
    def tag(self) -> str:
        return (
            f"mnist_{self.activation}_b{self.batch_size}_lr{self.lr:g}"
            f"_e{self.n_epochs}_k{self.kde_samples}_s{self.seed}"
        )


def run_mnist(cfg: MnistConfig) -> TrainingLog:
    """Train on MNIST, measuring the information plane at each checkpoint.

    Activations are *not* stored: 10 000 test points x 1094 units x 40
    checkpoints would be several gigabytes.  Instead ``measure_fn`` estimates
    the mutual information inside the training loop and only the numbers are
    kept.  This is why ``kde_samples`` matters -- the 1024-unit layer needs a
    ``kde_samples`` x ``kde_samples`` distance matrix at every checkpoint.
    """
    trn, tst = load_mnist()
    net = mnist_net(cfg.activation, seed=cfg.seed)

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(tst.X), size=min(cfg.kde_samples, len(tst.X)), replace=False)
    X_an = tst.X[idx]
    y_an = tst.y[idx]

    def measure(epoch: int, model, acts: list[np.ndarray]) -> dict:
        kde_up_x, kde_up_y, kde_lo_x, bin_x, bin_y = [], [], [], [], []
        for a in acts:
            a = np.asarray(a, dtype=np.float64)
            ux, uy = kde.mi_kde(a, y_an, var=cfg.kde_var, bound="upper")
            lx, _ = kde.mi_kde(a, y_an, var=cfg.kde_var, bound="lower")
            bx, by = binning.mi_binned(a, y_an, scheme="size", bin_size=cfg.bin_size)
            kde_up_x.append(ux)
            kde_up_y.append(uy)
            kde_lo_x.append(lx)
            bin_x.append(bx)
            bin_y.append(by)
        return {
            "I_XT_kde_upper": np.array(kde_up_x),
            "I_TY_kde_upper": np.array(kde_up_y),
            "I_XT_kde_lower": np.array(kde_lo_x),
            "I_XT_bin": np.array(bin_x),
            "I_TY_bin": np.array(bin_y),
        }

    return train(
        net,
        trn.X,
        trn.Y,
        X_test=tst.X,
        Y_test=tst.Y,
        X_analysis=X_an,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        loss="ce",
        classification=True,
        n_checkpoints=cfg.n_checkpoints,
        store_activations=False,  # far too large to keep
        measure_fn=measure,
        track_grad_snr=True,
        snr_mode="per_minibatch",  # per-sample grads over 825k params will not fit
        seed=cfg.seed,
        progress=False,
    )


# --------------------------------------------------------------------------- #
# extension: tiny transformer on the same 12-bit input space
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransformerRunConfig:
    """Settings for one tiny-transformer run.

    The input space is deliberately the *same* 4096 twelve-bit patterns used
    throughout, presented as 12 tokens.  Keeping it identical means the 4096
    patterns remain the entire input distribution, so binning-based mutual
    information stays exact and the numbers are directly comparable with the
    MLP results of section 2.
    """

    activation: str = "gelu"
    layernorm: bool = True
    n_layers: int = 2
    d_model: int = 8
    n_heads: int = 2
    d_ff: int = 16
    lr: float = 0.01
    batch_size: int = 256
    n_epochs: int = 4_000
    n_checkpoints: int = 50
    train_fraction: float = 0.8
    seed: int = 0

    @property
    def tag(self) -> str:
        ln = "ln" if self.layernorm else "noln"
        return (
            f"tf_{self.activation}_{ln}_L{self.n_layers}_d{self.d_model}"
            f"_lr{self.lr:g}_e{self.n_epochs}_s{self.seed}"
        )


def run_transformer(cfg: TransformerRunConfig) -> TrainingLog:
    """Train the tiny transformer, logging pooled activations and diagnostics."""
    import torch

    from .transformer import TinyTransformer, TransformerConfig

    tr, te, full = load_tishby(train_fraction=cfg.train_fraction, seed=cfg.seed)
    tcfg = TransformerConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        activation=cfg.activation,
        layernorm=cfg.layernorm,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        n_checkpoints=cfg.n_checkpoints,
        seed=cfg.seed,
    )
    net = TinyTransformer(tcfg)
    X_full = torch.from_numpy(full.X)

    def measure(epoch: int, model, acts: list[np.ndarray]) -> dict:
        return {
            "attn_entropy": model.attention_entropy(X_full),
            "residual_norm": model.residual_norms(X_full),
        }

    return train(
        net,
        tr.X,
        tr.Y,
        X_test=te.X,
        Y_test=te.Y,
        X_analysis=full.X,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        optimizer="adam",
        loss="bce",
        classification=True,
        n_checkpoints=cfg.n_checkpoints,
        store_activations=True,
        measure_fn=measure,
        track_grad_snr=True,
        snr_mode="per_minibatch",
        seed=cfg.seed,
        progress=False,
    )
