"""Datasets used in Saxe et al., *On the information bottleneck theory of deep
learning* (ICLR 2018 / J. Stat. Mech. 2019).

Three families of data are needed to reproduce the paper:

1. ``load_tishby``  -- the synthetic 12-bit dataset of Shwartz-Ziv & Tishby
   (2017).  12 binary inputs give :math:`2^{12} = 4096` distinct patterns, each
   carrying a binary label produced by a spherically symmetric rule.  Because
   the input alphabet is *finite and fully enumerated*, the empirical
   distribution of :math:`X` is the true distribution and binning-based mutual
   information is exact (no sampling error).  This is what makes the dataset
   attractive for information-plane experiments in the first place.

2. ``make_student_teacher`` -- the linear student/teacher construction of
   section 3.  Gaussian inputs are pushed through a random linear teacher, so
   the generalisation error and *all* mutual informations are available in
   closed form.

3. ``make_relevant_irrelevant`` -- the section 5 variant in which the teacher
   ignores part of the input, creating an explicitly task-irrelevant subspace.

Plus ``load_mnist`` for the larger-scale check of section 2 / appendix B.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = [
    "Dataset",
    "load_tishby",
    "load_mnist",
    "make_student_teacher",
    "make_relevant_irrelevant",
    "TeacherProblem",
]

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PKG_ROOT / "data"
_REFERENCE_DIR = _PKG_ROOT / "reference"


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """A classification dataset.

    Attributes
    ----------
    X : (P, d) float32
        Inputs.
    y : (P,) int64
        Integer class labels.
    Y : (P, n_classes) float32
        One-hot labels.
    """

    X: np.ndarray
    y: np.ndarray
    Y: np.ndarray
    n_classes: int

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.X.shape[0]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Dataset(P={len(self)}, d={self.X.shape[1]}, "
            f"n_classes={self.n_classes})"
        )


@dataclass
class TeacherProblem:
    """A linear student/teacher regression problem (paper section 3).

    The teacher generates :math:`Y = W_o X + \\epsilon_o`, with
    :math:`X \\sim \\mathcal{N}(0, I_{N_i}/N_i)`,
    :math:`W_o \\sim \\mathcal{N}(0, \\sigma_w^2)` and
    :math:`\\epsilon_o \\sim \\mathcal{N}(0, \\sigma_o^2)`.

    ``eps_o`` models *approximation error*: structure in the target that no
    network of this class can represent.  The paper is emphatic that this is a
    different object from the noise later added for the purpose of *computing*
    mutual information -- the former is part of the data, the latter part of
    the analysis.
    """

    X_train: np.ndarray  # (P, Ni)
    Y_train: np.ndarray  # (P, No)
    X_test: np.ndarray  # (P_test, Ni)
    Y_test: np.ndarray  # (P_test, No)
    W_o: np.ndarray  # (No, Ni) teacher weights
    sigma_w: float
    sigma_o: float
    n_relevant: int | None = None  # section 5: first n_relevant dims carry signal

    @property
    def snr(self) -> float:
        """Teacher signal-to-noise ratio :math:`\\sigma_w^2/\\sigma_o^2`."""
        return self.sigma_w**2 / self.sigma_o**2

    @property
    def n_inputs(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.Y_train.shape[1]


# --------------------------------------------------------------------------- #
# 1. Shwartz-Ziv & Tishby synthetic dataset
# --------------------------------------------------------------------------- #
def load_tishby(
    train_fraction: float = 0.8,
    seed: int = 0,
    path: str | os.PathLike | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load the 12-bit / 4096-pattern dataset of Shwartz-Ziv & Tishby (2017).

    The file ``var_u.mat`` holds ``F`` of shape (4096, 12) -- every 12-bit
    pattern exactly once -- and ``y`` of shape (1, 4096), a binary label with
    :math:`P(y=1) \\approx 0.517`.

    Parameters
    ----------
    train_fraction
        Fraction used for training.  The paper's replication uses 80%.
    seed
        Seed for the train/test shuffle.
    path
        Location of ``var_u.mat``; defaults to ``reference/var_u.mat``.

    Returns
    -------
    (train, test, full)
        ``full`` holds all 4096 patterns in their canonical order and is what
        the information-plane computations run on: the 4096 patterns *are* the
        input distribution, taken uniform, so :math:`H(X) = \\log_2 4096 = 12`
        bits exactly.
    """
    path = Path(path) if path is not None else _REFERENCE_DIR / "var_u.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it from "
            "https://github.com/artemyk/ibsgd/blob/iclr2018/datasets/var_u.mat"
        )

    mat = sio.loadmat(str(path))
    X = mat["F"].astype(np.float32)  # (4096, 12), entries in {0, 1}
    y = np.squeeze(mat["y"]).astype(np.int64)  # (4096,)

    n_classes = 2
    Y = np.eye(n_classes, dtype=np.float32)[y]
    full = Dataset(X=X, y=y, Y=Y, n_classes=n_classes)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_train = int(np.rint(train_fraction * len(X)))
    tr, te = perm[:n_train], perm[n_train:]

    train = Dataset(X=X[tr], y=y[tr], Y=Y[tr], n_classes=n_classes)
    test = Dataset(X=X[te], y=y[te], Y=Y[te], n_classes=n_classes)
    return train, test, full


# --------------------------------------------------------------------------- #
# 2. MNIST
# --------------------------------------------------------------------------- #
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)


def _download_mnist(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for fname in _MNIST_FILES.values():
        target = dest / fname
        if target.exists():
            continue
        last_err: Exception | None = None
        for mirror in _MNIST_MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + fname, target)
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
        else:  # pragma: no cover - network dependent
            raise RuntimeError(f"could not download {fname}") from last_err


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "big")
        ndim = magic & 0xFF
        shape = [int.from_bytes(fh.read(4), "big") for _ in range(ndim)]
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(shape)


def load_mnist(root: str | os.PathLike | None = None) -> tuple[Dataset, Dataset]:
    """Load MNIST, scaled to :math:`[0, 1]` as in the authors' ``utils.py``.

    Downloads the four IDX files on first use.
    """
    root = Path(root) if root is not None else _DATA_DIR / "mnist"
    _download_mnist(root)

    Xtr = _read_idx(root / _MNIST_FILES["train_images"]).reshape(-1, 784)
    Xte = _read_idx(root / _MNIST_FILES["test_images"]).reshape(-1, 784)
    ytr = _read_idx(root / _MNIST_FILES["train_labels"]).astype(np.int64)
    yte = _read_idx(root / _MNIST_FILES["test_labels"]).astype(np.int64)

    Xtr = (Xtr.astype(np.float32) / 255.0).copy()
    Xte = (Xte.astype(np.float32) / 255.0).copy()

    eye = np.eye(10, dtype=np.float32)
    return (
        Dataset(X=Xtr, y=ytr, Y=eye[ytr], n_classes=10),
        Dataset(X=Xte, y=yte, Y=eye[yte], n_classes=10),
    )


# --------------------------------------------------------------------------- #
# 3. Linear student / teacher (sections 3 and 5)
# --------------------------------------------------------------------------- #
def make_student_teacher(
    n_inputs: int = 100,
    n_outputs: int = 1,
    n_train: int = 100,
    n_test: int = 10_000,
    snr: float = 1.0,
    sigma_w: float = 1.0,
    seed: int = 0,
) -> TeacherProblem:
    """Generate the linear student/teacher problem of paper section 3.

    Inputs are :math:`X \\sim \\mathcal{N}(0, I_{N_i}/N_i)` so that
    :math:`\\mathbb{E}\\|X\\|^2 = 1` independent of dimension.  The teacher
    weights are i.i.d. :math:`\\mathcal{N}(0, \\sigma_w^2)`, and the output
    noise has variance :math:`\\sigma_o^2 = \\sigma_w^2 / \\mathrm{SNR}`.

    Notes
    -----
    Following Advani & Saxe (2017), overtraining in this model is worst when
    ``n_train == n_inputs`` -- the setting used for figure 4A/B.
    """
    rng = np.random.default_rng(seed)
    sigma_o = float(np.sqrt(sigma_w**2 / snr))

    W_o = rng.normal(0.0, sigma_w, size=(n_outputs, n_inputs))

    def sample(P: int) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(P, n_inputs))
        Y = X @ W_o.T + rng.normal(0.0, sigma_o, size=(P, n_outputs))
        return X.astype(np.float32), Y.astype(np.float32)

    X_train, Y_train = sample(n_train)
    X_test, Y_test = sample(n_test)

    return TeacherProblem(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        W_o=W_o,
        sigma_w=sigma_w,
        sigma_o=sigma_o,
    )


def make_relevant_irrelevant(
    n_relevant: int = 30,
    n_irrelevant: int = 70,
    n_outputs: int = 1,
    n_train: int = 100,
    n_test: int = 10_000,
    snr: float = 1.0,
    sigma_w: float = 1.0,
    seed: int = 0,
) -> TeacherProblem:
    """Section 5 variant: the teacher's weights on ``X_irrel`` are exactly zero.

    The input is partitioned as :math:`X = [X_{\\mathrm{rel}},
    X_{\\mathrm{irrel}}]` with the first ``n_relevant`` coordinates carrying all
    the signal.  A network that generalises *must* eventually suppress
    :math:`X_{\\mathrm{irrel}}`, which lets us ask whether that suppression
    shows up as a separate compression phase (it does not -- it happens
    concurrently with fitting).
    """
    n_inputs = n_relevant + n_irrelevant
    problem = make_student_teacher(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_train=n_train,
        n_test=n_test,
        snr=snr,
        sigma_w=sigma_w,
        seed=seed,
    )
    # Zero the teacher's weights on the irrelevant block and regenerate targets
    # so that X_irrel contributes nothing but nuisance variance.
    W_o = problem.W_o.copy()
    W_o[:, n_relevant:] = 0.0

    rng = np.random.default_rng(seed + 9973)
    sigma_o = problem.sigma_o
    Y_train = problem.X_train @ W_o.T + rng.normal(
        0.0, sigma_o, size=problem.Y_train.shape
    )
    Y_test = problem.X_test @ W_o.T + rng.normal(0.0, sigma_o, size=problem.Y_test.shape)

    return TeacherProblem(
        X_train=problem.X_train,
        Y_train=Y_train.astype(np.float32),
        X_test=problem.X_test,
        Y_test=Y_test.astype(np.float32),
        W_o=W_o,
        sigma_w=problem.sigma_w,
        sigma_o=sigma_o,
        n_relevant=n_relevant,
    )


def _checksum(arr: np.ndarray) -> str:  # pragma: no cover - debugging aid
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()[:12]
