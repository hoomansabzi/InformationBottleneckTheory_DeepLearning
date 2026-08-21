"""Populate the results cache by running every training experiment in parallel.

Notebooks call ``ibdl.cache.cached(tag, ...)``, so once this script has run they
execute in seconds.  Running a notebook without a populated cache still works --
it just trains the networks itself, serially.

Usage::

    ./.venv/bin/python tools/precompute.py            # everything not yet cached
    ./.venv/bin/python tools/precompute.py tishby     # one group
    ./.venv/bin/python tools/precompute.py --force    # recompute
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ibdl.cache import cache_path, cached, parallel_map  # noqa: E402
from ibdl.experiments import (  # noqa: E402
    LinearConfig,
    MnistConfig,
    RelevanceConfig,
    TishbyConfig,
    TransformerRunConfig,
    run_linear,
    run_mnist,
    run_relevance,
    run_tishby,
    run_transformer,
)

# --------------------------------------------------------------------------- #
# experiment groups
# --------------------------------------------------------------------------- #
ACTIVATIONS = ["tanh", "relu", "softsign", "softplus"]
N_REPEATS = 5  # the paper averages 50; 5 is enough to show the effect is robust


def tishby_configs() -> list[TishbyConfig]:
    cfgs: list[TishbyConfig] = []
    # Figures 1A/1B and I1: headline runs, with gradient SNR tracked.
    for act in ["tanh", "relu"]:
        cfgs.append(TishbyConfig(activation=act, seed=0, track_grad_snr=True))
    # Appendix B1: four activations x repeats, for the KDE / Kraskov panels.
    for act in ACTIVATIONS:
        for seed in range(N_REPEATS):
            if act in ("tanh", "relu") and seed == 0:
                continue  # already covered by the headline run
            cfgs.append(TishbyConfig(activation=act, seed=seed))
    # Figure 5: full-batch gradient descent, to remove SGD's stochasticity.
    for act in ["tanh", "relu"]:
        cfgs.append(
            TishbyConfig(activation=act, batch_size=None, lr=0.5, seed=0,
                         track_grad_snr=True)
        )
    # Figures 4C/4D: modest overfitting on 30% of the data.
    for act in ["tanh", "relu"]:
        cfgs.append(TishbyConfig(activation=act, train_fraction=0.3, seed=0))
    return cfgs


def linear_configs() -> list[LinearConfig]:
    # Figures 3 and 4 use the SAME architecture and dataset size (N_i = 100,
    # P = 100, one hidden layer of 100); what separates "generalises well" from
    # "overfits substantially" is the optimiser and how long it runs.  Figure 3
    # is batch gradient descent over 500 epochs (its colour bar tops out at
    # 499); figure 4 is SGD with 5 samples per batch, i.e. 20 updates per epoch,
    # which drives the P = N_i interpolation blow-up that Advani & Saxe (2017)
    # predict.
    return [
        # Figure 3: BGD, 500 epochs.  Generalises well, no compression.
        LinearConfig(hidden=(100,), n_train=100, batch_size=None, lr=0.02,
                     n_epochs=500, n_checkpoints=120, seed=0),
        # Figure 4A/B: SGD batch 5 -> substantial overtraining, still no compression.
        LinearConfig(hidden=(100,), n_train=100, batch_size=5, lr=0.02,
                     n_epochs=2_000, n_checkpoints=100, seed=1),
        # Figure H1: the same setting under BGD, for the SGD/BGD comparison.
        LinearConfig(hidden=(100,), n_train=100, batch_size=None, lr=0.02,
                     n_epochs=2_000, n_checkpoints=100, seed=1),
        # Figure F1: five hidden layers of 50 units, same regime as figure 3.
        LinearConfig(hidden=(50, 50, 50, 50, 50), n_train=100, batch_size=None,
                     lr=0.02, n_epochs=500, n_checkpoints=120, seed=0),
        # Long-run companion to figure 3: what BGD does past the paper's window.
        LinearConfig(hidden=(100,), n_train=100, batch_size=None, lr=0.02,
                     n_epochs=20_000, n_checkpoints=120, seed=0),
        # Figure I2 minimal model: 1-1-1, small init, SGD batch 1.
        LinearConfig(hidden=(1,), n_inputs=1, n_train=100, snr=10.0, batch_size=1,
                     lr=0.001, n_epochs=20_000, init_std=0.05, seed=6),
    ]


def relevance_configs() -> list[RelevanceConfig]:
    # The paper does not state P, the teacher SNR or the initialisation scale for
    # figure 6.  The defaults on RelevanceConfig were chosen by sweep (see the
    # notebook) as the setting that reproduces all three panels: the network has
    # to have enough data and enough signal to actually identify the relevant
    # subspace before it can suppress the rest.
    return [RelevanceConfig(seed=s) for s in range(3)]


# --------------------------------------------------------------------------- #
# workers (module level so they can be pickled)
# --------------------------------------------------------------------------- #
def _job_tishby(cfg: TishbyConfig):
    return cached(cfg.tag, lambda: run_tishby(cfg), verbose=False)


def _job_linear(cfg: LinearConfig):
    return cached(cfg.tag, lambda: run_linear(cfg), verbose=False)


def _job_relevance(cfg: RelevanceConfig):
    return cached(cfg.tag, lambda: run_relevance(cfg), verbose=False)


def mnist_configs() -> list[MnistConfig]:
    """tanh versus ReLU on MNIST (figures 1C-D, B2-B3)."""
    return [MnistConfig(activation=a, seed=0) for a in ("tanh", "relu")]


def _job_mnist(cfg: MnistConfig):
    return cached(cfg.tag, lambda: run_mnist(cfg), verbose=False)


def transformer_configs() -> list[TransformerRunConfig]:
    """Feed-forward nonlinearity x LayerNorm, on the same 12-bit input space.

    The hypothesis under test: a transformer's compression should be driven by
    its *bounded* operations (softmax attention, LayerNorm) rather than by its
    feed-forward nonlinearity.  If so, the LayerNorm axis should matter more
    than the activation axis -- the opposite of the paper's MLP finding.
    """
    cfgs = []
    for act in ("gelu", "relu", "tanh"):
        for ln in (True, False):
            cfgs.append(TransformerRunConfig(activation=act, layernorm=ln,
                                             n_epochs=2_000, seed=0))
    return cfgs


def _job_transformer(cfg: TransformerRunConfig):
    return cached(cfg.tag, lambda: run_transformer(cfg), verbose=False)


# --------------------------------------------------------------------------- #
# information planes (estimation, separate from training)
# --------------------------------------------------------------------------- #
def plane_jobs() -> list[dict]:
    """Every (run, estimator) pair the notebooks need.

    Estimating a KDE plane needs a 4096 x 4096 distance matrix per layer per
    checkpoint, so these are worth farming out just as much as the training is.
    """
    jobs: list[dict] = []
    # Binning planes for every cached Tishby run.
    for cfg in tishby_configs():
        jobs.append(dict(cfg=cfg, estimator="binning", scheme="uniform"))
    # KDE planes for the activation-function comparison (figure B1).
    for act in ACTIVATIONS:
        for seed in range(N_REPEATS):
            cfg = TishbyConfig(
                activation=act, seed=seed,
                track_grad_snr=(act in ("tanh", "relu") and seed == 0),
            )
            jobs.append(dict(cfg=cfg, estimator="kde", scheme="uniform"))
    # Alternative binning schemes on the headline tanh / ReLU runs (App. C).
    for act in ["tanh", "relu"]:
        cfg = TishbyConfig(activation=act, seed=0, track_grad_snr=True)
        jobs.append(dict(cfg=cfg, estimator="binning", scheme="exact"))
        if act == "tanh":
            jobs.append(dict(cfg=cfg, estimator="binning", scheme="net_input"))
    return jobs


def _job_plane(job: dict):
    from ibdl.data import load_tishby
    from ibdl.planes import tishby_plane

    _, _, full = load_tishby()
    return tishby_plane(
        job["cfg"], full.y, estimator=job["estimator"], scheme=job["scheme"]
    )


def _plane_tag(job: dict) -> str:
    suffix = f"{job['estimator']}_{job['scheme']}"
    if job["estimator"] == "kde":
        suffix += "_v0.1"
    return f"plane_{job['cfg'].tag}_{suffix}"


def kraskov_jobs() -> list[str]:
    return ["tanh", "relu"]


def _job_kraskov(activation: str):
    from ibdl.planes import kraskov_entropy_curve

    return kraskov_entropy_curve(activation, n_seeds=N_REPEATS)


GROUPS = {
    "tishby": (tishby_configs, _job_tishby),
    "linear": (linear_configs, _job_linear),
    "relevance": (relevance_configs, _job_relevance),
    "mnist": (mnist_configs, _job_mnist),
    "transformer": (transformer_configs, _job_transformer),
    "planes": (plane_jobs, _job_plane),
    "kraskov": (kraskov_jobs, _job_kraskov),
}

#: How to derive the cache tag for each group's config objects.
TAGGERS = {
    "planes": _plane_tag,
    "kraskov": lambda a: f"kraskov_{a}_k2_n{N_REPEATS}",
}


def main(argv: list[str]) -> int:
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("-")] or list(GROUPS)

    for name in names:
        if name not in GROUPS:
            print(f"unknown group {name!r}; choose from {list(GROUPS)}")
            return 1
        make_configs, job = GROUPS[name]
        cfgs = make_configs()
        tag_of = TAGGERS.get(name, lambda c: c.tag)
        if force:
            for c in cfgs:
                cache_path(tag_of(c)).unlink(missing_ok=True)
        todo = [c for c in cfgs if not cache_path(tag_of(c)).exists()]
        print(f"\n=== {name}: {len(cfgs)} configs, {len(todo)} to compute ===")
        if not todo:
            continue
        t0 = time.time()
        parallel_map(job, todo)
        print(f"=== {name} finished in {time.time() - t0:.0f}s ===")

    total = sum(p.stat().st_size for p in (ROOT / "results").glob("*.pkl.gz"))
    n = len(list((ROOT / "results").glob("*.pkl.gz")))
    print(f"\ncache: {n} entries, {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
