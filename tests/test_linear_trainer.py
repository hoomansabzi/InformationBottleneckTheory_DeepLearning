"""Check the NumPy linear trainer against the PyTorch trainer.

The NumPy path exists purely for speed, so it must be numerically
indistinguishable from the reference torch implementation: same gradients, same
per-sample gradient statistics, same trajectory under full-batch descent.

Run with::

    ./.venv/bin/python tests/test_linear_trainer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ibdl.data import make_student_teacher  # noqa: E402
from ibdl.estimators import gaussian  # noqa: E402
from ibdl.linear_np import LinearNetNP, _grad_stats, _grads, train_linear  # noqa: E402
from ibdl.models import DeepLinear  # noqa: E402
from ibdl.train import LOSSES, _per_sample_grad_stats, train  # noqa: E402

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _matched_pair(sizes, seed=0):
    """A torch DeepLinear and a NumPy LinearNetNP with identical weights."""
    tnet = DeepLinear(sizes, bias=False)
    nnet = LinearNetNP(sizes, seed=seed)
    for i, layer in enumerate(tnet.linears):
        nnet.weights[i] = layer.weight.detach().cpu().numpy().astype(np.float64).copy()
    return tnet, nnet


def test_forward_matches() -> None:
    tnet, nnet = _matched_pair([20, 12, 5, 1])
    X = np.random.default_rng(0).normal(size=(37, 20))
    with torch.no_grad():
        t_out, t_acts = tnet.forward_with_acts(torch.tensor(X, dtype=torch.float32))
    n_out, n_acts = nnet.forward(X)
    err = float(np.abs(t_out.numpy() - n_out).max())
    aerr = max(float(np.abs(a.numpy() - b).max()) for a, b in zip(t_acts, n_acts))
    check("forward output matches torch", err < 1e-5, f"max diff {err:.2e}")
    check("forward activations match torch", aerr < 1e-5, f"max diff {aerr:.2e}")


def test_layer_maps_match() -> None:
    tnet, nnet = _matched_pair([30, 15, 7, 2])
    err = max(
        float(np.abs(a - b).max()) for a, b in zip(tnet.layer_maps(), nnet.layer_maps())
    )
    check("cumulative layer maps match torch", err < 1e-6, f"max diff {err:.2e}")


def test_gradients_match() -> None:
    """Mean-batch gradients must agree with torch autograd on the same loss."""
    tnet, nnet = _matched_pair([16, 8, 3])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(23, 16))
    Y = rng.normal(size=(23, 3))

    tX = torch.tensor(X, dtype=torch.float32)
    tY = torch.tensor(Y, dtype=torch.float32)
    tnet.zero_grad()
    LOSSES["mse"](tnet(tX), tY).backward()
    t_grads = [layer.weight.grad.detach().numpy() for layer in tnet.linears]
    n_grads = _grads(nnet, X, Y)

    errs = [float(np.abs(a - b).max()) for a, b in zip(t_grads, n_grads)]
    check("gradients match torch autograd", max(errs) < 1e-5,
          f"per-layer max diff {['%.1e' % e for e in errs]}")


def test_grad_stats_match() -> None:
    """Per-sample gradient mean/std norms (eqs I.1-I.2) must agree."""
    tnet, nnet = _matched_pair([12, 6, 2])
    rng = np.random.default_rng(2)
    X = rng.normal(size=(64, 12))
    Y = rng.normal(size=(64, 2))

    m_t, s_t = _per_sample_grad_stats(
        tnet, torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32),
        LOSSES["mse"],
    )
    m_n, s_n = _grad_stats(nnet, X, Y)
    me = max(abs(a - b) / max(1e-12, abs(b)) for a, b in zip(m_t, m_n))
    se = max(abs(a - b) / max(1e-12, abs(b)) for a, b in zip(s_t, s_n))
    check("per-sample grad MEAN norms match torch", me < 1e-4, f"max rel diff {me:.2e}")
    check("per-sample grad STD norms match torch", se < 1e-4, f"max rel diff {se:.2e}")


def test_bgd_trajectory_matches() -> None:
    """Full-batch descent must follow the same trajectory in both trainers."""
    p = make_student_teacher(n_inputs=30, n_train=60, n_test=500, snr=1.0, seed=3)
    tnet, nnet = _matched_pair([30, 20, 1])

    ck = [0, 1, 5, 20, 100, 400]
    tlog = train(tnet, p.X_train, p.Y_train, X_test=p.X_test, Y_test=p.Y_test,
                 n_epochs=400, batch_size=None, lr=0.05, loss="mse",
                 classification=False, checkpoints=ck, store_activations=False,
                 store_layer_maps=True, track_grad_snr=False, progress=False)
    nlog = train_linear(nnet, p.X_train, p.Y_train, X_test=p.X_test, Y_test=p.Y_test,
                        n_epochs=400, batch_size=None, lr=0.05, checkpoints=ck,
                        track_grad_snr=False)

    lerr = float(np.abs(tlog.train_loss - nlog.train_loss).max())
    terr = float(np.abs(tlog.test_loss - nlog.test_loss).max())
    werr = float(np.abs(tlog.weight_norms - nlog.weight_norms).max())
    check("BGD train-loss trajectory matches torch", lerr < 2e-4, f"max diff {lerr:.2e}")
    check("BGD test-loss trajectory matches torch", terr < 2e-4, f"max diff {terr:.2e}")
    check("BGD weight-norm trajectory matches torch", werr < 2e-4, f"max diff {werr:.2e}")


def test_learns_the_teacher() -> None:
    """With plenty of data the student should recover the teacher's map."""
    p = make_student_teacher(n_inputs=20, n_train=4000, n_test=2000, snr=100.0, seed=4)
    net = LinearNetNP([20, 20, 1], seed=0)
    log = train_linear(net, p.X_train, p.Y_train, X_test=p.X_test, Y_test=p.Y_test,
                       n_epochs=4000, batch_size=None, lr=0.5, n_checkpoints=20)
    eg = gaussian.generalization_error(net.total_map(), p.W_o, p.sigma_o)
    floor = p.sigma_o**2
    check("student recovers teacher (E_g near noise floor)", eg < 1.15 * floor,
          f"E_g={eg:.4f} vs floor {floor:.4f}")
    check("test loss decreased", log.test_loss[-1] < 0.6 * log.test_loss[0],
          f"{log.test_loss[0]:.3f} -> {log.test_loss[-1]:.3f}")


def test_speedup() -> None:
    """The whole point of this module: it must actually be much faster."""
    p = make_student_teacher(n_inputs=1, n_outputs=1, n_train=100, n_test=200,
                             snr=1.0, seed=5)
    n_epochs = 300

    tnet, nnet = _matched_pair([1, 1, 1])
    t0 = time.time()
    train(tnet, p.X_train, p.Y_train, n_epochs=n_epochs, batch_size=1, lr=0.001,
          loss="mse", classification=False, checkpoints=[n_epochs],
          store_activations=False, track_grad_snr=False, progress=False)
    t_torch = time.time() - t0

    t0 = time.time()
    train_linear(nnet, p.X_train, p.Y_train, n_epochs=n_epochs, batch_size=1,
                 lr=0.001, checkpoints=[n_epochs], track_grad_snr=False)
    t_numpy = time.time() - t0

    check("numpy trainer is faster than torch", t_numpy < t_torch,
          f"torch {t_torch:.2f}s vs numpy {t_numpy:.2f}s "
          f"({t_torch / max(t_numpy, 1e-9):.1f}x speedup)")


def test_snr_phase_transition() -> None:
    """A small-init 1-1-1 net must show the drift -> diffusion SNR drop (fig I2).

    Also checks the accompanying claim of appendix D: the weight norm *grows*
    over training, so in this one-hidden-unit network no compression is possible
    even though the SNR transition is plainly there.
    """
    p = make_student_teacher(n_inputs=1, n_outputs=1, n_train=100, n_test=2000,
                             snr=10.0, seed=6)
    net = LinearNetNP([1, 1, 1], init_std=0.05, seed=0)
    log = train_linear(net, p.X_train, p.Y_train, X_test=p.X_test, Y_test=p.Y_test,
                       n_epochs=20_000, batch_size=1, lr=0.001, n_checkpoints=60)
    snr = log.grad_snr[:, 0]
    drop = float(np.nanmax(snr) / max(np.nanmin(snr), 1e-12))
    check("gradient SNR shows a phase transition", drop > 20,
          f"max {np.nanmax(snr):.3f} -> min {np.nanmin(snr):.4f} ({drop:.0f}x drop)")
    w0, w1 = float(log.weight_norms[0, 0]), float(log.weight_norms[-1, 0])
    check("weight norm grows over training (appendix D)", w1 > w0,
          f"||W1||: {w0:.3f} -> {w1:.3f}")
    check("network actually learns", log.train_loss[-1] < 0.5 * log.train_loss[0],
          f"train MSE {log.train_loss[0]:.3f} -> {log.train_loss[-1]:.3f}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} linear-trainer checks\n" + "=" * 70)
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
