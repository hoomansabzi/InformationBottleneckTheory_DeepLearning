# On the Information Bottleneck Theory of Deep Learning — a reproduction

A from-scratch reproduction and critical study of

> A. M. Saxe, Y. Bansal, J. Dapello, M. Advani, A. Kolchinsky, B. D. Tracey and D. D. Cox,
> *On the information bottleneck theory of deep learning*,
> **J. Stat. Mech.** (2019) 124020 (updated version of the ICLR 2018 paper).

Course project for **Information Theory and Inference**.

---

## ► Start here

Two self-contained, presentation-ready notebooks in **[`final/`](final/)** — everything is
implemented and explained inside them, with no dependency on the library below:

| notebook | what it is |
|---|---|
| [`final/1_information_bottleneck.ipynb`](final/1_information_bottleneck.ipynb) | the full reproduction of the paper — all three claims tested |
| [`final/2_transformer_extension.ipynb`](final/2_transformer_extension.ipynb) | the extension beyond the paper — attention and LayerNorm |

Everything else in this repository is the wider research codebase behind them: the `ibdl/`
library, seven exploratory notebooks, extra estimators, MNIST, the tests, and the LaTeX
report and slides.

---

## What the paper argues

Shwartz-Ziv & Tishby (2017) reported that deep network training passes through a **fitting**
phase and then a distinct **compression** phase, visible as leftward motion in the
*information plane* — a plot of $I(T;Y)$ against $I(X;T)$ for each hidden layer. Three
claims were built on it:

1. training has two phases, the second being compression;
2. compression *causes* good generalisation;
3. compression is driven by the diffusion-like noise of SGD.

Saxe et al. show none holds in general. The underlying reason is definitional: for a
**deterministic continuous** network, $H(h \mid X) = -\infty$, so

$$I(h;X) = \infty$$

at every point in training. Every finite number ever plotted comes from an added assumption
(binning, or noise), and the trajectory reports on that assumption as much as on the network.

## Headline results reproduced

| result | number |
|---|---|
| tanh network compression (12-bit data, binning) | **10.66 bits** |
| same network, `tanh` → `relu` | **0.47 bits** (and *better* test accuracy) |
| same tanh network, only the **bin edges** changed | layer 5: **5.59 → 0.32 bits** |
| deep linear networks, **exact** mutual information | **0.00 bits** compression, at any depth |
| two linear configs with identical information planes | $E_g$ differs by **20×** |
| deterministic full-batch GD (no diffusion at all) | **17.9 bits** — *more* than SGD |
| 1–1–1 net where compression is impossible | gradient SNR still falls **16 502×** |
| task-irrelevant subspace | compresses **53%**, starting at **39% of fitting** |

## Beyond the paper

* **The timing of task-irrelevant compression, quantified.** The paper asserts fitting and
  compression are concurrent; we measure it. Compression begins at epoch 85, when fitting is
  39% complete (33% and 7% for other seeds) — interleaved, not a second phase.
* **The Kraskov estimator is invalid for ReLU layers.** ReLU has an **atom at zero** — a
  unit that is off outputs *exactly* 0, and units switch off together — so 32–50% of
  activation vectors are bit-identical (**0%** for tanh). The distribution is mixed
  discrete–continuous, differential entropy is undefined for it, and the $\varepsilon$-guard
  in eq. (B.10) silently takes over. This is checkable, and it checks out: the guard term
  alone, $d f \log_2\varepsilon$, reproduces the measured entropies (−86.4 vs −92.0, −75.0 vs
  −81.9, −80.0 vs −75.0 bits). The estimator is reporting $\varepsilon$, not the data — and
  since the duplicate count moves during training, so does the "entropy", which is the whole
  of the apparent 55–98 bit ReLU compression. The paper's appendix B.3 does not flag this.
* **Two inconsistencies in the paper's own specification** (see `report/report.tex` §3.2):
  $\Sigma_X$ omitted from eqs (5), (6), (G.1)–(G.3) and a missing $\tfrac12$ in (6); and
  figures 3 and 4 described with identical parameters but opposite conclusions.
* **A transformer extension — hypothesis confirmed.** Softmax attention and LayerNorm are
  *bounded* operations the paper never examined. Crossing the feed-forward nonlinearity with
  LayerNorm on/off, the **normalisation axis dominates the activation axis** (+2.56 bits mean
  vs 1.31–2.14 bits of spread): GELU compresses **0.01 bits without LayerNorm and 2.47 with
  it**. Two surprises refine the mechanism — the compression shows up in the *feed-forward*
  activations (LayerNorm caps their input), and removing LayerNorm makes attention saturate
  *harder* while compressing *less*, because the residual norm then explodes from 2.1 to ≈100.
  **What causes apparent compression is not saturation as such, but a representation whose
  scale stops growing while the weights keep growing.**

---

## Layout

```
ibdl/                  the library
  data.py              Tishby 12-bit dataset, MNIST, linear student/teacher
  models.py            MLPs with swappable nonlinearity, deep linear networks
  minimal.py           the three-neuron model of section 2, solved exactly
  estimators/
    binning.py         eqs (1)-(4); uniform / net-input / machine-precision schemes
    kde.py             Kolchinsky-Tracey bounds, eqs (B.1)-(B.6), blocked
    kraskov.py         Kozachenko-Leonenko, eq (B.10), with validity diagnostics
    gaussian.py        exact linear-Gaussian, eqs (6), (G.1)-(G.4), + subspaces
  train.py             instrumented trainer (activations, gradient SNR, weight norms)
  linear_np.py         fast NumPy trainer for linear nets (17x faster, matches torch)
  transformer.py       the tiny transformer used in notebook 07
  experiments.py       every experiment configuration, in one auditable place
  planes.py            cached information-plane estimation
  cache.py             disk cache + parallel execution
  plotting.py          information planes and diagnostics

final/                 the two presentation notebooks (self-contained) + their figures
notebooks/             01-07, one per paper section (executed, with outputs)
tools/                 notebook generators + tools/precompute.py
tests/                 46 validation checks
report/                report.tex -> report.pdf
slides/                slides.tex -> slides.pdf, slides_final.tex -> slides_final.pdf
figures/               every figure, as PDF (for LaTeX) and PNG
reference/             authors' var_u.mat + original estimator code (for cross-checking)

results/               generated on first run, not committed (~1.3 GB)
final/cache/           generated on first run, not committed (~630 MB)
data/                  MNIST, downloaded on first use, not committed
```

## Notebooks

| # | notebook | reproduces |
|---|---|---|
| 01 | `01_minimal_model.ipynb` | Figs. 2, C1 — why tanh appears to compress, solved exactly |
| 02 | `02_tishby_replication.ipynb` | Figs. 1A–B, A1, B1, B5, C2, C3, E1, E2 |
| 03 | `03_linear_networks.ipynb` | Figs. 3, 4A–B, F1, H1, eq. (C.5) — exact MI |
| 04 | `04_sgd_vs_bgd.ipynb` | Figs. 5, 4C–D, I1, I2 — the SGD mechanism |
| 05 | `05_simultaneous_fitting.ipynb` | Fig. 6 — when compression *does* happen |
| 06 | `06_mnist_kde.ipynb` | Figs. 1C–D, B2–B3 — scaling up |
| 07 | `07_transformer.ipynb` | **extension** — attention and LayerNorm |

## Running it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name ibdl --display-name "Python (IB project)"
```

Validate the estimators (fast, no training):

```bash
.venv/bin/python tests/test_estimators.py && .venv/bin/python tests/test_linear_trainer.py
```

Populate the results cache — everything runs in parallel across cores:

```bash
.venv/bin/python tools/precompute.py
```

Then open the notebooks; with a populated cache they execute in seconds. Without one they
still work, they just train the networks themselves.

Rebuild the documents:

```bash
cd report && latexmk -pdf report.tex
```

### What is not in this repository

Three directories are deliberately absent, because they are large and every one of them
regenerates itself:

| absent | size | how to get it back |
|---|---|---|
| `results/` | ~1.3 GB | `.venv/bin/python tools/precompute.py`, or just run the notebooks |
| `final/cache/` | ~630 MB | run the two notebooks in `final/` once |
| `data/` | 12 MB | downloaded automatically on first use by `ibdl.data.load_mnist` |

So a fresh clone will not have a cache, and the notebooks will train the networks the first
time you run them: about 6 minutes for `final/1`, 18 for `final/2`. Every notebook here is
committed **with its outputs**, so you can read all of it — figures, tables and numbers —
without running anything at all.

## Validation

All 46 checks pass. The estimators agree with the **original authors' implementations**
(`reference/simplebinmi.py`, `reference/kde.py`) to relative $10^{-10}$, and with
analytically known values (uniform codes, Gaussian channels, $H(U[0,1]^d) = 0$,
$H(cX) = H(X) + d\log c$) to the precision of each estimator. The NumPy and PyTorch trainers
agree to $10^{-7}$.

## Data

`reference/var_u.mat` is the original 12-bit dataset from
[github.com/artemyk/ibsgd](https://github.com/artemyk/ibsgd/tree/iclr2018), together with
the authors' estimator code, used to cross-check our implementations. MNIST is downloaded on
first use.
