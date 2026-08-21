# Final version — read these two notebooks

Everything needed for the presentation is here. Both notebooks are **self-contained**: they
import nothing from the `ibdl/` package, and every estimator, model and training loop they
use is written out in the notebook itself, alongside the explanation of what it does and why.

| notebook | what it is |
|---|---|
| [`1_information_bottleneck.ipynb`](1_information_bottleneck.ipynb) | The full reproduction of Saxe et al., *On the information bottleneck theory of deep learning* — all three claims tested and dismantled, section by section. |
| [`2_transformer_extension.ipynb`](2_transformer_extension.ipynb) | The extension beyond the paper: does a **transformer** compress, and is it the activation function or **LayerNorm** that causes it? |

Read notebook 1 first; notebook 2 assumes its estimator and vocabulary.

## Running them

```bash
.venv/bin/jupyter lab
```

Then open either notebook and pick the kernel **"Python (IB project)"**. Both are already
executed, so all figures and numbers are visible without running anything.

From an **empty** cache, notebook 1 takes about 6 minutes (it trains 5 networks) and
notebook 2 about 18 minutes (6 transformers). Every expensive result is cached to
`final/cache/`, so re-running either notebook after that takes seconds. Delete a file in
`final/cache/` to force that one experiment to recompute; delete the whole folder to start
clean (it is ~630 MB, entirely regenerable).

The only external input is `../reference/var_u.mat`, the original 12-bit dataset from the
authors of the paper being replicated.

## What is in this folder

```
1_information_bottleneck.ipynb    the main notebook  (53 cells, 17 figures)
2_transformer_extension.ipynb     the extension      (19 cells,  3 figures)
cache/                            memoised experiment results (~630 MB, regenerable)
figures/                          every figure, as PNG and PDF
```

The notebooks are generated from [`../tools/final1_ib.py`](../tools/final1_ib.py) and
[`../tools/final2_transformer.py`](../tools/final2_transformer.py), which keeps the prose
under normal version control; edit either the notebook or its generator, whichever suits.

## The rest of the project

The wider research codebase is still in place and is where the deeper material lives — the
`ibdl/` library, the seven research notebooks in `notebooks/`, the extra estimators (KDE
bounds, Kraskov $k$-NN), the MNIST experiments, 46 validation tests, and the LaTeX report and
slides. Nothing there is needed to read or present the two notebooks here.

The two sets are cross-linked: notebook 1 ends with a **"Where to go deeper"** map from each
of its sections to the research notebook that expands it, and each of the seven opens with a
banner pointing back here. The one thing the research notebooks cover that these do not is
**MNIST** (`notebooks/06_mnist_kde.ipynb`).

Because these two notebooks are an **independent re-implementation**, their numbers are their
own: the estimators were checked against the library's (they agree to machine precision), but
individual runs differ in checkpoint schedule and RNG, so a figure quoted here can differ in
magnitude from the corresponding one in `notebooks/`. Every qualitative conclusion is the
same, and the headline numbers of the reproduction (tanh 10.66 bits vs ReLU 0.47; layer 5
falling from 5.59 to 0.32 bits when only the bin edges change; 0.00 bits of compression in
the exact linear case) come out identical.
