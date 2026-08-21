# Third-party reference material

Nothing in this folder is ours. It is kept so that our own implementations can be
cross-checked against the code and data of the authors we are reproducing, and it is
**not** covered by the project's MIT licence — see `../LICENSE`.

| file | origin | used for |
|---|---|---|
| `var_u.mat` | [github.com/artemyk/ibsgd](https://github.com/artemyk/ibsgd/tree/iclr2018) (`datasets/var_u.mat`) | the 12-bit / 4096-pattern dataset every experiment runs on |
| `simplebinmi.py` | [github.com/ravidziv/IDNNs](https://github.com/ravidziv/IDNNs), via `artemyk/ibsgd` | the original binning estimator, cross-checked in `../tests/test_estimators.py` |
| `kde.py` | [github.com/artemyk/ibsgd](https://github.com/artemyk/ibsgd/tree/iclr2018) | the original Kolchinsky–Tracey KDE bounds, cross-checked likewise |
| `utils.py` | [github.com/artemyk/ibsgd](https://github.com/artemyk/ibsgd/tree/iclr2018) | the authors' data loading, for comparison |

Our estimators in `../ibdl/estimators/` agree with these to relative $10^{-10}$; that
agreement is what `../tests/test_estimators.py` checks. The runtime code does not import
anything from this folder — only the tests do.

`var_u.mat` is the one file the notebooks genuinely need. If you would rather not vendor it,
delete it and fetch it from the link above; `ibdl.data.load_tishby` prints that URL if the
file is missing.
