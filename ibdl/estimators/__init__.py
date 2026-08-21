"""Mutual-information estimators used in Saxe et al.

Four routes to a finite number, each resting on a different assumption:

===============  ==================================  ==========================
module           assumption                          paper reference
===============  ==================================  ==========================
:mod:`binning`   discretise :math:`h` into bins      section 2, eqs (1)-(4)
:mod:`kde`       :math:`T = h + \\mathcal{N}(0,\\sigma^2)`  appendix B.1, eqs (B.1)-(B.6)
:mod:`kraskov`   homoscedastic noise, kNN entropy    appendix B.3, eq (B.10)
:mod:`gaussian`  linear net, exact, additive noise   section 3, eqs (6), (G.1)-(G.4)
===============  ==================================  ==========================

The paper's thesis lives in this table.  For a deterministic continuous network
:math:`I(h;X) = \\infty`, so *every* finite information-plane number is a
statement about the assumption in column two, not purely about the network.
Swapping assumptions (or even just the bin edges, appendix C) can turn a
compression phase on and off while the network is untouched.
"""

from . import binning, gaussian, kde, kraskov

__all__ = ["binning", "gaussian", "kde", "kraskov"]
