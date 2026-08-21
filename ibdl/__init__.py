"""Reproduction of Saxe et al., *On the information bottleneck theory of deep
learning* (ICLR 2018; J. Stat. Mech. 2019, 124020).

Sub-modules
-----------
``data``        datasets (Tishby 12-bit, MNIST, linear student/teacher)
``models``      MLPs with swappable nonlinearity, deep linear networks
``minimal``     the three-neuron model of section 2, solved exactly
``estimators``  the four mutual-information estimators used in the paper
``train``       training loops with activation / gradient-SNR logging
``linear_np``   fast NumPy trainer for the deep linear experiments
``plotting``    information-plane and diagnostic figures

Sub-modules are imported lazily so that e.g. ``ibdl.data`` works without
pulling in torch.
"""

import importlib
from typing import Any

__version__ = "0.1.0"

_SUBMODULES = (
    "cache",
    "data",
    "estimators",
    "experiments",
    "linear_np",
    "minimal",
    "models",
    "plotting",
    "train",
)

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_SUBMODULES))
