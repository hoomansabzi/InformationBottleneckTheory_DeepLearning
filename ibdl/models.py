"""Network architectures used in the paper.

Everything is a plain ``torch.nn.Module`` that exposes a ``forward_with_acts``
method returning the list of *post-activation* hidden representations.  Those
representations are the variable :math:`h` of the paper; the analysis variable
:math:`T` is obtained from :math:`h` only afterwards, by binning or by adding
noise (see :mod:`ibdl.estimators`).  Keeping the two separate is the whole
point of the paper, so the code keeps them separate too.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "get_activation",
    "MLP",
    "tishby_net",
    "mnist_net",
    "DeepLinear",
]


# --------------------------------------------------------------------------- #
# activation functions
# --------------------------------------------------------------------------- #
def _softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x.abs())


#: The activation functions compared in the paper.  The distinction that
#: matters is *double-saturating* (tanh, softsign, sigmoid: bounded above and
#: below) versus *single-saturating or unbounded* (relu, softplus, linear).
#: The paper's central empirical claim is that only the former compress.
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "softsign": _softsign,
    "softplus": nn.functional.softplus,
    "sigmoid": torch.sigmoid,
    "linear": lambda x: x,
    "gelu": nn.functional.gelu,
    "silu": nn.functional.silu,
    "softmax": lambda x: torch.softmax(x, dim=-1),
}

#: Whether each activation saturates on *both* sides, and its output range.
#: ``bounds`` is used to pick default binning limits (paper section 2: 30 bins
#: evenly spaced between -1 and 1 for tanh).
ACTIVATION_INFO: dict[str, dict] = {
    "tanh": {"double_saturating": True, "bounds": (-1.0, 1.0)},
    "softsign": {"double_saturating": True, "bounds": (-1.0, 1.0)},
    "sigmoid": {"double_saturating": True, "bounds": (0.0, 1.0)},
    "relu": {"double_saturating": False, "bounds": None},
    "softplus": {"double_saturating": False, "bounds": None},
    "linear": {"double_saturating": False, "bounds": None},
    "gelu": {"double_saturating": False, "bounds": None},
    "silu": {"double_saturating": False, "bounds": None},
    "softmax": {"double_saturating": True, "bounds": (0.0, 1.0)},
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        return ACTIVATIONS[name]
    except KeyError:  # pragma: no cover - user error path
        raise ValueError(
            f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}"
        ) from None


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #
def _truncated_normal_(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Truncated normal at two standard deviations.

    This mirrors ``keras.initializers.truncated_normal``, which is what the
    authors' Keras code uses.  Small initial weights matter for the argument of
    appendix D: a tanh network *starts* in its linear regime and can only
    represent a nonlinear function by growing its weights into saturation --
    which is precisely what produces the apparent compression.
    """
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


def _resolve_std(init_std: float | str | None, fan_in: int) -> float:
    """Resolve an ``init_std`` specification to a number.

    ``"fan_in"`` (the default) gives :math:`1/\\sqrt{\\mathrm{fan\\_in}}`, the
    scaling used by the IDNNs reference implementation.  A fixed value applied
    to every layer makes activations decay geometrically through the narrowing
    12-10-7-5-4-3-2 stack, which would put the deep layers at essentially zero
    activity before training even starts.
    """
    if init_std is None or init_std == "fan_in":
        return 1.0 / np.sqrt(fan_in)
    if isinstance(init_std, str):  # pragma: no cover - user error path
        raise ValueError(f"unknown init_std spec {init_std!r}")
    return float(init_std)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Fully connected network with a configurable hidden nonlinearity.

    Parameters
    ----------
    layer_sizes
        Full width specification including input and output, e.g.
        ``[12, 10, 7, 5, 4, 3, 2]`` for the Shwartz-Ziv & Tishby architecture.
    activation
        Name of the hidden nonlinearity (key of :data:`ACTIVATIONS`).
    output_activation
        Nonlinearity of the final layer.  The paper's replication uses
        ``"sigmoid"`` on a 2-unit output layer; note that this sigmoid layer is
        double-saturating and therefore *does* compress even in ReLU networks
        (visible as the only compressing curve in figure 1B).
    init_std
        Standard deviation of the truncated-normal weight init.  ``"fan_in"``
        (default) uses :math:`1/\\sqrt{\\mathrm{fan\\_in}}` per layer.
    bias
        Whether to use biases.

    Notes
    -----
    ``forward_with_acts`` returns the hidden activations of *every* layer
    including the output layer, matching the six curves per information plane
    in figure 1.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        output_activation: str = "sigmoid",
        init_std: float | str | None = "fan_in",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and an output")
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.output_activation_name = output_activation
        self._act = get_activation(activation)
        self._out_act = get_activation(output_activation)

        self.linears = nn.ModuleList(
            nn.Linear(a, b, bias=bias)
            for a, b in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        for layer in self.linears:
            _truncated_normal_(layer.weight, _resolve_std(init_std, layer.in_features))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @property
    def n_layers(self) -> int:
        """Number of weight layers (== number of logged representations)."""
        return len(self.linears)

    def forward_with_acts(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(logits_pre_activation_of_last_layer, activations)``.

        ``activations[i]`` is the post-nonlinearity activity :math:`h_i` of
        layer ``i``.  The last entry is the output-layer activity.
        """
        acts: list[torch.Tensor] = []
        h = x
        last = len(self.linears) - 1
        pre_out = None
        for i, layer in enumerate(self.linears):
            z = layer(h)
            if i == last:
                pre_out = z
                h = self._out_act(z)
            else:
                h = self._act(z)
            acts.append(h)
        return pre_out, acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-activation output (logits) for use with a loss."""
        pre_out, _ = self.forward_with_acts(x)
        return pre_out

    def layer_names(self) -> list[str]:
        names = [f"L{i + 1} ({n})" for i, n in enumerate(self.layer_sizes[1:])]
        names[-1] += " out"
        return names

    def weight_norms(self) -> list[float]:
        """Frobenius norm of each weight matrix (appendix D / figure I1)."""
        return [float(layer.weight.detach().norm()) for layer in self.linears]


def tishby_net(activation: str = "tanh", seed: int | None = None, **kw) -> MLP:
    """The 12-10-7-5-4-3-2 network of Shwartz-Ziv & Tishby (2017).

    Seven fully connected layers; the final layer holds two sigmoidal units.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return MLP([12, 10, 7, 5, 4, 3, 2], activation=activation, **kw)


def mnist_net(activation: str = "tanh", seed: int | None = None, **kw) -> MLP:
    """The 784-1024-20-20-20-10 network of paper figure 1C/D.

    The three narrow layers keep the kernel density estimator well conditioned:
    KDE accuracy degrades quickly with dimension, and the estimate is formed
    from only 10 000 test points.
    """
    if seed is not None:
        torch.manual_seed(seed)
    kw.setdefault("output_activation", "softmax")
    return MLP([784, 1024, 20, 20, 20, 10], activation=activation, **kw)


# --------------------------------------------------------------------------- #
# deep linear networks
# --------------------------------------------------------------------------- #
class DeepLinear(nn.Module):
    """Deep *linear* network :math:`\\hat{Y} = W_{D+1} W_D \\cdots W_1 X`.

    Linear activations stop the network computing nonlinear functions, but the
    optimisation problem stays non-convex and the learning dynamics stay
    genuinely nonlinear (Saxe, McClelland & Ganguli 2014).  The payoff is that
    every hidden layer is jointly Gaussian with the input, so the mutual
    information can be computed *exactly* under the stated noise assumption --
    no binning, no estimator error.  Any compression seen here would therefore
    be real rather than an artefact of estimation.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        init_std: float | str | None = "fan_in",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.linears = nn.ModuleList(
            nn.Linear(a, b, bias=bias)
            for a, b in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        for layer in self.linears:
            _truncated_normal_(layer.weight, _resolve_std(init_std, layer.in_features))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @property
    def n_layers(self) -> int:
        return len(self.linears)

    def forward_with_acts(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        acts: list[torch.Tensor] = []
        h = x
        for layer in self.linears:
            h = layer(h)
            acts.append(h)
        return h, acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_acts(x)
        return out

    # -- exact linear maps, needed for the closed-form mutual information ---- #
    def layer_maps(self) -> list[np.ndarray]:
        """Cumulative maps :math:`\\bar{W}_l = W_l \\cdots W_1` for each layer.

        These are the :math:`\\bar{W}` of paper equation (6).
        """
        maps: list[np.ndarray] = []
        W = None
        for layer in self.linears:
            Wl = layer.weight.detach().cpu().numpy().astype(np.float64)
            W = Wl if W is None else Wl @ W
            maps.append(W.copy())
        return maps

    def total_map(self) -> np.ndarray:
        """:math:`W_{tot} = W_{D+1} \\cdots W_1`, the end-to-end linear map."""
        return self.layer_maps()[-1]

    def weight_norms(self) -> list[float]:
        return [float(layer.weight.detach().norm()) for layer in self.linears]

    def layer_names(self) -> list[str]:
        return [f"L{i + 1} ({n})" for i, n in enumerate(self.layer_sizes[1:])]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def iter_named_weights(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, p in module.named_parameters():
        if name.endswith("weight"):
            yield name, p
