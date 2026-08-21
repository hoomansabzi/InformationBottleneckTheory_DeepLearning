"""A tiny transformer, analysed with the paper's own machinery.

Motivation
----------
Saxe et al. show that a compression phase appears exactly when a network's
representation is squeezed through a **double-saturating** map and then binned or
noised.  Their evidence comes from tanh, sigmoid and softsign versus ReLU,
softplus and linear.  Modern architectures are built from ReLU-family
nonlinearities, so a naive reading predicts they never compress.

But a transformer block contains two operations the paper never examined, both of
which are *bounded*:

* **softmax attention** -- each output is a convex combination of value vectors,
  so it can never leave their convex hull, and the attention weights themselves
  saturate towards one-hot as the logits grow;
* **LayerNorm** -- projects the residual stream onto a sphere of fixed radius
  :math:`\\sqrt{d}`, so however large the weights grow the representation cannot
  spread out.

Both are saturating in the relevant sense.  The paper's own mechanism therefore
predicts that a transformer should show an apparent compression phase *driven by
its normalisation and attention*, not by its feed-forward nonlinearity -- and
that removing LayerNorm should remove it.  That is the hypothesis this module
exists to test.

Design
------
The input is the **same 12-bit / 4096-pattern dataset** used throughout, but fed
as a sequence of 12 tokens.  Keeping the input space identical matters: the 4096
patterns are the entire input distribution, so binning-based mutual information
stays exact and every number is directly comparable with notebooks 2 and 4.

The model is deliberately narrow (``d_model = 8``) and the tracked
representations are *pooled* over positions.  A full ``12 x d_model`` sequence
representation would be so high-dimensional that every one of the 4096 inputs
would land in its own bin and :math:`I(X;T)` would sit pinned at
:math:`\\log_2 4096 = 12` bits with no dynamics -- the machine-precision failure
mode of appendix C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .models import get_activation

__all__ = ["TinyTransformer", "TransformerConfig", "make_token_inputs"]


def make_token_inputs(X: np.ndarray) -> np.ndarray:
    """Turn the 12-bit patterns into integer token ids of shape ``(P, 12)``."""
    return np.asarray(X, dtype=np.int64)


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture and training settings for the tiny transformer."""

    n_layers: int = 2
    d_model: int = 8
    n_heads: int = 2
    d_ff: int = 16
    activation: str = "gelu"  # feed-forward nonlinearity
    layernorm: bool = True
    head_hidden: tuple[int, ...] = (4,)  # narrowing head, as in the Tishby net
    n_tokens: int = 12
    vocab: int = 2
    lr: float = 0.01
    batch_size: int = 256
    n_epochs: int = 4_000
    n_checkpoints: int = 50
    seed: int = 0

    @property
    def tag(self) -> str:
        ln = "ln" if self.layernorm else "noln"
        return (
            f"tf_{self.activation}_{ln}_L{self.n_layers}_d{self.d_model}"
            f"_h{self.n_heads}_lr{self.lr:g}_e{self.n_epochs}_s{self.seed}"
        )


class _SelfAttention(nn.Module):
    """Multi-head self-attention, returning the attention weights as well.

    The weights are exposed because their entropy is one of the quantities we
    want to watch: as attention sharpens towards one-hot, it saturates in
    exactly the sense that drives compression in the paper's tanh networks.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, self.d_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)  # bounded: rows lie on the simplex
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out), attn


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d) if cfg.layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if cfg.layernorm else nn.Identity()
        self.attn = _SelfAttention(d, cfg.n_heads)
        self.ff1 = nn.Linear(d, cfg.d_ff)
        self.ff2 = nn.Linear(cfg.d_ff, d)
        self._act = get_activation(cfg.activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        a, attn = self.attn(self.ln1(x))
        x = x + a
        h_ff = self._act(self.ff1(self.ln2(x)))
        x = x + self.ff2(h_ff)
        return x, {"attn": attn, "attn_out": a, "ff_hidden": h_ff, "residual": x}


class TinyTransformer(nn.Module):
    """A small encoder-only transformer over 12 binary tokens.

    ``forward_with_acts`` returns the representations tracked in the information
    plane.  All sequence-valued representations are **mean-pooled over
    positions** so that their dimensionality stays low enough for the binning
    and KDE estimators to say something non-trivial (see the module docstring).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        d = cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab, d)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.n_tokens, d))
        nn.init.normal_(self.tok_emb.weight, std=0.5)
        nn.init.normal_(self.pos_emb, std=0.5)

        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d) if cfg.layernorm else nn.Identity()

        sizes = [d, *cfg.head_hidden]
        self.head_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])
        )
        self.out = nn.Linear(sizes[-1], 2)
        self._head_act = get_activation(cfg.activation)

    # -- names of the tracked representations ------------------------------- #
    def layer_names(self) -> list[str]:
        names = ["embed"]
        for i in range(self.cfg.n_layers):
            names += [f"B{i + 1} attn", f"B{i + 1} ff", f"B{i + 1} resid"]
        names += ["pooled"]
        names += [f"head{i + 1}" for i in range(len(self.head_layers))]
        names += ["out"]
        return names

    def forward_with_acts(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = x.long()
        h = self.tok_emb(tokens) + self.pos_emb  # (B, T, d)

        acts: list[torch.Tensor] = [h.mean(dim=1)]
        for block in self.blocks:
            h, info = block(h)
            acts.append(info["attn_out"].mean(dim=1))
            acts.append(info["ff_hidden"].mean(dim=1))
            acts.append(info["residual"].mean(dim=1))

        pooled = self.ln_f(h).mean(dim=1)
        acts.append(pooled)

        z = pooled
        for layer in self.head_layers:
            z = self._head_act(layer(z))
            acts.append(z)

        logits = self.out(z)
        acts.append(torch.sigmoid(logits))
        return logits, acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_acts(x)
        return logits

    # -- diagnostics --------------------------------------------------------- #
    def weight_norms(self) -> list[float]:
        return [
            float(p.detach().norm())
            for n, p in self.named_parameters()
            if n.endswith("weight") and p.dim() >= 2
        ]

    @torch.no_grad()
    def attention_entropy(self, x: torch.Tensor) -> np.ndarray:
        """Mean entropy (bits) of the attention distributions, per block.

        A uniform distribution over 12 positions has entropy
        :math:`\\log_2 12 \\approx 3.58` bits; a fully sharpened, one-hot
        attention pattern has 0.  Falling entropy means attention is saturating,
        which is the transformer's analogue of a tanh unit entering its flat
        region.
        """
        h = self.tok_emb(x.long()) + self.pos_emb
        out = []
        for block in self.blocks:
            h, info = block(h)
            a = info["attn"].clamp_min(1e-12)
            ent = -(a * a.log2()).sum(-1)  # (B, heads, T)
            out.append(float(ent.mean()))
        return np.array(out)

    @torch.no_grad()
    def residual_norms(self, x: torch.Tensor) -> np.ndarray:
        """Mean L2 norm of the residual stream after each block.

        With LayerNorm the *normalised* stream is pinned to radius
        :math:`\\sqrt{d}`; without it the stream is free to grow, which is what
        lets an unnormalised network keep increasing its entropy.
        """
        h = self.tok_emb(x.long()) + self.pos_emb
        out = []
        for block in self.blocks:
            h, _ = block(h)
            out.append(float(h.norm(dim=-1).mean()))
        return np.array(out)
