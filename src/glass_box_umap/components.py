from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VmapPReLU(nn.PReLU):
    """PReLU that becomes stateless in eval mode.

    This is used by :method:`GlassBoxUMAP.compute_contributions`, which computes
    per-sample Jacobians via ``vmap(jacrev(...))``. Using this subclass in place of
    ``nn.PReLU`` makes this possible. ``vmap`` requires all operations to be stateless,
    but ``nn.PReLU`` reads from a learnable ``weight`` parameter during forward. In eval
    mode, this subclass caches the learned slope as a plain float and delegates to
    ``F.leaky_relu``, which is a pure function of its input.

    The output is identical to ``nn.PReLU`` in both training and eval modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_slope: float | None = None

    def train(self, mode: bool = True) -> VmapPReLU:
        if mode:
            self._cached_slope = None
        return super().train(mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.prelu(input, self.weight)

        if self._cached_slope is None:
            self._cached_slope = self.weight.item()

        return F.leaky_relu(input, self._cached_slope)


class LayerNormDetached(nn.Module):
    """A LayerNorm with detached variance calculation during evaluation."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)

        # Detach variance calculation during evaluation
        if not self.training:
            var = x.clone().detach().var(dim=-1, keepdim=True, unbiased=False)
        else:
            var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + 1e-12)  # Added epsilon for stability
        return self.scale * norm_x
