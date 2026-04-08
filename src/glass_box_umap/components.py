from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import nn


class VmapPReLU(nn.PReLU):
    """PReLU that becomes stateless in eval mode.

    This is used by :method:`GlassBoxUMAP.compute_attributions`, which computes
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


class DeepPReLUNet(nn.Module):
    """A network with PReLU activation and LayerNormDetached."""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        hidden_size: int = 256,
        n_hidden_layers: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        input_size = math.prod(input_dims)
        self.flatten = nn.Flatten()

        layers = []
        for i in range(n_hidden_layers):
            in_dim = input_size if i == 0 else hidden_size

            layers.append(nn.Linear(in_dim, hidden_size, bias=False))
            layers.append(VmapPReLU())

            if i < n_hidden_layers - 1:
                layers.append(LayerNormDetached(hidden_size))

            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, n_components, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.flatten(x))
