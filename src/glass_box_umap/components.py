from __future__ import annotations

import torch
from torch import nn


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
