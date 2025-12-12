from __future__ import annotations

import torch
from torch import nn


class LayerNormDetached(nn.Module):
    """
    A LayerNorm implementation where the variance calculation is detached from the
    computation graph during evaluation, potentially stabilizing training.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNormDetached."""
        mean = x.mean(dim=-1, keepdim=True)
        # Detach variance calculation during evaluation
        if not self.training:
            var = x.clone().detach().var(dim=-1, keepdim=True, unbiased=False)
        else:
            var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + 1e-12)  # Added epsilon for stability
        return self.scale * norm_x


class DeepReLUNet(nn.Module):
    """
    A deep neural network using PReLU activation and LayerNormDetached.
    """

    def __init__(self, input_size: int = 50, hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.PReLU(),
            LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.PReLU(),
            LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.PReLU(),
            LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.PReLU(),
            LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PReLU network."""
        return self.model(x)
