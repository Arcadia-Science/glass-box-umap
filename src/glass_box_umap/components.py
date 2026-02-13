import math

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
            layers.append(nn.PReLU())

            if i < n_hidden_layers - 1:
                layers.append(LayerNormDetached(hidden_size))

            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, n_components, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.flatten(x))
