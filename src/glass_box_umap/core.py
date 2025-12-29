from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from glass_box_umap.components import LayerNormDetached
from glass_box_umap.parametric_umap.registry import register_encoder

from .parametric_umap import ParametricUMAP

GLASSBOX_ENCODER_NAME = "glassbox_encoder"


@register_encoder(GLASSBOX_ENCODER_NAME)
class DeepPReLUNet(nn.Module):
    """A network with PReLU activation and LayerNormDetached."""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        hidden_size: int = 256,
        n_hidden_layers: int = 5,
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

        layers.append(nn.Linear(hidden_size, n_components, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.flatten(x))


@dataclass
class GlassBoxUMAP(ParametricUMAP):
    """Glass Box UMAP main class."""

    encoder_name: str = field(default=GLASSBOX_ENCODER_NAME, init=False)

    _feature_contributions: NDArray[np.float16] | None = field(init=False, default=None)
    _jacobians: torch.Tensor | None = field(init=False, default=None)

    def compute_attributions(
        self,
        X: torch.Tensor,
        raw_features: NDArray[np.floating],
        projector: NDArray[np.floating] | None = None,
        batch_size: int = 40,
    ) -> None:
        """Computes Jacobian of the learned embedding w.r.t input features.

        Optionally projects back to a raw feature space.

        Args:
            X:
                The input data provided to the UMAP model (e.g. PCA scores). Shape:
                (n_samples, n_input_dims)
            raw_features:
                The original high-dim features (centered/scaled). Shape: (n_samples,
                n_raw_features). Used to weight the gradients (Gradient * Input).
            projector:
                Linear mapping from Raw -> Input. (e.g. PCA Loadings). Shape:
                (n_raw_features, n_input_dims). If None, assumes Raw == Input (Identity
                mapping).
        """
        self._fitted_model.eval()
        self._fitted_model.to(self._device)
        encoder = self._fitted_model.encoder

        jacobians_input = self._compute_batch_jacobian(encoder, X, batch_size)
        if projector is not None:
            proj_tensor = torch.tensor(projector, dtype=torch.float32).T
            jacobians_raw = torch.einsum("bij,jk->bik", jacobians_input, proj_tensor)
        else:
            jacobians_raw = jacobians_input

        feature_contributions = (jacobians_raw.numpy() * raw_features[:, np.newaxis, :]).astype(
            np.float16
        )

        self._jacobians = jacobians_input
        self._feature_contributions = feature_contributions

    def _compute_batch_jacobian(
        self,
        module: torch.nn.Module,
        X: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        jacobian_list = []
        for j in range(0, len(X), batch_size):
            batch = X[j : j + batch_size].to(self._device)
            jacobian = torch.autograd.functional.jacobian(
                module,
                batch,
                vectorize=True,
                strategy="reverse-mode",
            )
            jacobian_list.append(torch.einsum("bibj->bij", jacobian).detach().cpu())

        return torch.cat(jacobian_list, dim=0)
