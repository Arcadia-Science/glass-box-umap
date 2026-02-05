from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from glass_box_umap.components import DeepPReLUNet
from glass_box_umap.parametric_umap.registry import register_encoder

from .parametric_umap import ParametricUMAP

GLASSBOX_ENCODER_NAME = "glassbox_encoder"
register_encoder(GLASSBOX_ENCODER_NAME)(DeepPReLUNet)


@dataclass
class GlassBoxUMAP(ParametricUMAP):
    """Glass Box UMAP main class."""

    encoder_name: str = field(default=GLASSBOX_ENCODER_NAME, init=False)

    def compute_attributions(
        self,
        X: torch.Tensor,
        raw_features: NDArray[np.floating],
        projector: NDArray[np.floating] | None = None,
        batch_size: int | None = None,
    ) -> tuple[NDArray[np.float16], torch.Tensor]:
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

        if batch_size is None:
            batch_size = self.batch_size

        jacobians_input = self._compute_batch_jacobian(encoder, X, batch_size)
        if projector is not None:
            proj_tensor = torch.tensor(projector, dtype=torch.float32).T
            jacobians_raw = torch.einsum("bij,jk->bik", jacobians_input, proj_tensor)
        else:
            jacobians_raw = jacobians_input

        feature_contributions = (jacobians_raw.numpy() * raw_features[:, np.newaxis, :]).astype(
            np.float16
        )

        return feature_contributions, jacobians_input

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
