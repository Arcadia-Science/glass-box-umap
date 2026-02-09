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
        batch_size: int | None = None,
    ) -> tuple[NDArray[np.float16], torch.Tensor]:
        """Computes Jacobian of the learned embedding w.r.t input features.

        Projects gradients back to raw feature space if PCA preprocessing was used.
        Uses Gradient x Input method with mean-centered features.

        Args:
            X:
                The input data (same format as passed to fit/transform).
                Shape: (n_samples, n_input_dims)
        """
        self._fitted_model.eval()
        self._fitted_model.to(self._device)
        encoder = self._fitted_model.encoder

        if batch_size is None:
            batch_size = self.batch_size

        X_centered = X.detach().cpu().numpy() - self._mean

        if self._pca is not None:
            X_processed = self._pca.transform(X_centered)
        else:
            X_processed = X_centered

        X_encoder = torch.from_numpy(X_processed.astype(np.float32))
        jacobians_input = self._compute_batch_jacobian(encoder, X_encoder, batch_size)

        if self._pca is not None:
            proj_tensor = torch.tensor(self._pca.components_, dtype=torch.float32)
            jacobians_raw = torch.einsum("bij,jk->bik", jacobians_input, proj_tensor)
        else:
            jacobians_raw = jacobians_input

        feature_contributions = (jacobians_raw.numpy() * X_centered[:, np.newaxis, :]).astype(
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
