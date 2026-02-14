from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from glass_box_umap.components import DeepPReLUNet
from glass_box_umap.parametric_umap.registry import register_encoder

from .parametric_umap import ParametricUMAP
from .parametric_umap.core import _to_numpy

GLASSBOX_ENCODER_NAME = "glassbox_encoder"
register_encoder(GLASSBOX_ENCODER_NAME)(DeepPReLUNet)


@dataclass
class GlassBoxUMAP(ParametricUMAP):
    """Glass Box UMAP model.

    Attributes:
        n_neighbors: Number of nearest neighbors used to construct the
            high-dimensional graph.
        min_dist: Minimum distance between points in the low-dimensional
            embedding.
        metric: Distance metric used for computing nearest neighbors.
        n_components: Dimensionality of the learned embedding.
        random_state: Random seed for reproducibility. If ``None``, no seed
            is set.
        encoder_kwargs: Additional keyword arguments passed to the encoder
            constructor.
        pca_components: Number of PCA components for input preprocessing.
            If ``None``, no PCA is applied.
        lr: Learning rate for the optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training and (default) inference.
        negative_sample_rate: Number of negative samples per positive edge
            in the UMAP loss.
        repulsion_strength: Weighting of the repulsive term in the UMAP loss.
        num_workers: Number of data loading workers.
        checkpoint_dir: Directory for saving training checkpoints. If ``None``,
            a temporary directory is used.
    """

    encoder_name: str = field(default=GLASSBOX_ENCODER_NAME, init=False)

    def compute_attributions(
        self,
        X: NDArray[np.floating] | torch.Tensor,
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

        assert self._mean is not None
        X_centered = _to_numpy(X) - self._mean

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
