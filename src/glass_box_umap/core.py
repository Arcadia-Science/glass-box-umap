from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from .jacobian import compute_jacobian, project_jacobian, reduce_contributions
from .parametric_umap import ParametricUMAP
from .parametric_umap.core import _to_numpy_float32


@dataclass(eq=False, kw_only=True)
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
            If ``None``, no PCA is applied. PCA requires 2D input
            ``(n_samples, n_features)``; leave this ``None`` when fitting on
            multi-dimensional data (e.g. images for a convolutional encoder).
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

    def compute_contributions(
        self,
        X: NDArray[np.floating] | torch.Tensor,
        batch_size: int | None = None,
        reduction: Literal["l2"] | None = None,
    ) -> NDArray[np.float32]:
        """Compute per-feature contributions to the embedding via Gradient x Input.

        Projects gradients back to raw feature space if PCA preprocessing was used.

        Args:
            X:
                The input data (same format as passed to fit/transform).
                Shape: (n_samples, n_features).
            batch_size:
                Batch size for Jacobian computation. Defaults to ``self.batch_size``.
            reduction:
                How to reduce contributions across embedding dimensions. If ``"l2"``,
                takes the L2 norm across components, returning shape
                (n_samples, n_features). If ``None``, returns the full
                (n_samples, n_components, n_features) array.

        Returns:
            Feature contributions array. Shape is (n_samples, n_components, n_features)
            when reduction is ``None``, or (n_samples, n_features) when a reduction
            is applied.
        """
        self._fitted_model.eval()
        self._fitted_model.to(self._device)

        if batch_size is None:
            batch_size = self.batch_size

        assert self._mean is not None
        X_centered = _to_numpy_float32(X) - self._mean

        if self._pca is not None:
            X_encoder = torch.from_numpy(self._pca.transform(X_centered).astype(np.float32))
        else:
            X_encoder = torch.from_numpy(X_centered)

        X_encoder = X_encoder.to(self._device)

        jacobians = self.compute_jacobian(X_encoder, batch_size=batch_size)

        if self._pca is not None:
            proj_tensor = torch.tensor(
                self._pca.components_,
                dtype=torch.float32,
                device=self._device,
            )
            jacobians = project_jacobian(jacobians, proj_tensor)

        X_centered_t = torch.from_numpy(X_centered).unsqueeze(1).to(self._device)
        feature_contributions = (jacobians * X_centered_t).cpu().numpy()

        if reduction is not None:
            feature_contributions = reduce_contributions(feature_contributions, method=reduction)

        return feature_contributions

    def compute_jacobian(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        return compute_jacobian(self._fitted_model.encoder, x, batch_size)
