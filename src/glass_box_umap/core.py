import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.func import functional_call, jacrev, vmap

from glass_box_umap.components import DeepPReLUNet
from glass_box_umap.parametric_umap.registry import register_encoder

from .parametric_umap import ParametricUMAP
from .parametric_umap.core import _to_numpy_float32

GLASSBOX_DEFAULT_ENCODER_NAME = "glassbox_default_encoder"
register_encoder(GLASSBOX_DEFAULT_ENCODER_NAME)(DeepPReLUNet)


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

    # Overwrite base class default with `GLASSBOX_DEFAULT_ENCODER_NAME`.
    encoder_name: str = field(default=GLASSBOX_DEFAULT_ENCODER_NAME)

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
            batch_size:
                Batch size for Jacobian computation. Defaults to ``self.batch_size``.
        """
        self._fitted_model.eval()
        self._fitted_model.to(self._device)
        encoder = self._fitted_model.encoder

        if batch_size is None:
            batch_size = self.batch_size

        assert self._mean is not None
        X_centered = _to_numpy_float32(X) - self._mean

        if self._pca is not None:
            X_encoder = torch.from_numpy(self._pca.transform(X_centered).astype(np.float32))
        else:
            X_encoder = torch.from_numpy(X_centered)

        X_encoder = X_encoder.to(self._device)

        jacobians_input = self.compute_jacobian(encoder, X_encoder, batch_size=batch_size)

        if self._pca is not None:
            proj_tensor = torch.tensor(
                self._pca.components_, dtype=torch.float32, device=self._device
            )
            jacobians_raw = torch.einsum("bij,jk->bik", jacobians_input, proj_tensor)
        else:
            jacobians_raw = jacobians_input

        X_centered_t = torch.from_numpy(X_centered).unsqueeze(1).to(self._device)
        feature_contributions = (jacobians_raw * X_centered_t).cpu().numpy()

        return feature_contributions, jacobians_input

    def compute_jacobian(
        self,
        model: nn.Module,
        x: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Compute the Jacobian of a model using ``vmap`` + ``jacrev`` with ``functional_call``.

        Compatible with LayerNormDetached, LeakyReLU, and other stateless layers.

        Args:
            model: Encoder network (will be deep-copied and set to eval mode).
            x: Input tensor of shape ``(n, in_dim)``.
            batch_size: Number of samples per Jacobian batch.

        Returns:
            Jacobian tensor of shape ``(n, out_dim, in_dim)``.
        """
        model = copy.deepcopy(model).eval()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def func_single(x_single: torch.Tensor) -> torch.Tensor:
            return functional_call(model, {**params, **buffers}, (x_single.unsqueeze(0),)).squeeze(
                0
            )

        jac_fn = vmap(jacrev(func_single))

        results = []
        for start in range(0, x.shape[0], batch_size):
            x_batch = x[start : start + batch_size]
            with torch.no_grad():
                J_batch = jac_fn(x_batch)
            results.append(J_batch)

        return torch.cat(results, dim=0)
