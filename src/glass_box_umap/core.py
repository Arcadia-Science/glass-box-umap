import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.func import functional_call, jacrev, vmap  # type: ignore[reportPrivateImportUsage]

from glass_box_umap.components import DeepPReLUNet
from glass_box_umap.parametric_umap.registry import register_encoder

from .parametric_umap import ParametricUMAP
from .parametric_umap.core import _to_numpy_float32

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
        X = _to_numpy_float32(X) - self._mean

        if self._pca is not None:
            X = self._pca.transform(X).astype(np.float32)

        X = torch.from_numpy(X).to(self._device)

        # Convert PReLU -> LeakyReLU for vmap-compatible Jacobian computation
        encoder_for_jac = self.prelu_to_leaky(encoder)
        jacobians_input = self.compute_jacobian(encoder_for_jac, X, batch_size=batch_size)

        if self._pca is not None:
            proj_tensor = torch.tensor(self._pca.components_, dtype=torch.float32)
            jacobians_raw = torch.einsum("bij,jk->bik", jacobians_input, proj_tensor)
        else:
            jacobians_raw = jacobians_input

        feature_contributions = (jacobians_raw.numpy() * X[:, np.newaxis, :]).astype(np.float16)

        return feature_contributions, jacobians_input

    def prelu_to_leaky(self, model: nn.Module) -> nn.Module:
        """Replace all PReLU modules with LeakyReLU using the learned slopes.

        This is needed for Jacobian computation via ``vmap`` + ``jacrev``, which
        requires stateless activations.

        Args:
            model: The model to convert (not modified in-place).

        Returns:
            A deep copy of the model with PReLU replaced by LeakyReLU.
        """
        model = copy.deepcopy(model)
        for name, module in model.named_modules():
            if isinstance(module, nn.PReLU):
                slope = (
                    module.weight.detach().item()
                    if module.weight.numel() == 1
                    else module.weight.detach().mean().item()
                )
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], nn.LeakyReLU(negative_slope=slope))
        return model

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

    def verify_jacobian(
        self,
        Z: NDArray[np.floating],
        J: NDArray[np.floating],
        X: NDArray[np.floating],
        tol: float = 1e-4,
    ) -> float:
        """Verify that ``f(x) ≈ J(x) @ x`` and print diagnostics.

        Args:
            Z: Embedding output, shape ``(n, out_dim)``.
            J: Jacobian, shape ``(n, out_dim, in_dim)``.
            X: Input data, shape ``(n, in_dim)``.
            tol: Relative error threshold for PASS/FAIL.

        Returns:
            Relative max error.
        """
        Z_reconstructed = np.einsum("noi,ni->no", J, X)
        max_err = np.abs(Z - Z_reconstructed).max()
        mean_err = np.abs(Z - Z_reconstructed).mean()
        rel_err = max_err / (np.abs(Z).max() + 1e-8)
        print("\n── Jacobian Exactness Verification ──")
        print(f"  f(x)       range : [{Z.min():.4f}, {Z.max():.4f}]")
        print(f"  J(x)@x     range : [{Z_reconstructed.min():.4f}, {Z_reconstructed.max():.4f}]")
        print(f"  Max |f(x) - J(x)@x|  : {max_err:.2e}")
        print(f"  Mean |f(x) - J(x)@x| : {mean_err:.2e}")
        print(f"  Relative max error    : {rel_err:.2e}")
        print(f"  Verification {'PASSED ✓' if rel_err < tol else 'FAILED ✗'}")
        return rel_err
