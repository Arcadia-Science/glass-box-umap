import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.func import functional_call, jacrev, vmap


def compute_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Compute the Jacobian of a model using ``vmap`` + ``jacrev`` with ``functional_call``.

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
        return functional_call(model, {**params, **buffers}, (x_single.unsqueeze(0),)).squeeze(0)

    jac_fn = vmap(jacrev(func_single))

    results = []
    for start in range(0, x.shape[0], batch_size):
        x_batch = x[start : start + batch_size]
        with torch.no_grad():
            J_batch = jac_fn(x_batch)
        results.append(J_batch)

    return torch.cat(results, dim=0)


def project_jacobian(jacobian: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,jk->bik", jacobian, proj_tensor)


@dataclass
class JacobianVerification:
    """Result of verifying that ``f(x) ≈ J(x) @ x``.

    Attributes:
        z_range: (min, max) of the embedding output.
        reconstruction_range: (min, max) of the Jacobian reconstruction.
        max_error: Maximum absolute error between embedding and reconstruction.
        mean_error: Mean absolute error between embedding and reconstruction.
        relative_error: Max error relative to the embedding's magnitude.
    """

    z_range: tuple[float, float]
    reconstruction_range: tuple[float, float]
    max_error: float
    mean_error: float
    relative_error: float


def verify_jacobian(
    Z: NDArray[np.floating],
    J: NDArray[np.floating],
    X: NDArray[np.floating],
) -> JacobianVerification:
    """Verify that ``f(x) ≈ J(x) @ x``.

    Args:
        Z: Embedding output, shape ``(n, out_dim)``.
        J: Jacobian, shape ``(n, out_dim, in_dim)``.
        X: Input data, shape ``(n, in_dim)``.

    Returns:
        A ``JacobianVerification`` with error diagnostics.
    """
    Z_reconstructed = np.einsum("noi,ni->no", J, X)
    return JacobianVerification(
        z_range=(float(Z.min()), float(Z.max())),
        reconstruction_range=(float(Z_reconstructed.min()), float(Z_reconstructed.max())),
        max_error=float(np.abs(Z - Z_reconstructed).max()),
        mean_error=float(np.abs(Z - Z_reconstructed).mean()),
        relative_error=float(np.abs(Z - Z_reconstructed).max() / (np.abs(Z).max() + 1e-8)),
    )
