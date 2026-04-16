import copy
from dataclasses import dataclass
from typing import Literal

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


def reduce_contributions(
    contributions: NDArray[np.floating],
    method: Literal["l2"] = "l2",
) -> NDArray[np.floating]:
    """Reduce per-feature contributions across embedding dimensions.

    Args:
        contributions:
            Feature contributions with shape (n_samples, n_components, n_features).
        method:
            Reduction method. ``"l2"`` takes the L2 norm across components.

    Returns:
        Reduced contributions with shape (n_samples, n_features).
    """
    match method:
        case "l2":
            return np.linalg.norm(contributions, axis=1)


def groups_from_top_features(
    contributions: NDArray[np.floating],
    feature_names: NDArray[np.str_] | None = None,
    top_n: int = 60,
) -> tuple[NDArray[np.integer], list[str], NDArray[np.bool_]]:
    """Assign each point to its highest-contributing feature, keeping only the most common.

    Args:
        contributions:
            Per-feature contribution scores with shape (n_samples, n_features) or
            (n_samples, n_components, n_features). If 3D, L2 reduction is applied
            across embedding dimensions.
        feature_names:
            Feature names with shape (n_features,), matching columns of contributions.
            If None, defaults to "0", "1", "2", etc.
        top_n:
            Number of most frequent top features to keep.

    Returns:
        group_ids: Integer group ID per kept point, shape (n_kept,).
        group_names: Name for each group, indexed by group ID.
        mask: Boolean mask of shape (n_samples,) indicating which points are kept.
    """
    if contributions.ndim == 3:
        contributions = reduce_contributions(contributions)

    if feature_names is None:
        feature_names = np.array([str(i) for i in range(contributions.shape[1])])

    top_feature_per_point = feature_names[contributions.argmax(axis=1)]
    unique_features, counts = np.unique(top_feature_per_point, return_counts=True)
    top_features = unique_features[np.argsort(counts)[::-1][:top_n]]

    mask = np.isin(top_feature_per_point, top_features)
    feature_to_id = {name: i for i, name in enumerate(top_features)}
    group_ids = np.array([feature_to_id[f] for f in top_feature_per_point[mask]])
    group_names = list(top_features)

    return group_ids, group_names, mask


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
