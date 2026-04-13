from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
