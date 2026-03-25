"""Gene-space attribution utilities for scRNA-seq data.

Maps Jacobians from PCA space to gene space and computes per-gene
contributions to the UMAP embedding.
"""

from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


def compute_gene_contributions(
    J: NDArray[np.floating],
    adata: Any,  # anndata.AnnData; full stubs pending scverse/anndata#2173
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.str_]]:
    """Map Jacobian from PCA space to gene space and compute per-gene contributions.

    Uses the chain rule to project the PCA-space Jacobian through the PCA
    loadings matrix, then multiplies by mean-centered HVG expression to get
    each gene's additive contribution to the embedding.

    Args:
        J: Jacobian in PCA space, shape ``(n, out_dim, n_pcs)``.
        adata: AnnData object with ``.var.highly_variable``, ``.varm["PCs"]``,
            and ``adata.uns["pca_mean_hvg"]`` populated by preprocessing.

    Returns:
        A tuple of:
            - ``contributions_gene``: shape ``(n, out_dim, n_hvgs)``
            - ``importance_gene``: L2 norm over embedding dims, shape ``(n, n_hvgs)``
            - ``gene_names_hvg``: array of HVG gene names
    """
    hvg_mask = adata.var.highly_variable.values
    W_hvg = adata.varm["PCs"][hvg_mask, :]  # (n_hvgs, n_pcs)
    gene_names_hvg = adata.var_names[hvg_mask].values

    # Jacobian w.r.t. HVG expression: J_gene = J_pc @ W_hvg.T
    J_gene = np.einsum("noi,gi->nog", J, W_hvg)  # (n, out_dim, n_hvgs)

    # Gene-space input (mean-centered HVG expression)
    X_hvg = adata[:, hvg_mask].X
    if hasattr(X_hvg, "toarray"):
        X_hvg = X_hvg.toarray()
    pca_mean = adata.uns["pca_mean_hvg"]
    X_hvg_centered = X_hvg - pca_mean

    # Per-gene contributions and importance
    contributions_gene = np.einsum(
        "nog,nog->nog", J_gene, X_hvg_centered[:, np.newaxis, :]
    )  # (n, out_dim, n_hvgs)
    importance_gene = np.linalg.norm(contributions_gene, axis=1)  # (n, n_hvgs)

    return contributions_gene, importance_gene, gene_names_hvg


def verify_gene_reconstruction(
    contributions_gene: NDArray[np.floating],
    Z: NDArray[np.floating],
    tol: float = 1e-3,
) -> float:
    """Verify that gene-space contributions sum to the embedding.

    Args:
        contributions_gene: shape ``(n, out_dim, n_hvgs)``.
        Z: Embedding, shape ``(n, out_dim)``.
        tol: Relative error threshold for PASS/FAIL.

    Returns:
        Relative max error.
    """
    Z_reconstructed = contributions_gene.sum(axis=2)
    max_err = np.abs(Z_reconstructed - Z).max()
    mean_err = np.abs(Z_reconstructed - Z).mean()
    rel_err = max_err / (np.abs(Z).max() + 1e-8)
    print("\n── Gene-Space Reconstruction ──")
    print(f"  Max error  : {max_err:.2e}")
    print(f"  Mean error : {mean_err:.2e}")
    print(f"  Rel error  : {rel_err:.2e}")
    print(f"  Verification {'PASSED ✓' if rel_err < tol else 'FAILED ✗'}")
    return rel_err
