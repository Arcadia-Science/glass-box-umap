"""
Glass-Box UMAP — Bone Marrow (BMMC) Gene Expression
====================================================
Trains a parametric UMAP on bone marrow scRNA-seq data,
computes exact Jacobian-based feature contributions,
and identifies top contributing genes per cell / cluster.
"""

import time
from pathlib import Path

import numpy as np
import torch
from glass_box_umap import GlassBoxUMAP
from glass_box_umap.attribution import compute_gene_contributions, verify_gene_reconstruction
from glass_box_umap.diagnostics import verify_jacobian
from scipy.stats import spearmanr
from utils_gene_exp import (
    make_tab40_cmap,
    plot_embedding,
    plot_top_gene_map,
    prepare_data,
    print_top_genes_per_cluster,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    DEVICE = "mps"
    N_PCS = 200
    EPOCHS = 4
    REP_STRENGTH = 3.0
    N_NEIGHBORS = 64
    BATCH_SIZE = 128 * 4 * 8  # higher may be better
    NUM_BATCHES = 900
    BATCH_KEY = "Samplename"
    GROUPBY_KEY = "cell_type"
    DESC = (
        f"mar24_vis_gene_exp_ep{str(EPOCHS)}_repStr_{str(int(REP_STRENGTH))}_"
        f"nn_{str(N_NEIGHBORS)}_batch_{str(BATCH_SIZE)}_regOut_scale_mar23"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    adata = prepare_data(groupby_key=GROUPBY_KEY, n_pcs=N_PCS, batch_key=BATCH_KEY)
    labels = adata.obs[GROUPBY_KEY].astype("category").cat.codes
    label_names = adata.obs[GROUPBY_KEY].cat.categories

    # ── Fit parametric UMAP ───────────────────────────────────────────────────
    t0 = time.time()

    reducer = GlassBoxUMAP(
        epochs=EPOCHS,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        repulsion_strength=REP_STRENGTH,
        min_dist=0.8,
        n_neighbors=N_NEIGHBORS,
        encoder_name="default",
        checkpoint_dir=Path(f"tmp_{EPOCHS}"),
        random_state=11,
        num_workers=10,
    )
    reducer.fit(adata.obsm["X_pca"])
    output = reducer.transform(adata.obsm["X_pca"])

    print(f"Training time: {(time.time() - t0) / 60:.1f} min")

    # ── Plot embedding ────────────────────────────────────────────────────────
    tab40 = make_tab40_cmap()
    plot_embedding(
        output,
        labels,
        "tab40",
        f"fc_umap_glassbox_{DESC}.png",
        title="Parametric UMAP — Bone Marrow Gene Expression",
        label_names=label_names,
    )

    # ── Compute Jacobians ─────────────────────────────────────────────────────
    print("Computing Jacobians...")
    encoder = reducer._fitted_model.encoder
    X_tensor = torch.tensor(adata.obsm["X_pca"], device=DEVICE)

    encoder.eval()
    with torch.no_grad():
        Z_np = encoder(X_tensor).cpu().numpy()

    encoder_for_jac = reducer.prelu_to_leaky(encoder)
    J = reducer.compute_jacobian(encoder_for_jac, X_tensor)
    J_np = J.cpu().numpy()
    X = X_tensor.cpu().numpy()

    # ── Verify Jacobian exactness ─────────────────────────────────────────────
    print(verify_jacobian(Z_np, J_np, X))

    # ── Gene-space contributions ──────────────────────────────────────────────
    contributions_gene, importance_gene, gene_names_hvg = compute_gene_contributions(J_np, adata)
    verify_gene_reconstruction(contributions_gene, Z_np)
    print_top_genes_per_cluster(importance_gene, gene_names_hvg, labels.values, label_names)

    # ── Top-gene colored UMAP ─────────────────────────────────────────────────
    top_gene_idx = np.argmax(importance_gene, axis=1)
    top_gene_names = gene_names_hvg[top_gene_idx]

    plot_top_gene_map(
        Z_np,
        top_gene_names,
        gene_names_hvg,
        f"residual_umap_{DESC}_{EPOCHS}_top_gene_colored.png",
    )

    # ── Correlation sanity checks ─────────────────────────────────────────────
    n_umi = np.array(adata.obs["n_genes_by_counts"])

    for gene in ["MALAT1", "HBB"]:
        idx = np.where(gene_names_hvg == gene)[0]
        if len(idx) > 0:
            r, p = spearmanr(n_umi, importance_gene[:, idx[0]])
            print(f"  {gene} vs nUMI: r={r:.3f}, p={p:.2e}")

    # ── Seq-depth correlation screen (top 100 genes) ─────────────────────────────
    n_umi = np.array(adata.obs["n_genes_by_counts"])

    # Rank genes by mean importance across all cells
    mean_importance = importance_gene.mean(axis=0)
    top100_idx = np.argsort(mean_importance)[::-1][:100]

    results = []
    for idx in top100_idx:
        r, p = spearmanr(n_umi, importance_gene[:, idx])
        results.append((gene_names_hvg[idx], r, p, mean_importance[idx]))

    # Sort by |r| to surface the worst offenders
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # print(f"\n── Seq-depth correlation screen (top 100 genes by mean importance) ──")
    # print(f"{'Gene':<12} {'r':>8} {'p':>10} {'mean_imp':>10}  flag")
    # flagged = []
    # for gene, r, p, imp in results:
    #     flag = " ◄ ARTIFACT?" if abs(r) > 0.2 and p < 1e-10 else ""
    #     if flag:
    #         flagged.append(gene)
    #     print(f"{gene:<12} {r:>8.3f} {p:>10.2e} {imp:>10.4f}{flag}")

    # print(f"\nFlagged genes: {flagged}")

    # Compute residual importance after regressing out cell type mean
    from scipy.stats import spearmanr

    cell_type_mean = np.zeros_like(importance_gene)
    for ct in np.unique(labels):
        mask = labels == ct
        cell_type_mean[mask] = importance_gene[mask].mean(axis=0)

    importance_residual = importance_gene - cell_type_mean

    # Now correlate residuals with nUMI
    # results_corrected = []
    # for i, gene in enumerate(gene_names_hvg[top100_idx]):
    #     r, p = spearmanr(n_umi, importance_residual[:, top100_idx[i]])
    #     results_corrected.append((gene, r, p))
    # print(results_corrected)
