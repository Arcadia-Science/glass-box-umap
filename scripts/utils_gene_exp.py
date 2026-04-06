import os
import subprocess

import anndata as ad
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle

# ── Data loading & preprocessing ──────────────────────────────────────────────


def download_bone_marrow_data(
    url="ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
    filename="GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
) -> ad.AnnData:
    """Download, unzip, and load the bone marrow dataset."""
    unzipped = filename.replace(".gz", "")
    if not os.path.isfile(unzipped):
        if not os.path.isfile(filename):
            subprocess.run(["wget", url, "--no-verbose"])
        subprocess.run(["gunzip", filename])
    return sc.read_h5ad(unzipped)


def preprocess_adata(
    adata: ad.AnnData,
    min_genes: int = 100,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    batch_key: str = "Samplename",
    run_scrublet: bool = False,
) -> ad.AnnData:
    """Full scRNA-seq preprocessing: QC filtering, normalize, HVG selection, PCA."""
    print("--- Starting Preprocessing ---")

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # Flag QC gene categories
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    adata.var["technical"] = adata.var_names.str.startswith(("MALAT1", "NEAT1", "FOS", "JUN"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb", "technical"], inplace=True, log1p=True
    )

    # Remove QC genes, filter cells/genes
    genes_to_remove = adata.var["mt"] | adata.var["ribo"]  # | adata.var["technical"]
    adata._inplace_subset_var(~genes_to_remove)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if run_scrublet:
        sc.pp.scrublet(adata, batch_key=batch_key)

    # Normalize, log-transform, HVGs
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)

    # Now regress out on the full matrix (or subset to HVGs first)
    sc.pp.regress_out(adata, ["total_counts"])

    # Store HVG mean for later gene-space reconstruction
    X_hvg = adata[:, adata.var.highly_variable].X
    if hasattr(X_hvg, "toarray"):
        X_hvg = X_hvg.toarray()
    adata.uns["pca_mean_hvg"] = X_hvg.mean(axis=0)

    # PCA
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)

    return adata


def prepare_data(
    groupby_key: str = "cell_type",
    n_pcs: int = 50,
    batch_key: str = "Samplename",
    n_top_cell_types: int = 32,
) -> ad.AnnData:
    """Load, preprocess, and subset to top cell types."""
    adata_raw = download_bone_marrow_data()
    adata_raw.var_names_make_unique()

    adata_processed = preprocess_adata(
        adata_raw, n_top_genes=2000, n_pcs=n_pcs, batch_key=batch_key
    )

    top_types = adata_processed.obs[groupby_key].value_counts().nlargest(n_top_cell_types).index
    adata_final = adata_processed[adata_processed.obs[groupby_key].isin(top_types)].copy()

    return adata_final


def print_top_genes_per_cluster(importance_gene, gene_names_hvg, labels, label_names, n_top=8):
    """Print top contributing genes for each cluster."""
    print(f"\n── Top {n_top} Genes Per Cluster ──")
    for cluster_id in range(labels.max() + 1):
        mask = labels == cluster_id
        mean_imp = importance_gene[mask].mean(axis=0)
        top_genes = gene_names_hvg[np.argsort(mean_imp)[::-1][:n_top]]
        print(f"  {label_names[cluster_id]}: {list(top_genes)}")


# ── Plotting helpers ──────────────────────────────────────────────────────────


def make_tab40_cmap(seed=None):
    """Create a 40-color colormap with a randomized order."""
    top = plt.get_cmap("tab20b").colors
    bottom = plt.get_cmap("tab20c").colors
    combined = np.vstack((top, bottom))

    # Generate a random permutation
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(combined))
    shuffled_colors = combined[shuffled_indices]

    cmap = ListedColormap(shuffled_colors, name="tab40")
    # Optional: Register it if you want to call it by string later
    plt.colormaps.register(cmap=cmap)

    return cmap


def plot_embedding(
    Z, labels, cmap, filename, title="Parametric UMAP", show=False, label_names=None
):
    """Scatter plot of embedding colored by cell type."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap=cmap,
        s=0.1,
        marker=MarkerStyle("o", fillstyle="full"),
        alpha=0.5,
        rasterized=True,
    )
    plt.colorbar(scatter, ax=ax, label="Cell Type")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    if label_names is not None:
        for code, name in enumerate(label_names):
            mask = labels == code
            if mask.sum() == 0:
                continue
            cx, cy = np.median(Z[mask, 0]), np.median(Z[mask, 1])
            ax.annotate(
                name,
                (cx, cy),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            )
    if show:
        plt.show()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_gene_map(Z, top_gene_names, gene_names_hvg, filename, n_label=60, show=False):
    """UMAP colored by each cell's top contributing gene, with centroid labels."""
    unique_genes, counts = np.unique(top_gene_names, return_counts=True)
    top_n_genes = unique_genes[np.argsort(counts)[::-1][:n_label]]
    cmap = cm.get_cmap("tab40", 40)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, gene in enumerate(top_n_genes):
        mask = top_gene_names == gene
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=0.3,
            alpha=0.7,
            color=cmap(i % 40),
            label=gene,
            rasterized=True,
        )

    # Centroid labels
    for gene in top_n_genes:
        mask = top_gene_names == gene
        # cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        cx, cy = np.median(Z[mask, 0]), np.median(Z[mask, 1])
        ax.annotate(
            gene,
            (cx, cy),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )

    ax.set_title("UMAP — colored by top contributing gene")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=8, fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_embedding_and_top_genes(
    Z,
    labels,
    top_gene_names,
    gene_names_hvg,
    filename,
    title="Parametric UMAP",
    label_names=None,
    n_label=60,
    show=False,
):
    """Side-by-side plot: cell-type embedding (left) and top-gene map (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left: cell-type embedding ─────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap="tab40",
        s=0.1,
        marker=MarkerStyle("o", fillstyle="full"),
        alpha=0.5,
        rasterized=True,
    )
    # plt.colorbar(scatter, ax=ax, label="Cell Type")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)
    if label_names is not None:
        for code, name in enumerate(label_names):
            mask = labels == code
            if mask.sum() == 0:
                continue
            cx, cy = np.median(Z[mask, 0]), np.median(Z[mask, 1])
            ax.annotate(
                name,
                (cx, cy),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            )

    # ── Right: top-gene map ───────────────────────────────────────────────────
    ax = axes[1]
    unique_genes, counts = np.unique(top_gene_names, return_counts=True)
    top_n_genes = unique_genes[np.argsort(counts)[::-1][:n_label]]
    gene_cmap = cm.get_cmap("tab40", 40)

    for i, gene in enumerate(top_n_genes):
        mask = top_gene_names == gene
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=0.3,
            alpha=0.7,
            color=gene_cmap(i % 40),
            label=gene,
            rasterized=True,
        )
    for gene in top_n_genes:
        mask = top_gene_names == gene
        cx, cy = np.median(Z[mask, 0]), np.median(Z[mask, 1])
        ax.annotate(
            gene,
            (cx, cy),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )
    ax.set_title("UMAP — colored by top contributing gene", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # ax.legend(markerscale=8, fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
