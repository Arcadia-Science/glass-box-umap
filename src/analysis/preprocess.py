# -*- coding: utf-8 -*-
"""
Refactored script for single-cell analysis using Scanpy.

This script processes CITE-seq data of bone marrow mononuclear cells (BMMCs)
by performing quality control, normalization, dimensionality reduction,
and clustering.
"""

# Core libraries
import os
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pooch

# --- Function Definitions ---

def download_and_load_data(url, filename):
    """
    Downloads, unzips, and loads the AnnData file.

    Args:
        url (str): The URL to the compressed .h5ad file.
        known_hash (str): The expected SHA256 hash of the file for validation.

    Returns:
        ad.AnnData: The loaded AnnData object.
    """
    print("Downloading and loading data...")
    
    # # Download the compressed file from the GEO FTP server
    # change to python
    # !wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
    # ftp_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
    import subprocess
    # subprocess.run(["wget", url])
    # # Unzip the file
    # change to python
    # !gunzip GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
    # h5ad_filename = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"

    subprocess.run(["gunzip", filename])

    adata = ad.read_h5ad('GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad')

    # Use pooch to download and cache the file
    # scanpy example
    # file_path = pooch.retrieve(
    #     url=url,
    #     known_hash=known_hash,
    #     processor=pooch.Decompress(), # Automatically decompress .gz
    # )
    # adata = ad.read_h5ad(file_path)
    print("Data loaded successfully.")
    return adata


def initial_preprocessing(adata):
    """
    Performs initial preprocessing steps like ensuring unique names and
    annotating gene types.

    Args:
        adata (ad.AnnData): The AnnData object.

    Returns:
        ad.AnnData: The preprocessed AnnData object.
    """
    print("⚙️  Running initial preprocessing...")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # Annotate mitochondrial, ribosomal, and hemoglobin genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    print("Initial preprocessing complete.")
    return adata


def run_quality_control(adata, perform_qc=True):
    """
    Calculates and visualizes quality control metrics.

    Args:
        adata (ad.AnnData): The AnnData object.
        perform_qc (bool): If True, calculates and plots QC metrics.

    Returns:
        ad.AnnData: The AnnData object, possibly with new QC metrics.
    """
    if not perform_qc:
        print("Skipping quality control step.")
        return adata

    print("🔬 Running quality control...")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )

    # Visualize QC metrics
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
    )
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")
    print("Quality control complete.")
    return adata


def filter_and_detect_doublets(adata, run_scrublet=False):
    """
    Filters cells/genes and runs doublet detection.

    Args:
        adata (ad.AnnData): The AnnData object.
        run_scrublet (bool): If True, runs Scrublet for doublet detection.

    Returns:
        ad.AnnData: The filtered AnnData object.
    """
    print("🧹 Filtering data and detecting doublets...")
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # Optional doublet detection
    if run_scrublet:
        print("🧬 Running Scrublet for doublet detection...")
        sc.pp.scrublet(adata, batch_key="sample")
    
    print("Filtering and doublet detection complete.")
    return adata


def normalize_and_select_features(adata):
    """
    Normalizes data and selects highly variable genes.

    Args:
        adata (ad.AnnData): The AnnData object.

    Returns:
        ad.AnnData: The processed AnnData object.
    """
    print("⚖️  Normalizing data and selecting features...")
    # Save raw counts
    adata.layers["counts"] = adata.X.copy()
    # Normalize and log-transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Find and plot highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pl.highly_variable_genes(adata)
    print("Normalization and feature selection complete.")
    return adata


def run_dimensionality_reduction(adata):
    """
    Performs PCA and visualizes the results.

    Args:
        adata (ad.AnnData): The AnnData object.

    Returns:
        ad.AnnData: The AnnData object with PCA results.
    """
    print("📉 Reducing dimensionality with PCA...")
    sc.tl.pca(adata)

    # Visualize PCA results
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
    sc.pl.pca(
        adata,
        color=["cell_type", "pct_counts_mt"],
        dimensions=[(0, 1), (2, 3)],
        ncols=2,
    )
    print("PCA complete.")
    return adata


def compute_embedding_and_clusters(adata):
    """
    Computes nearest neighbors, UMAP embedding, and Leiden clustering.

    Args:
        adata (ad.AnnData): The AnnData object.

    Returns:
        ad.AnnData: The AnnData object with UMAP and clustering results.
    """
    print("📊 Computing neighborhood graph, UMAP, and clusters...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    print("Embedding and clustering complete.")
    return adata


def visualize_final_results(adata, perform_qc_reassessment=True):
    """
    Generates final UMAP visualizations for cell types, clusters, and QC metrics.

    Args:
        adata (ad.AnnData): The final AnnData object.
        perform_qc_reassessment (bool): If True, plots UMAPs with QC metrics.
    """
    print("🎨 Generating final visualizations...")
    # Visualize cell types and clusters on UMAP
    sc.pl.umap(adata, color="cell_type", size=2)
    sc.pl.umap(adata, color=["leiden"])

    # Re-assess QC metrics on the UMAP
    if perform_qc_reassessment:
        sc.pl.umap(
            adata,
            color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
            wspace=0.5,
            ncols=2,
        )
    print("🎉 Analysis visualization complete!")

def preprocess_bone_marrow(DOWNLOAD_DATA = True, FILTER_BY_SAMPLES = False, QUALITY_CONTROL = True,
                            data_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",                                
                            h5ad_filename = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
    ):
    """
    Main function to run the single-cell analysis workflow.
    """
    # --- Parameters ---
    # Set to False to skip downloading if the file already exists and is unzipped
    # DOWNLOAD_DATA = True
    # # NOTE: The original script had this flag but didn't use it to load separate samples.
    # # Set to True to run doublet detection (requires `batch_key`).
    # FILTER_BY_SAMPLES = False
    # # Set to False to skip the optional QC metric calculation and plotting
    # QUALITY_CONTROL = False

    # --- Workflow ---
    if DOWNLOAD_DATA:
        adata = download_and_load_data(data_url, h5ad_filename)
    else:
        # Assumes the file is already downloaded and unzipped in the current directory
        adata = ad.read_h5ad(h5ad_filename)

    adata = initial_preprocessing(adata)
    adata = run_quality_control(adata, perform_qc=QUALITY_CONTROL)
    adata = filter_and_detect_doublets(adata, run_scrublet=FILTER_BY_SAMPLES)
    adata = normalize_and_select_features(adata)
    adata = run_dimensionality_reduction(adata)
    adata = compute_embedding_and_clusters(adata)
    visualize_final_results(adata, perform_qc_reassessment=QUALITY_CONTROL)
    
    return adata
