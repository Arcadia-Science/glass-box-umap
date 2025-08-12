# -*- coding: utf-8 -*-
"""
Refactored script for training a parametric UMAP model, calculating its Jacobian
to find feature importances, and visualizing the results for different cell types.
"""

# --- Core Libraries ---
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc

from .utils import PUMAP, Datamodule, Model
from .networks import deepReLUNet, deepSiLUNet, deepBilinearNet

# --- Function Definitions ---

def setup_environment(seed_value: int):
    """Sets random seeds for reproducibility."""
    print(f"🌱 Setting random seed to {seed_value}")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

def prepare_data(adata: sc.AnnData, pca_components: int = 50, pca_start: int = 0):
    """
    Prepares training data from an AnnData object.

    Args:
        adata: The annotated data matrix.
        pca_components: The number of principal components to use.

    Returns:
        A tuple containing the training dataset (PCA embeddings) and the
        mean-centered gene expression matrix.
    """
    print("Preparing data for training...")
    train_dataset = torch.tensor([adata.obsm["X_pca"][:, pca_start:pca_components]], dtype=torch.float32)
    
    # Calculate mean-centered gene expression matrix
    adf = adata.to_df()
    adfmz = adf.values - adf.mean(axis=0).values
    
    print(f" Data prepared with training shape: {train_dataset.shape}")
    return train_dataset, adfmz

def train_parametric_umap(network, train_dataset, config: dict, precomputed_distances=None):
    """
    Initializes and trains the Parametric UMAP model.

    Args:
        network: The neural network instance to be trained.
        train_dataset: The input data for training.
        config: A dictionary with training parameters (lr, epochs, etc.).

    Returns:
        The trained PUMAP encoder object.
    """
    print(" Training Parametric UMAP model...")
    # NOTE: PUMAP is an external class, assumed to be available.
    encoder = PUMAP(
        network,
        lr=config.get("lr", 8e-4),
        epochs=config.get("epochs", 24),
        batch_size=config.get("batch_size", 1024),
        random_state=config.get("seed", 24),
        **config.get("pumap_kwargs", {})
    )
    # The fit method is assumed to exist on the PUMAP object
    encoder.fit(train_dataset, precomputed_graph=precomputed_distances)
    print(" Model training complete.")
    return encoder

def generate_and_plot_embedding(encoder, adata: sc.AnnData, train_dataset: torch.Tensor):
    """
    Generates embedding, adds it to adata, and creates visualization plots.

    Args:
        encoder: The trained PUMAP encoder.
        adata: The annotated data matrix for plotting.
        train_dataset: The data to transform into an embedding.

    Returns:
        The generated embedding as a numpy array.
    """
    print("✨ Generating and visualizing embedding...")
    embedding = encoder.transform(train_dataset.squeeze()).squeeze()

    # Plot with Seaborn
    plt.figure()
    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=adata.obs["cell_type"], s=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
    plt.title("Parametric UMAP Embedding")
    plt.show()

    # Plot with Scanpy
    adata.obsm['X_parametric_umap'] = embedding
    sc.pl.umap(adata, use_raw=False, alpha=0.85, color='cell_type')
    
    return embedding

def compute_gene_space_jacobian(encoder, adata: sc.AnnData, train_dataset: torch.Tensor, adfmz, batch_size: int = 40, return_jacobian: bool = False):
    """
    Computes the model's Jacobian and projects it to the original gene space.

    Args:
        encoder: The trained model encoder.
        adata: The AnnData object, containing PCA components.
        train_dataset: The input data (in PCA space).
        adfmz: The mean-centered gene expression matrix.
        batch_size: The batch size for Jacobian calculation.

    Returns:
        The Jacobian projected into gene space and weighted by expression.
    """
    print("🧠 Computing Jacobian and projecting to gene space...")
    encoder.eval()
    
    # 1. Compute Jacobian in batches (in PCA space)
    num_samples = train_dataset.shape[1]
    jacobians_pca = []
    for i in range(0, num_samples, batch_size):
        data_batch = train_dataset[0][i:i+batch_size, :]
        jac_batch = torch.autograd.functional.jacobian(
            encoder, data_batch, vectorize=True, strategy="reverse-mode"
        ).squeeze()
        # Handle the diagonal from the vectorized output
        if jac_batch.ndim == 4:
             jacobians_pca.extend([jac_batch[j, :, j, :] for j in range(jac_batch.shape[0])])
        else: # Handle the last, non-vectorized batch
             jacobians_pca.append(jac_batch)

    # 2. Project Jacobian to gene space for each sample
    jacobxall = []
    pcs = adata.varm["PCs"]
    for i in range(num_samples):
        jac_pca = jacobians_pca[i]
        # Project Jacobian from PCA space to gene space: J_gene = J_pca @ PCs.T
        jnp = (pcs @ jac_pca.detach().cpu().float().numpy().T).T
        # Weight by that cell's gene expression
        jnpx = jnp * adfmz[i]
        jacobxall.append(jnpx.astype('float16'))

    print(" Jacobian calculation complete.")
    if return_jacobian:
        return np.array(jacobxall), jacobians_pca
    else:
        return np.array(jacobxall)

def plot_feature_importance(adata: sc.AnnData, embedding, jacobxall, celltypes: list, n_features: int = 20, stat: str = 'mean', showPlot=True):
    """
    For each cell type, plots the UMAP and the most important features derived
    from the Jacobian.

    Args:
        adata: The AnnData object.
        embedding: The UMAP embedding coordinates.
        jacobxall: The expression-weighted Jacobian in gene space.
        celltypes: A list of cell types to plot.
        n_features: Number of top genes to annotate.
        stat: The statistic to use for aggregation ('mean' or 'median').

    Returns:
        A dictionary mapping cell types to their top features.
    """
    print("🎨 Plotting feature importances for cell types...")
    genes = adata.to_df().columns.values
    cv = adata.obs.cell_type.astype('category').cat.codes
    class_features, class_genesorted = {}, {}

    for cell_type in celltypes:
        print(f"\nAnalyzing cell type: {cell_type}")

        is_cell_type = adata.obs["cell_type"] == cell_type
        
        if showPlot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: UMAP with highlighted cell type
            is_cell_type = adata.obs["cell_type"] == cell_type
            ax1.scatter(embedding[:, 0], embedding[:, 1], c = cv, cmap='tab20', s=2, alpha=0.2)
            ax1.scatter(embedding[is_cell_type, 0], embedding[is_cell_type, 1], s=5, label=cell_type)
            ax1.set_title(f"UMAP with '{cell_type}' Highlighted")
            ax1.set_xlabel("UMAP 1")
            ax1.set_ylabel("UMAP 2")
            ax1.legend()

        # Plot 2: Feature importance plot
        jx0 = jacobxall[:, 0, :]
        jx1 = jacobxall[:, 1, :]
        
        if stat == 'median':
            jx0_agg = np.median(jx0[is_cell_type, :], axis=0)
            jx1_agg = np.median(jx1[is_cell_type, :], axis=0)
        else: # Default to mean
            jx0_agg = np.mean(jx0[is_cell_type, :], axis=0)
            jx1_agg = np.mean(jx1[is_cell_type, :], axis=0)
        if showPlot:
            ax2.scatter(jx0_agg, jx1_agg, s=1, alpha=0.7)
            ax2.set_title(f"Mean Feature Contributions for '{cell_type}'")
            ax2.set_xlabel("Contribution to UMAP 1")
            ax2.set_ylabel("Contribution to UMAP 2")
            ax2.axhline(0, color='grey', lw=0.5)
            ax2.axvline(0, color='grey', lw=0.5)
        
        # Annotate top features
        magnitude = np.sqrt(jx0_agg**2 + jx1_agg**2)
        idx_sorted = np.argsort(magnitude)[::-1]
        
        top_genes = []
        for i in idx_sorted[:n_features]:
            if showPlot:
                ax2.annotate(genes[i], (jx0_agg[i], jx1_agg[i]))
            top_genes.append(genes[i])
        class_genesorted[cell_type] = top_genes
        
        if showPlot:
            plt.tight_layout()
            plt.show()
        
    # return class_genesorted

        class_features[cell_type]=[jx0_agg[idx_sorted[:n_features]],jx1_agg[idx_sorted[:n_features]]]
        class_genesorted[cell_type]=[idx_sorted[:n_features], class_genesorted[cell_type], magnitude[idx_sorted[:n_features]]]
        #     # pdf.savefig()
        #     # plt.close()
    return class_features,class_genesorted

def export_results(class_genesorted, filename: str):
    """Saves the feature importance results to a CSV file."""
    print(f" Exporting results to {filename}...")
    df = pd.DataFrame.from_dict(class_genesorted, orient='index')
    df.to_csv(filename)
    print(" Export complete.")

# --- Main Execution Block ---

# def main():
#     """
#     Main function to run the end-to-end workflow.
#     """
    # # --- Configuration ---
    # config = {
    #     "seed": 24,
    #     "pca_components": 50,
    #     "network_class": "deepReLUNet", # or "bfbilinear"
    #     "train_params": {
    #         "lr": 8e-4,
    #         "epochs": 96,
    #         "batch_size": 1024,
    #         "pumap_kwargs": {"non_parametric_embeddings": True}
    #     },
    #     "jacobian_batch_size": 40,
    #     "plot_params": {
    #         "n_features": 16,
    #         "stat": "mean"
    #     },
    #     "output_filename": "bone_marrow_features.csv"
    # }
    
    # --- Workflow ---
#     setup_environment(config["seed"])

    # Load pre-processed AnnData object (assumed to exist)
    # adata = sc.read_h5ad("path/to/your/adata.h5ad")
    # For demonstration, we assume `adata` is already in the environment
    # if 'adata' not in locals():
    #     print("Error: AnnData object `adata` not found. Please load it before running.")
    #     return

    # # Select cell types to analyze
    # config["plot_params"]["celltypes"] = adata.obs["cell_type"].value_counts().index[:5].tolist()

    # # Step 1: Prepare data
    # train_dataset, adata_mean_zero = prepare_data(adata, config["pca_components"])
    
    # # Step 2: Define and train model
    # # Note: `network` must be an instantiated model class
    # network = deepReLUNet() # Replace with your model instantiation
    # encoder = train_parametric_umap(network, train_dataset, config["train_params"])
    
    # # Step 3: Generate and plot embedding
    # embedding = generate_and_plot_embedding(encoder, adata, train_dataset)
    
    # # Step 4: Compute Jacobian in gene space
    # jacobxall = compute_gene_space_jacobian(encoder.encoder, adata, train_dataset, adata_mean_zero, config["jacobian_batch_size"])
    
    # # Step 5: Plot feature importances
    # class_genesorted = plot_feature_importance(adata, embedding, jacobxall, **config["plot_params"])
    
    # # Step 6: Export results
    # export_results(class_genesorted, config["output_filename"])

# if __name__ == "__main__":
#     # To run this script, ensure you have an AnnData object named `adata`
#     # and the required custom classes (PUMAP, deepReLUNet) available.
#     # main()
#     print("Refactored script loaded. Call main() to execute the workflow.")
#     print("Ensure the `adata` object is loaded and custom classes are defined.")