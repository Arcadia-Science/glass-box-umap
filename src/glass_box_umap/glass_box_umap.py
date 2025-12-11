import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from glass_box_umap.parametric_umap.main import PUMAP
from glass_box_umap.utils import get_accelerator


class LayerNormDetached(nn.Module):
    '''
    A LayerNorm implementation where the variance calculation is detached from the
    computation graph during evaluation, potentially stabilizing training.
    '''
    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for LayerNormDetached.'''
        mean = x.mean(dim=-1, keepdim=True)
        # Detach variance calculation during evaluation
        if not self.training:
            var = x.clone().detach().var(dim=-1, keepdim=True, unbiased=False)
        else:
            var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + 1e-12) # Added epsilon for stability
        return self.scale * norm_x

class deepReLUNet(nn.Module):
    """
    A deep neural network using PReLU activation and LayerNormDetached.
    """
    def __init__(self, input_size: int = 50, hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False), nn.PReLU(), LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(), LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(), LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(), LayerNormDetached(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PReLU network."""
        return self.model(x)

import random

# You may also need: import pytorch_lightning as pl

def set_global_seeds(seed: int):
    """Sets global seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: For the Pytorch Lightning trainer
    # pl.seed_everything(seed)

    # You might also want deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlassBoxUMAP:
    """Encapsulates parametric UMAP model fitting and feature attribution.

    This class follows a scikit-learn style API to initialize hyperparameters,
    fit the model using pre-processed data (e.g., PCA), and compute attributions
    using PCA data, gene expression, and PCA components.

    Attributes:
        n_components: The dimension of the space to embed into.
        n_neighbors: The size of local neighborhood (in terms of number of
            neighbors) used for manifold approximation.
        min_dist: The effective minimum distance between embedded points.
        repulsion_strength: Weighting applied to negative samples in low
            dimensional embedding optimization.
        n_fits: The number of models to fit (ensemble size).
        epochs: The number of training epochs.
        lr: The learning rate for the optimizer.
        batch_size: The size of the training batches.
        random_state: The seed used by the random number generator.
        input_size: The size of the input feature vector.
        hidden_size: The size of the hidden layers in the network.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.3,
        repulsion_strength: float = 3.0,
        n_fits: int = 1,
        epochs: int = 64,
        lr: float = 1e-4,
        batch_size: int = 2048,
        random_state: int = 12,
        input_size: int = 50,
        hidden_size: int = 1152,  # 1024 + 128
    ) -> None:
        """Initializes the model with all hyperparameters.

        Args:
            n_components: The dimension of the space to embed into.
            n_neighbors: The size of local neighborhood.
            min_dist: The effective minimum distance between embedded points.
            repulsion_strength: Weighting applied to negative samples.
            n_fits: Number of models to fit.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
            random_state: Random seed.
            input_size: Input feature dimension.
            hidden_size: Hidden layer dimension.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.repulsion_strength = repulsion_strength
        self.n_fits = n_fits
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Internal state
        self._models: list[Any] = []
        self._embeddings: list[np.ndarray] = []
        self._jacobians: list[torch.Tensor] = []
        self._feature_contributions: list[np.ndarray] = []
        self._train_data: torch.Tensor | None = None
        self._device = get_accelerator()

    def fit(
        self,
        features: np.ndarray,
        load_models: bool = False,
        load_n_fits: int = 1,
        save_models: bool = True,
        model_path_pattern: str = "models/umap_{i}.pth",
    ) -> "GlassBoxUMAP":
        """Fits the Parametric UMAP model to the input features.

        Args:
            features: The input data (e.g., PCA embeddings) with shape
                (n_samples, n_features).
            load_models: If True, skips training and loads pre-trained models
                from `model_path_pattern`.
            load_n_fits: Number of fits to load if `load_models` is True.
            save_models: If True, saves the trained model weights to
                `model_path_pattern` after fitting.
            model_path_pattern: A string pattern for the model file paths.
                Must contain `{i}` for formatting.

        Returns:
            The instance itself (self).

        Raises:
            FileNotFoundError: If `load_models` is True but a file is missing.
            RuntimeError: If there is an error loading the model state dict.
        """
        self._train_data = torch.tensor(features, dtype=torch.float32)
        self._models = []
        self._embeddings = []

        for i in range(self.n_fits):
            # Note: set_global_seeds is assumed to be available in scope
            set_global_seeds(2 * self.random_state + i)

            # Note: deepReLUNet is assumed to be available in scope
            network = deepReLUNet(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.n_components,
            )

            # Note: PUMAP is assumed to be available in scope
            pumap_model = PUMAP(
                encoder=network,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state + i,
                lr=self.lr,
                epochs=0 if load_models else self.epochs,
                batch_size=self.batch_size,
                num_workers=8,
            )

            model_file = model_path_pattern.format(i=i)

            if load_models:
                if i < load_n_fits:
                    try:
                        pumap_model.device = self._device
                        pumap_model.encoder.to(self._device)
                        # We must "fit" with 1 epoch/step to initialize graph
                        pumap_model.fit(self._train_data)

                        state_dict = torch.load(
                            model_file, map_location=self._device
                        )
                        pumap_model.encoder.load_state_dict(state_dict)
                        pumap_model.encoder.eval()

                        self._models.append(pumap_model)
                        embedding = pumap_model.transform(self._train_data)
                        self._embeddings.append(embedding)

                    except FileNotFoundError:
                        print(f"Error: Model file not found at {model_file}")
                        raise
                    except Exception as e:
                        print(f"Error loading state dict for model {i}: {e}")
                        raise RuntimeError(f"Failed to load model {i}") from e
            else:
                pumap_model.fit(self._train_data)

                if save_models:
                    os.makedirs(os.path.dirname(model_file), exist_ok=True)
                    torch.save(pumap_model.encoder.state_dict(), model_file)

                self._models.append(pumap_model)
                embedding = pumap_model.transform(self._train_data)
                self._embeddings.append(embedding)

        return self

    def transform(
        self, features: np.ndarray, fit_index: int = 0
    ) -> np.ndarray:
        """Transforms new data into the embedding space using a trained model.

        Args:
            features: New data to transform.
            fit_index: Index of the model to use for transformation.

        Returns:
            The UMAP embedding as a numpy array.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            IndexError: If `fit_index` is out of bounds.
        """
        if not self._models:
            raise RuntimeError("The model must be fitted before transforming.")
        if fit_index >= len(self._models):
            raise IndexError("fit_index is out of bounds.")

        features_tensor = torch.tensor(features, dtype=torch.float32)
        return self._models[fit_index].transform(features_tensor)

    def fit_transform(self, features: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Fits the model to features and returns the embedding.

        Args:
            features: The input data (e.g., PCA embeddings).
            **kwargs: Additional arguments passed to self.fit().

        Returns:
            The UMAP embedding for the first fit (fit_index=0).
        """
        self.fit(features, **kwargs)
        return self._embeddings[0]

    def compute_attributions(
        self,
        centered_gene_expression: np.ndarray,
        pca_components: np.ndarray,
        jacobian_batch_size: int = 40,
    ) -> "GlassBoxUMAP":
        """Computes the Jacobian and projects it to the original gene space.

        Args:
            centered_gene_expression: Mean-centered gene expression data with
                shape (n_samples, n_genes).
            pca_components: The PCA loading matrix (e.g., adata.varm["PCs"])
                with shape (n_genes, n_pcs).
            jacobian_batch_size: Batch size for Jacobian calculation.

        Returns:
            The instance itself (self).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._models:
            raise RuntimeError(
                "Model must be fitted before computing contributions."
            )
        if self._train_data is None:
            raise RuntimeError(
                "Internal training data not set. Please call fit() first."
            )

        self._feature_contributions = []
        self._jacobians = []

        for model in self._models:
            encoder = model.encoder
            encoder.eval()

            # 1. Compute Jacobian in batches (in PCA space)
            num_samples = self._train_data.shape[0]
            jacobians_pca_list = []

            for j in range(0, num_samples, jacobian_batch_size):
                data_batch = self._train_data[j : j + jacobian_batch_size, :]

                # Output shape: (batch, n_components, batch, n_features)
                jac_batch = torch.autograd.functional.jacobian(
                    encoder, data_batch, vectorize=True, strategy="reverse-mode"
                )
                # Un-fuse the vectorized diagonal output.
                # Project: (batch, n_components, n_features)
                jac_batch_unfused = torch.einsum("bibj->bij", jac_batch)
                jacobians_pca_list.append(jac_batch_unfused.detach().cpu())

            jacobians_pca_tensor = torch.cat(jacobians_pca_list, dim=0)

            # 2. Project Jacobian from PCA space back to gene space
            # J_gene[i, emb, gene] = J_pca[i, emb, pc] * PCs[gene, pc]
            gene_space_jacobian = torch.einsum(
                "bij,kj->bik",
                jacobians_pca_tensor,
                torch.tensor(pca_components, dtype=torch.float32),
            )

            # 3. Weight by each cell's mean-centered gene expression
            feature_contributions = (
                gene_space_jacobian.numpy()
                * centered_gene_expression[:, np.newaxis, :]
            )

            # Cast to float16 for memory efficiency
            feature_contributions = feature_contributions.astype("float16")

            self._feature_contributions.append(feature_contributions)
            self._jacobians.append(jacobians_pca_tensor)

        return self

    def get_feature_importance(
        self, adata: Any, groupby: str, gene_names: np.ndarray
    ) -> pd.DataFrame:
        """Aggregates feature contributions by a specified group.

        Args:
            adata: The AnnData object, needed for .obs groupings.
            groupby: The column in adata.obs to group by (e.g., 'cell_type').
            gene_names: An array of gene names.

        Returns:
            A DataFrame with mean and SEM contributions for each gene in
            each group.

        Raises:
            RuntimeError: If compute_attributions() has not been run.
        """
        if not self._feature_contributions:
            raise RuntimeError("Must run compute_attributions() first.")

        all_run_jacobians = np.array(self._feature_contributions)
        n_runs = all_run_jacobians.shape[0]
        # Assuming adata.obs[groupby] is categorical
        all_groups = adata.obs[groupby].cat.categories

        summary_dfs = []
        for group in all_groups:
            is_group_mask = (adata.obs[groupby] == group).values
            if np.sum(is_group_mask) == 0:
                continue

            # Shape: (n_runs, n_cells_in_group, n_dims, n_genes)
            jacobians_for_group = all_run_jacobians[:, is_group_mask, :, :]

            # Calculate magnitudes (L2 norm across UMAP dims)
            # Shape: (n_runs, n_cells_in_group, n_genes)
            magnitudes = np.linalg.norm(jacobians_for_group, axis=2, ord=2)

            run_mean_contributions = []
            for run_idx in range(n_runs):
                run_mags = magnitudes[run_idx, :, :]  # (n_cells, n_genes)

                # Normalize each cell by its own total contribution
                cell_sums = np.sum(run_mags, axis=1, keepdims=True)
                normalized_mags = run_mags / (cell_sums + 1e-9)

                # Get mean contribution for each gene across cells *for this run*
                run_mean_contributions.append(np.mean(normalized_mags, axis=0))

            # Aggregate across all runs. Shape: (n_runs, n_genes)
            run_means_array = np.array(run_mean_contributions)

            # Final stats across runs
            mean_contributions = np.mean(run_means_array, axis=0)
            sem_contributions = np.std(run_means_array, axis=0) / np.sqrt(
                n_runs
            )

            df = pd.DataFrame(
                {
                    "gene": gene_names,
                    "mean_contribution": mean_contributions,
                    "sem_contribution": sem_contributions,
                    groupby: group,
                }
            )
            summary_dfs.append(df)

        return pd.concat(summary_dfs, ignore_index=True)

    def save_analysis_summary(
        self,
        adata: Any,
        groupby: str,
        basename: str = "analysis_summary",
    ) -> None:
        """Saves all necessary data for offline plotting and analysis.

        Args:
            adata: The AnnData object, needed for .obs groupings and gene names.
            groupby: The column in adata.obs to group by (e.g., 'cell_type').
            basename: The prefix for the three output files.

        Raises:
            RuntimeError: If fit() or compute_attributions() have not been run.
        """
        if (
            not self._feature_contributions
            or not self._embeddings
            or not self._jacobians
            or self._train_data is None
        ):
            raise RuntimeError(
                "Must run fit() and compute_attributions() first."
            )

        # 1. Save population-level statistics
        stats_df = self.get_feature_importance(
            adata, groupby, adata.var_names.values
        )
        stats_filename = f"{basename}_stats.csv"
        stats_df.to_csv(stats_filename, index=False)

        # 2. Save plot data (NPZ)
        mean_vector_dict = {}
        all_groups = adata.obs[groupby].cat.categories
        for group in all_groups:
            is_group_mask = (adata.obs[groupby] == group).values
            # Using the first run [0] for plotting data
            mean_vector_dict[group] = np.mean(
                self._feature_contributions[0][is_group_mask], axis=0
            )

        jacobxall_first_run = self._feature_contributions[0]
        jacobian_magnitude = np.linalg.norm(jacobxall_first_run, axis=1)

        jacobian_0 = self._jacobians[0]
        pca_data_0 = self._train_data.squeeze().detach().cpu().numpy()
        reconstruction_0 = np.einsum(
            "ijk,ik->ij", jacobian_0.numpy(), pca_data_0
        )

        plot_data_filename = f"{basename}_plot_data.npz"
        np.savez_compressed(
            plot_data_filename,
            embedding=self._embeddings[0],
            group_labels=adata.obs[groupby].values,
            group_by_key=groupby,  # Store the key name
            mean_jacobian_vectors=mean_vector_dict,
            jacobian_magnitude=jacobian_magnitude,
            gene_names=adata.var_names.values,
            jacobian_reconstruction=reconstruction_0,
        )

        # 3. Save interactive plot data
        interactive_df = self._prepare_plotly_df(
            adata, groupby=groupby, fit_index=0, top_n_genes=8
        )
        interactive_filename = f"{basename}_interactive.csv"
        interactive_df.to_csv(interactive_filename, index=False)

    def _prepare_plotly_df(
        self,
        adata: Any,
        groupby: str,
        fit_index: int = 0,
        top_n_genes: int = 8,
    ) -> pd.DataFrame:
        """Prepares a DataFrame for interactive plotting.

        Args:
            adata: The AnnData object.
            groupby: The grouping key.
            fit_index: The model index to use.
            top_n_genes: Number of top contributing genes to include.

        Returns:
            A pandas DataFrame suitable for Plotly.
        """
        embedding = self._embeddings[fit_index]
        jacobxall = self._feature_contributions[fit_index]

        df = pd.DataFrame(embedding, columns=["UMAP 0", "UMAP 1"])
        df[groupby] = adata.obs[groupby].values

        # Calculate squared distance and find top contributing genes
        gene_dist_sq = jacobxall[:, 0, :] ** 2 + jacobxall[:, 1, :] ** 2
        genes = adata.var.index.values

        # Sort indices descending
        top_gene_indices = np.argsort(gene_dist_sq, axis=1)[:, ::-1][
            :, :top_n_genes
        ]

        for i in range(top_n_genes):
            df[f"gene_{i}"] = genes[top_gene_indices[:, i]]

        return df
