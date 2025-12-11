from __future__ import annotations
from collections.abc import Callable
from pathlib import Path

import dill
import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits

from ..utils import get_accelerator
from .data import UMAPDataset
from .graph import get_umap_graph
from .lightning import PeriodicCheckpoint, UMAPDataModule, UMAPLightningModule
from .model import DefaultDecoder, DefaultEncoder


class ParametricUMAP:
    """Parametric UMAP for learning embeddings with neural networks.

    This class provides a scikit-learn-like interface for training parametric
    UMAP models using PyTorch and PyTorch Lightning.

    Args:
        encoder: Custom encoder network. If None, uses DefaultEncoder.
        decoder: Custom decoder network, True for DefaultDecoder, None for no decoder.
        n_neighbors: Number of neighbors for UMAP graph construction.
        min_dist: UMAP min_dist parameter controlling embedding spread.
        metric: Distance metric for neighbor search.
        n_components: Dimensionality of the embedding space.
        beta: Weight for reconstruction loss when using a decoder.
        reconstruction_loss: Loss function for reconstruction.
        random_state: Random seed for reproducibility.
        lr: Learning rate for training.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        num_workers: Number of data loading workers.
        checkpoint_dir: Directory to save periodic checkpoints. None disables checkpointing.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | bool | None = None,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_components: int = 2,
        beta: float = 1.0,
        reconstruction_loss: Callable[[Tensor, Tensor], Tensor] = binary_cross_entropy_with_logits,
        random_state: int | None = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        checkpoint_dir: Path | None = None,
        checkpoint_every_n_epochs: int = 5,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        self._accelerator = get_accelerator()
        self.model: UMAPLightningModule

    def fit(self, X: Tensor) -> ParametricUMAP:
        """Fit the parametric UMAP model to data.

        Args:
            X: Input data tensor of shape (n_samples, ...).

        Returns:
            Self, for method chaining.
        """
        callbacks: list[pl.Callback] = []
        if self.checkpoint_dir is not None:
            callbacks.append(
                PeriodicCheckpoint(
                    save_dir=self.checkpoint_dir,
                    every_n_epochs=self.checkpoint_every_n_epochs,
                )
            )

        trainer = pl.Trainer(
            accelerator=self._accelerator,
            devices=1,
            max_epochs=self.epochs,
            callbacks=callbacks if callbacks else None,
        )

        dims = tuple(X.shape[1:])
        encoder = (
            self.encoder if self.encoder is not None else DefaultEncoder(dims, self.n_components)
        )

        decoder: nn.Module | None
        if self.decoder is None:
            decoder = None
        elif isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder is True:
            decoder = DefaultDecoder(dims, self.n_components)
        else:
            decoder = None

        self.model = UMAPLightningModule(
            self.lr,
            encoder,
            decoder,
            beta=self.beta,
            min_dist=self.min_dist,
            reconstruction_loss=self.reconstruction_loss,
        )
        graph = get_umap_graph(
            X.numpy(),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
        )
        datamodule = UMAPDataModule(
            UMAPDataset(X.numpy(), graph),
            self.batch_size,
            self.num_workers,
        )
        trainer.fit(model=self.model, datamodule=datamodule)

        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> NDArray[np.floating]:
        """Transform data to the embedding space.

        Args:
            X: Input data tensor of shape (n_samples, ...).

        Returns:
            Numpy array of embeddings with shape (n_samples, n_components).
        """
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z: Tensor) -> NDArray[np.floating]:
        """Reconstruct data from embeddings.

        Requires the model to have been trained with a decoder.

        Args:
            Z: Embedding tensor of shape (n_samples, n_components).

        Returns:
            Numpy array of reconstructed data.
        """
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path: Path) -> None:
        """Save the PUMAP model to disk.

        Args:
            path: File path for saving the model.
        """
        with path.open("wb") as f:
            dill.dump(self, f)


def load_pumap(path: Path) -> ParametricUMAP:
    """Load a PUMAP model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        The loaded PUMAP model.
    """
    with path.open("rb") as f:
        return dill.load(f)
