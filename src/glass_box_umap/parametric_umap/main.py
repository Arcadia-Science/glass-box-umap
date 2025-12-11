from collections.abc import Callable
from pathlib import Path

import dill
import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from umap import UMAP
from umap.umap_ import find_ab_params

from glass_box_umap.parametric_umap.data import MatchDataset, UMAPDataset
from glass_box_umap.parametric_umap.model import DefaultDecoder, DefaultEncoder
from glass_box_umap.parametric_umap.modules import get_umap_graph, umap_loss
from glass_box_umap.utils import get_accelerator


class PeriodicCheckpoint(pl.Callback):
    """Save checkpoints at regular epoch intervals.

    Args:
        save_dir: Directory to save checkpoint files.
        every_n_epochs: Save a checkpoint every N epochs.
        filename_template: Template for checkpoint filenames. Use {epoch} placeholder.
    """

    def __init__(
        self,
        save_dir: Path,
        every_n_epochs: int = 5,
        filename_template: str = "checkpoint_epoch_{epoch:03d}.ckpt",
    ) -> None:
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.filename_template = filename_template
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            filepath = self.save_dir / self.filename_template.format(epoch=epoch)
            trainer.save_checkpoint(filepath)


class UMAPLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training parametric UMAP.

    Supports two training modes:
    1. Standard UMAP training using edge-based loss
    2. Matching mode to reproduce non-parametric UMAP embeddings

    Args:
        lr: Learning rate for the AdamW optimizer.
        encoder: Neural network encoder mapping data to embedding space.
        decoder: Optional decoder for reconstruction loss.
        beta: Weight for reconstruction loss term.
        min_dist: UMAP min_dist parameter for computing a and b.
        reconstruction_loss: Loss function for reconstruction (if decoder provided).
        match_nonparametric_umap: If True, train to match non-parametric embeddings.
    """

    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        beta: float = 1.0,
        min_dist: float = 0.1,
        reconstruction_loss: Callable[[Tensor, Tensor], Tensor] = binary_cross_entropy_with_logits,
        match_nonparametric_umap: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        if not self.match_nonparametric_umap:
            edges_to_exp, edges_from_exp = batch
            embedding_to = self.encoder(edges_to_exp)
            embedding_from = self.encoder(edges_from_exp)

            encoder_loss = umap_loss(
                embedding_to,
                embedding_from,
                self._a,
                self._b,
                edges_to_exp.shape[0],
                negative_sample_rate=5,
                device=self.device,
            )
            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self.decoder is not None:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss

            return encoder_loss

        data, embedding = batch
        embedding_parametric = self.encoder(data)
        encoder_loss = mse_loss(embedding_parametric, embedding)
        self.log("encoder_loss", encoder_loss, prog_bar=True)

        if self.decoder is not None:
            recon = self.decoder(embedding_parametric)
            recon_loss = self.reconstruction_loss(recon, data)
            self.log("recon_loss", recon_loss, prog_bar=True)
            return encoder_loss + self.beta * recon_loss

        return encoder_loss


class UMAPDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for UMAP training.

    Args:
        dataset: PyTorch dataset providing training samples.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
    """

    def __init__(
        self,
        dataset: UMAPDataset | MatchDataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )


class PUMAP:
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
        match_nonparametric_umap: If True, train to match non-parametric embeddings.
        nonparametric_embeddings: Pre-computed embeddings to match (if match mode).
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
        match_nonparametric_umap: bool = False,
        nonparametric_embeddings: NDArray[np.floating] | None = None,
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
        self.match_nonparametric_umap = match_nonparametric_umap
        self.nonparametric_embeddings = nonparametric_embeddings
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        self._accelerator = get_accelerator()
        self.model: UMAPLightningModule

    def fit(self, X: Tensor) -> "PUMAP":
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

        if not self.match_nonparametric_umap:
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
        else:
            embeddings = self.nonparametric_embeddings
            if embeddings is None:
                non_parametric_umap = UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    n_components=self.n_components,
                    random_state=self.random_state,
                    verbose=True,
                )
                embeddings = non_parametric_umap.fit_transform(torch.flatten(X, 1, -1).numpy())
                self.nonparametric_embeddings = embeddings

            self.model = UMAPLightningModule(
                self.lr,
                encoder,
                decoder,
                beta=self.beta,
                reconstruction_loss=self.reconstruction_loss,
                match_nonparametric_umap=True,
            )
            datamodule = UMAPDataModule(
                MatchDataset(X.numpy(), embeddings),
                self.batch_size,
                self.num_workers,
            )
            trainer.fit(model=self.model, datamodule=datamodule)

        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> "NDArray[np.floating]":
        """Transform data to the embedding space.

        Args:
            X: Input data tensor of shape (n_samples, ...).

        Returns:
            Numpy array of embeddings with shape (n_samples, n_components).
        """
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z: Tensor) -> "NDArray[np.floating]":
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


def load_pumap(path: Path) -> PUMAP:
    """Load a PUMAP model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        The loaded PUMAP model.
    """
    with path.open("rb") as f:
        return dill.load(f)
