from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from ..utils import get_default_device
from .data import UMAPDataset
from .graph import get_umap_graph
from .lightning import UMAPDataModule, UMAPLightningModule
from .registry import create_encoder


@dataclass
class UMAPConfig:
    """Parameters specific to the UMAP algorithm and Encoder architecture."""

    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2
    random_state: int | None = None
    encoder_name: str = "default"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Parameters specific to the training loop."""

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 0
    checkpoint_dir: Path | None = None
    checkpoint_every_n_epochs: int = 5


@dataclass
class ParametricUMAP:
    umap_config: UMAPConfig = field(default_factory=UMAPConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)

    _model: UMAPLightningModule | None = field(init=False, default=None)
    _device: torch.device = field(init=False, default_factory=get_default_device)
    _input_dims: tuple[int, ...] | None = field(init=False, default=None)

    @classmethod
    def create(
        cls,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_components: int = 2,
        random_state: int | None = None,
        encoder_name: str = "default",
        encoder_kwargs: dict[str, Any] | None = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        checkpoint_dir: Path | None = None,
        checkpoint_every_n_epochs: int = 5,
    ) -> ParametricUMAP:
        """Create a ParametricUMAP instance."""
        umap_config = UMAPConfig(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
            encoder_name=encoder_name,
            encoder_kwargs=encoder_kwargs or {},
        )

        train_config = TrainConfig(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        )

        return cls(umap_config=umap_config, train_config=train_config)

    def to(self, device: str | torch.device) -> ParametricUMAP:
        """Move the model (if initialized) and update the target device."""
        self._device = torch.device(device)
        if self._model is not None:
            self._model.to(self._device)
        return self

    def _build_model(self, input_dims: tuple[int, ...]) -> UMAPLightningModule:
        """Lazy builder for the underlying Lightning Module."""
        self._input_dims = input_dims
        encoder = create_encoder(
            name=self.umap_config.encoder_name,
            input_dims=input_dims,
            n_components=self.umap_config.n_components,
            encoder_kwargs=self.umap_config.encoder_kwargs,
        )

        model = UMAPLightningModule(
            lr=self.train_config.lr,
            encoder=encoder,
            min_dist=self.umap_config.min_dist,
        ).to(self._device)

        return model

    def fit(self, X: Tensor) -> ParametricUMAP:
        callbacks: list[pl.Callback] = []
        if self.train_config.checkpoint_dir is not None:
            self.train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.train_config.checkpoint_dir,
                    every_n_epochs=self.train_config.checkpoint_every_n_epochs,
                    filename="checkpoint_epoch_{epoch:03d}",
                )
            )

        trainer = pl.Trainer(
            accelerator=self._device.type,
            devices=1,
            max_epochs=self.train_config.epochs,
            callbacks=callbacks or None,
            enable_checkpointing=self.train_config.checkpoint_dir is not None,
            logger=self.train_config.checkpoint_dir is not None,
        )

        input_dims = tuple(X.shape[1:])
        if self._model is None:
            self._model = self._build_model(input_dims)

        graph = get_umap_graph(
            X.detach().cpu().numpy(),
            n_neighbors=self.umap_config.n_neighbors,
            metric=self.umap_config.metric,
            random_state=self.umap_config.random_state,
        )
        datamodule = UMAPDataModule(
            UMAPDataset(X.detach().cpu().numpy(), graph),
            self.train_config.batch_size,
            self.train_config.num_workers,
        )

        trainer.fit(model=self._model, datamodule=datamodule)

        self._model.to(self._device)

        return self

    @torch.no_grad()
    def transform(self, X: Tensor) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call 'fit' first.")

        self._model.eval()

        if next(self._model.parameters()).device != self._device:
            self._model.to(self._device)

        return self._model.encoder(X.to(self._device)).detach().cpu().numpy()

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an untrained model.")

        state = {
            "umap_config": asdict(self.umap_config),
            "train_config": asdict(self.train_config),
            "input_dims": self._input_dims,
            "state_dict": self._model.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> ParametricUMAP:
        checkpoint = torch.load(path, map_location="cpu")

        umap_config = UMAPConfig(**checkpoint["umap_config"])
        train_config = TrainConfig(**checkpoint["train_config"])

        instance = cls(umap_config=umap_config, train_config=train_config)
        instance._model = instance._build_model(checkpoint["input_dims"])
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.to(instance._device)

        return instance
