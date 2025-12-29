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

from ..utils import device_to_lightning_acceleration_config, get_default_device
from .data import UMAPDataset
from .graph import get_umap_graph
from .lightning import UMAPDataModule, UMAPLightningModule
from .registry import create_encoder


@dataclass
class ParametricUMAP:
    # UMAP config
    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2
    random_state: int | None = None
    encoder_name: str = "default"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)

    # Train config
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 512
    negative_sample_rate: int = 5
    repulsion_strength: float = 3.0
    num_workers: int = 0
    checkpoint_dir: Path | None = None
    checkpoint_every_n_epochs: int = 5

    _model: UMAPLightningModule | None = field(init=False, default=None)
    _device: torch.device = field(init=False, default_factory=get_default_device)

    @property
    def _fitted_model(self) -> UMAPLightningModule:
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `fit` first.")
        return self._model

    def _build_model(self, input_dims: tuple[int, ...]) -> UMAPLightningModule:
        """Lazy builder for the underlying Lightning Module."""
        encoder = create_encoder(
            name=self.encoder_name,
            input_dims=input_dims,
            n_components=self.n_components,
            encoder_kwargs=self.encoder_kwargs,
        )

        model = UMAPLightningModule(
            lr=self.lr,
            encoder=encoder,
            input_dims=input_dims,
            min_dist=self.min_dist,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
        ).to(self._device)

        return model

    def to(self, device: str | torch.device) -> ParametricUMAP:
        """Move the model (if initialized) and update the target device."""
        self._device = torch.device(device)
        if self._model is not None:
            self._model.to(self._device)
        return self

    def fit(self, X: Tensor) -> ParametricUMAP:
        if self.random_state is not None:
            pl.seed_everything(self.random_state, workers=True)

        callbacks: list[pl.Callback] = []
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    every_n_epochs=self.checkpoint_every_n_epochs,
                    filename="checkpoint_epoch_{epoch:03d}",
                )
            )

        accelerator, devices = device_to_lightning_acceleration_config(self._device)

        # NOTE: Advanced training configuration (DDP, half-precision, etc.) is not
        # currently exposed to the user. To address this, we could expose a
        # `trainer_kwargs` to `fit` or `__init__`. Alternatively, we could provide a
        # backdoor for power users with a `trainer: pl.Trainer` arg in `fit` for
        # complete control/overriding.
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=self.epochs,
            callbacks=callbacks or None,
            enable_checkpointing=self.checkpoint_dir is not None,
            logger=self.checkpoint_dir is not None,
        )

        input_dims = tuple(X.shape[1:])
        self._model = self._build_model(input_dims)

        graph = get_umap_graph(
            X.detach().cpu().numpy(),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
        )
        datamodule = UMAPDataModule(
            UMAPDataset(X.detach().cpu().numpy(), graph),
            self.batch_size,
            self.num_workers,
        )

        trainer.fit(model=self._model, datamodule=datamodule)

        self._model.to(self._device)

        return self

    @torch.no_grad()
    def transform(self, X: Tensor, batch_size: int | None = None) -> NDArray[np.floating]:
        self._fitted_model.eval()

        if next(self._fitted_model.parameters()).device != self._device:
            self._fitted_model.to(self._device)

        if batch_size is None:
            batch_size = self.batch_size

        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch = batch.to(self._device)
            embedding = self._fitted_model.encoder(batch)
            results.append(embedding.detach().cpu())

        return torch.cat(results).numpy()

    def fit_transform(self, X: Tensor) -> NDArray[np.floating]:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path) -> None:
        attrs = asdict(self)
        del attrs["_model"]
        del attrs["_device"]

        state = {
            "attrs": attrs,
            "state_dict": self._fitted_model.state_dict(),
            "input_dims": self._fitted_model.input_dims,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> ParametricUMAP:
        checkpoint = torch.load(path, map_location="cpu")

        instance = cls(**checkpoint["attrs"])
        instance._model = instance._build_model(checkpoint["input_dims"])
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.to(instance._device)

        return instance
