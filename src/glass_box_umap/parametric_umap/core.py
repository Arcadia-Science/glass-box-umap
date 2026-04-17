import tempfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from torch import Tensor
from typing_extensions import Self

from ..utils import device_to_lightning_acceleration_config, get_default_device
from .data import UMAPDataset
from .graph import get_umap_graph
from .lightning import UMAPDataModule, UMAPLightningModule
from .registry import create_encoder


def _to_numpy_float32(X: NDArray[np.floating] | Tensor) -> NDArray[np.float32]:
    if isinstance(X, Tensor):
        return X.detach().cpu().numpy()
    return cast(NDArray[np.float32], X.astype(np.float32))


@dataclass
class ParametricUMAP:
    """Parametric UMAP model.

    Attributes:
        n_neighbors:
            Number of nearest neighbors used to construct the high-dimensional graph.
        min_dist:
            Minimum distance between points in the low-dimensional embedding.
        metric:
            Distance metric used for computing nearest neighbors.
        n_components:
            Dimensionality of the learned embedding.
        random_state:
            Random seed for reproducibility. If ``None``, no seed is set.
        encoder_name:
            Name of the registered encoder architecture.
        encoder_kwargs:
            Additional keyword arguments passed to the encoder constructor.
        pca_components:
            Number of PCA components for input preprocessing. If ``None``, no PCA is
            applied.
        lr:
            Learning rate for the optimizer.
        epochs:
            Number of training epochs.
        batch_size:
            Batch size for training and (default) inference.
        num_batches:
            Cap the number of batches per epoch. Useful for large graphs where
            a full pass would be prohibitively long. If ``None``, trains on
            all batches.
        negative_sample_rate:
            Number of negative samples per positive edge in the UMAP loss.
        repulsion_strength:
            Weighting of the repulsive term in the UMAP loss.
        num_workers:
            Number of data loading workers.
        checkpoint_dir:
            Directory for saving training checkpoints. If ``None``, a temporary
            directory is used.
        restore_best_weights:
            If ``True``, restore the model weights from the epoch with the
            lowest loss after training. If ``False``, keep the weights from
            the final epoch.
    """

    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2
    random_state: int | None = None
    encoder_name: str = "default"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)

    # Preprocessing
    pca_components: int | None = None

    # Train config
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 512
    num_batches: int | None = None
    negative_sample_rate: int = 5
    repulsion_strength: float = 3.0
    num_workers: int = 0
    checkpoint_dir: Path | None = None
    restore_best_weights: bool = False
    extra_callbacks: list[pl.Callback] = field(default_factory=list)

    _model: UMAPLightningModule | None = field(init=False, default=None)
    _pca: PCA | None = field(init=False, default=None)
    _mean: NDArray[np.floating] | None = field(init=False, default=None)
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

    def to(self, device: str | torch.device) -> Self:
        """Move the model (if initialized) and update the target device."""
        self._device = torch.device(device)
        if self._model is not None:
            self._model.to(self._device)
        return self

    def fit(self, X: NDArray[np.floating] | Tensor) -> Self:
        if self.random_state is not None:
            pl.seed_everything(self.random_state, workers=True)

        X = _to_numpy_float32(X)
        self._mean = X.mean(axis=0)
        assert self._mean is not None
        X = X - self._mean

        if self.pca_components is not None:
            self._pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self._pca.fit_transform(X).astype(np.float32)

        X = cast(NDArray[np.float32], X)
        input_dims = tuple(X.shape[1:])
        self._model = self._build_model(input_dims)

        with ExitStack() as stack:
            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_dir = self.checkpoint_dir
            else:
                ckpt_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

            best_checkpoint = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="umap_loss",
                mode="min",
                save_top_k=1,
                save_on_train_epoch_end=True,
                filename="best",
            )

            logger = TensorBoardLogger(save_dir=ckpt_dir, name="logs")

            accelerator, devices = device_to_lightning_acceleration_config(self._device)

            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                max_epochs=self.epochs,
                limit_train_batches=self.num_batches,
                callbacks=[best_checkpoint, *self.extra_callbacks],
                enable_checkpointing=True,
                logger=logger,
                log_every_n_steps=1,
            )

            graph = get_umap_graph(
                X,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                random_state=self.random_state,
            )

            datamodule = UMAPDataModule(
                UMAPDataset(X, graph),
                self.batch_size,
                self.num_workers,
            )

            trainer.fit(model=self._model, datamodule=datamodule)

            if self.restore_best_weights:
                best_ckpt = torch.load(best_checkpoint.best_model_path, map_location="cpu")
                self._model.load_state_dict(best_ckpt["state_dict"])

        self._model.to(self._device)

        return self

    @torch.no_grad()
    def transform(
        self, X: NDArray[np.floating] | Tensor, batch_size: int | None = None
    ) -> NDArray[np.floating]:
        self._fitted_model.eval()

        if next(self._fitted_model.parameters()).device != self._device:
            self._fitted_model.to(self._device)

        assert self._mean is not None
        X = _to_numpy_float32(X) - self._mean

        if self._pca is not None:
            X = self._pca.transform(X).astype(np.float32)

        X = torch.from_numpy(X)

        if batch_size is None:
            batch_size = self.batch_size

        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch = batch.to(self._device)
            embedding = self._fitted_model.encoder(batch)
            results.append(embedding.detach().cpu())

        return torch.cat(results).numpy()

    def fit_transform(self, X: NDArray[np.floating] | Tensor) -> NDArray[np.floating]:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path) -> None:
        attrs = asdict(self)

        del attrs["_model"]
        del attrs["_pca"]
        del attrs["_mean"]
        del attrs["_device"]
        del attrs["extra_callbacks"]

        state = {
            "attrs": attrs,
            "state_dict": self._fitted_model.state_dict(),
            "input_dims": self._fitted_model.input_dims,
            "pca": self._pca,
            "mean": self._mean,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        init_field_names = {f.name for f in fields(cls) if f.init}
        attrs = {k: v for k, v in checkpoint["attrs"].items() if k in init_field_names}
        instance = cls(**attrs)
        instance._model = instance._build_model(checkpoint["input_dims"])
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.to(instance._device)
        instance._pca = checkpoint.get("pca")
        instance._mean = checkpoint.get("mean")

        return instance
