from __future__ import annotations
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import Self

from ..utils import get_default_device
from ._equality import parametric_umap_equal
from .data import UMAPDataset
from .graph import NeighborSearch, get_umap_graph
from .module import UMAPModule
from .registry import DEFAULT_ENCODER, create_encoder
from .training import seed_everything, train_umap

if TYPE_CHECKING:
    # Imported only where they're used: importing Lightning and scikit-learn takes seconds.
    import pytorch_lightning as pl
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import PCA

Trainer = Literal["auto", "torch", "lightning"]


def _to_numpy_float32(X: NDArray[np.floating] | Tensor) -> NDArray[np.float32]:
    if isinstance(X, Tensor):
        return X.detach().cpu().numpy()
    return cast(NDArray[np.float32], X.astype(np.float32))


@dataclass(eq=False, kw_only=True)
class ParametricUMAP:
    """Parametric UMAP model.

    Attributes:
        n_neighbors:
            Number of nearest neighbors used to construct the high-dimensional graph.
        min_dist:
            Minimum distance between points in the low-dimensional embedding.
        metric:
            Distance metric used for computing nearest neighbors.
        neighbor_search:
            How to find each sample's nearest neighbors. ``"nndescent"`` (the default)
            uses pynndescent's approximate search and umap-learn's fuzzy graph.
            ``"exact"`` computes exact neighbors with PyTorch and builds the same fuzzy
            graph in NumPy, so nothing is compiled with numba: the first fit in a new
            process is about 20 seconds faster, but maps differ slightly from
            ``"nndescent"``'s because approximate and exact neighbors differ. It supports
            ``metric="euclidean"`` only, and its time grows with the square of the
            number of samples. ``"auto"`` uses ``"exact"`` below 4,096 samples, as
            umap-learn does, and ``"nndescent"`` above.
        n_components:
            Dimensionality of the learned embedding.
        negative_sample_rate:
            Number of negative samples per positive edge in the UMAP loss.
        repulsion_strength:
            Weighting of the repulsive term in the UMAP loss.
        pca_components:
            Number of PCA components for input preprocessing. If ``None``, no PCA is
            applied. PCA requires 2D input ``(n_samples, n_features)``; leave this
            ``None`` when fitting on multi-dimensional data (e.g. images for a
            convolutional encoder).
        encoder_name:
            Name of the registered encoder architecture.
        encoder_kwargs:
            Additional keyword arguments passed to the encoder constructor.
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
        num_workers:
            Number of data loading workers. Used by the Lightning trainer only.
        checkpoint_dir:
            Directory for the Lightning trainer's checkpoints and TensorBoard logs.
            Setting it selects the Lightning trainer (see ``trainer``). If ``None``, a
            temporary directory is used.
        restore_best_weights:
            If ``True``, restore the model weights from the epoch with the
            lowest loss after training. If ``False``, keep the weights from
            the final epoch.
        random_state:
            Random seed for reproducibility. If ``None``, no seed is set.
        quiet:
            If ``True``, suppress Lightning logs and progress output.
        extra_callbacks:
            Additional Lightning callbacks to attach to the trainer. Adding any selects
            the Lightning trainer (see ``trainer``).
        trainer:
            Which training loop to use. ``"torch"`` is a plain PyTorch loop: it gives the
            same weights as ``"lightning"``, bit for bit, two to three times faster,
            with no logging and no files written. ``"lightning"`` trains with PyTorch
            Lightning and writes checkpoints and TensorBoard logs. ``"auto"`` (the
            default) uses ``"lightning"`` when ``extra_callbacks`` or ``checkpoint_dir``
            is set, and ``"torch"`` otherwise.
    """

    # UMAP algorithm
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2
    neighbor_search: NeighborSearch = "nndescent"
    negative_sample_rate: int = 5
    repulsion_strength: float = 1.0

    # Preprocessing & encoder
    pca_components: int | None = None
    encoder_name: str = DEFAULT_ENCODER
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)

    # Optimization
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 10_000
    num_batches: int | None = None

    # Training infra
    num_workers: int = 0
    checkpoint_dir: Path | None = None
    restore_best_weights: bool = True

    # Runtime
    random_state: int | None = None
    quiet: bool = False
    extra_callbacks: list[pl.Callback] = field(default_factory=list)
    trainer: Trainer = "auto"

    # Internal state
    _model: UMAPModule | None = field(init=False, default=None)
    _pca: PCA | None = field(init=False, default=None)
    _mean: NDArray[np.floating] | None = field(init=False, default=None)
    _device: torch.device = field(init=False, default_factory=get_default_device)

    def __eq__(self, other: ParametricUMAP) -> bool:
        """Semantic equality of ParametricUMAP objects.

        See Also:
            - :func:`parametric_umap_equal` for definition.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented

        return parametric_umap_equal(self, other)

    @property
    def _fitted_model(self) -> UMAPModule:
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `fit` first.")
        return self._model

    def _build_model(self, input_dims: tuple[int, ...]) -> UMAPModule:
        """Lazy builder for the underlying model."""
        encoder = create_encoder(
            name=self.encoder_name,
            input_dims=input_dims,
            n_components=self.n_components,
            encoder_kwargs=self.encoder_kwargs,
        )

        model = UMAPModule(
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

    def _resolve_trainer(self) -> Literal["torch", "lightning"]:
        lightning_only = bool(self.extra_callbacks) or self.checkpoint_dir is not None
        if self.trainer == "auto":
            return "lightning" if lightning_only else "torch"
        if self.trainer == "torch":
            if lightning_only:
                raise ValueError(
                    "extra_callbacks and checkpoint_dir need the Lightning trainer. "
                    "Use trainer='lightning' or trainer='auto'."
                )
            return "torch"
        if self.trainer == "lightning":
            return "lightning"
        raise ValueError(f"trainer must be 'auto', 'torch' or 'lightning', not {self.trainer!r}.")

    def fit(self, X: NDArray[np.floating] | Tensor) -> Self:
        trainer = self._resolve_trainer()

        if self.random_state is not None:
            if trainer == "lightning":
                import pytorch_lightning as pl

                pl.seed_everything(self.random_state, workers=True, verbose=not self.quiet)
            else:
                seed_everything(self.random_state)

        X = _to_numpy_float32(X)
        self._mean = X.mean(axis=0)
        assert self._mean is not None
        X = X - self._mean

        if self.pca_components is not None:
            from sklearn.decomposition import PCA

            self._pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self._pca.fit_transform(X).astype(np.float32)

        X = cast(NDArray[np.float32], X)
        input_dims = tuple(X.shape[1:])
        self._model = self._build_model(input_dims)

        if trainer == "lightning":
            self._fit_lightning(X)
        else:
            dataset = UMAPDataset(X, self._umap_graph(X), random_state=self.random_state)
            train_umap(
                self._model,
                dataset,
                epochs=self.epochs,
                batch_size=self.batch_size,
                device=self._device,
                num_batches=self.num_batches,
                restore_best_weights=self.restore_best_weights,
                progress=not self.quiet,
            )

        self._model.to(self._device)

        return self

    def _umap_graph(self, X: NDArray[np.float32]) -> csr_matrix:
        # NNDescent requires 2D (n_samples, n_features). Flatten any
        # higher-dim input (e.g. images for ConvEncoder) for graph
        # construction only; UMAPDataset still receives the original X.
        return get_umap_graph(
            X.reshape(X.shape[0], -1) if X.ndim > 2 else X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            quiet=self.quiet,
            neighbor_search=self.neighbor_search,
        )

    def _fit_lightning(self, X: NDArray[np.float32]) -> None:
        """Train with PyTorch Lightning, writing checkpoints and TensorBoard logs."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        from ..utils import device_to_lightning_acceleration_config
        from .lightning import MemoryLoggerCallback, UMAPDataModule, UMAPLightningModule
        from .logging_config import get_progress_bar, suppress_lightning_logs

        assert self._model is not None
        # Shares the encoder, so training updates self._model in place.
        lightning_module = UMAPLightningModule(
            lr=self.lr,
            encoder=self._model.encoder,
            input_dims=self._model.input_dims,
            min_dist=self.min_dist,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
        ).to(self._device)

        with ExitStack() as stack:
            if self.quiet:
                stack.enter_context(suppress_lightning_logs())

            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_dir = self.checkpoint_dir
            else:
                ckpt_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

            best_checkpoint = ModelCheckpoint(
                dirpath=ckpt_dir / "checkpoints",
                monitor="loss_epoch",
                mode="min",
                save_top_k=1,
                save_on_train_epoch_end=True,
                filename="best",
            )

            logger = TensorBoardLogger(save_dir=ckpt_dir, name="logs", version="")

            accelerator, devices = device_to_lightning_acceleration_config(self._device)

            progress_bar = None if self.quiet else get_progress_bar()
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                max_epochs=self.epochs,
                limit_train_batches=self.num_batches,
                callbacks=[
                    best_checkpoint,
                    MemoryLoggerCallback(),
                    *([progress_bar] if progress_bar is not None else []),
                    *self.extra_callbacks,
                ],
                enable_checkpointing=True,
                logger=logger,
                log_every_n_steps=1,
                enable_progress_bar=not self.quiet,
                enable_model_summary=not self.quiet,
            )

            datamodule = UMAPDataModule(
                UMAPDataset(X, self._umap_graph(X), random_state=self.random_state),
                self.batch_size,
                self.num_workers,
            )

            if not self.quiet:
                print(f"View progress with `tensorboard --logdir {logger.log_dir}`")

            trainer.fit(model=lightning_module, datamodule=datamodule)

            if self.restore_best_weights:
                best_ckpt = torch.load(best_checkpoint.best_model_path, map_location="cpu")
                lightning_module.load_state_dict(best_ckpt["state_dict"])

    @torch.no_grad()
    def transform(
        self, X: NDArray[np.floating] | Tensor, batch_size: int | None = None
    ) -> NDArray[np.floating]:
        was_training = self._fitted_model.training
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

        if was_training:
            self._fitted_model.train()

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
