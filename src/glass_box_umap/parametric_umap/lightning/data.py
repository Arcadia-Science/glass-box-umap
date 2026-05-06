import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data import UMAPDataset


class UMAPDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for UMAP training.

    Args:
        dataset: PyTorch dataset providing training samples.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
    """

    def __init__(
        self,
        dataset: UMAPDataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.dataset.edge_weights, dtype=torch.double),  # type: ignore
            num_samples=len(self.dataset),
            replacement=True,
        )
        return DataLoader(
            dataset=self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            # TODO: a tiny tail batch (e.g. 1 edge when num_edges = batch_size + 1)
            # produces a high-variance gradient step with the same LR weight as a
            # full batch. drop_last=True would fix this, but it also wipes out the
            # only batch when len(dataset) < batch_size (e.g. small test fixtures),
            # so we need a smarter approach (e.g. rounding num_samples up to a
            # multiple of batch_size in the sampler).
        )
