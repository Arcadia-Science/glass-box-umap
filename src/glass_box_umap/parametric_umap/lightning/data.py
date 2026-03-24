import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler

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
        num_batches: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        if self.num_batches > 0:
            sampler = RandomSampler(
                self.dataset, replacement=True, num_samples=self.num_batches * self.batch_size
            )

            return DataLoader(
                dataset=self.dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                persistent_workers=self.num_workers > 0,
            )
