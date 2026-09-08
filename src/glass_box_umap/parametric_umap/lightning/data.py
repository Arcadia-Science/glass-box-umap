from collections.abc import Iterator

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from ..data import UMAPDataset

MULTINOMIAL_CATEGORY_LIMIT = 2**24


class SplitAndMergeWeightedSampler(Sampler[int]):
    """Weighted-with-replacement sampler that bypasses torch.multinomial's 2**24 cap.

    Splits ``weights`` into contiguous chunks of at most ``2**24`` entries,
    runs ``torch.multinomial`` on each chunk independently, offsets each
    chunk's local indices into the global index space, and concatenates.

    Each chunk is allocated an equal share of ``num_samples``, so the
    cross-chunk allocation is only unbiased when the input weights are in
    random order. ``UMAPDataset`` permutes its edge arrays at construction
    time to satisfy this; if you reuse this sampler elsewhere, shuffle the
    weights once before passing them in.
    """

    def __init__(self, weights: Tensor, num_samples: int) -> None:
        self.weights = weights
        self.num_samples = num_samples
        n = weights.numel()
        chunk_size = MULTINOMIAL_CATEGORY_LIMIT
        n_chunks = (n + chunk_size - 1) // chunk_size
        self._chunk_starts = [i * chunk_size for i in range(n_chunks)]
        self._chunk_ends = [min(start + chunk_size, n) for start in self._chunk_starts]
        base, remainder = divmod(num_samples, n_chunks)
        self._samples_per_chunk = [base + (1 if i < remainder else 0) for i in range(n_chunks)]

    def sample_tensor(self) -> Tensor:
        """Draw all edge indices while retaining them as one tensor."""
        chunks = []
        for start, end, n_samp in zip(
            self._chunk_starts, self._chunk_ends, self._samples_per_chunk, strict=True
        ):
            local = torch.multinomial(self.weights[start:end], n_samp, replacement=True)
            chunks.append(local + start)
        return torch.cat(chunks)

    def __iter__(self) -> Iterator[int]:
        yield from iter(self.sample_tensor().tolist())

    def __len__(self) -> int:
        return self.num_samples


class SplitAndMergeWeightedBatchSampler(Sampler[Tensor]):
    """Yield sampled edge indices as tensors, one vectorized batch at a time."""

    def __init__(self, weights: Tensor, num_samples: int, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sampler = SplitAndMergeWeightedSampler(weights, num_samples)
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tensor]:
        sampled = self.sampler.sample_tensor()
        yield from sampled.split(self.batch_size)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def _identity_collate(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return batch


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
        batch_sampler = SplitAndMergeWeightedBatchSampler(
            weights=torch.as_tensor(self.dataset.edge_weights, dtype=torch.double),
            num_samples=len(self.dataset),
            batch_size=self.batch_size,
        )
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=_identity_collate,
            # TODO: a tiny tail batch (e.g. 1 edge when num_edges = batch_size + 1)
            # produces a high-variance gradient step with the same LR weight as a
            # full batch. drop_last=True would fix this, but it also wipes out the
            # only batch when len(dataset) < batch_size (e.g. small test fixtures),
            # so we need a smarter approach (e.g. rounding num_samples up to a
            # multiple of batch_size in the sampler).
        )
