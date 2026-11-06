from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import Sampler

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

    def sample(self) -> Tensor:
        """Draw one epoch's indices as a single int64 tensor."""
        chunks = []
        for start, end, n_samp in zip(
            self._chunk_starts, self._chunk_ends, self._samples_per_chunk, strict=True
        ):
            local = torch.multinomial(self.weights[start:end], n_samp, replacement=True)
            chunks.append(local + start)
        return torch.cat(chunks)

    def __iter__(self) -> Iterator[int]:
        yield from iter(self.sample().tolist())

    def __len__(self) -> int:
        return self.num_samples
