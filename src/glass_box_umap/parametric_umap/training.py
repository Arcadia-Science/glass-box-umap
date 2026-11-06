"""A plain PyTorch training loop for parametric UMAP.

It trains the same way as the Lightning trainer in ``core.py``, which version 0.1.0 always
used, and draws random numbers in the same order. So for the same ``random_state`` on the
same machine it gives the same weights, bit for bit. What it leaves out is the overhead:
edges are gathered a batch at a time instead of one Python call per edge, and there is no
logger, no checkpoint file and no per-step memory logging. The best epoch's weights are kept
in memory.
"""

import math
import random

import numpy as np
import torch

from .data import UMAPDataset
from .module import UMAPModule
from .sampler import SplitAndMergeWeightedSampler

_MAX_SEED = 2**32 - 1


def seed_everything(seed: int) -> int:
    """Seed Python's, NumPy's and PyTorch's global generators.

    The same seeding as ``pytorch_lightning.seed_everything``, without importing Lightning.

    Args:
        seed: An integer from 0 to 2**32 - 1.

    Returns:
        The seed.
    """
    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"{seed} is not in bounds, numpy accepts from 0 to {_MAX_SEED}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def train_umap(
    module: UMAPModule,
    dataset: UMAPDataset,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_batches: int | None = None,
    restore_best_weights: bool = True,
    progress: bool = False,
) -> list[float]:
    """Train ``module`` on the edges of ``dataset``.

    Each epoch draws ``len(dataset)`` edges with replacement, weighted by edge weight, and
    takes one AdamW step per batch of ``batch_size`` edges. The last batch may be smaller.

    Args:
        module: The model to train, in place.
        dataset: The UMAP graph's edges and the data they connect.
        epochs: Number of passes.
        batch_size: Edges per optimizer step.
        device: Where to train.
        num_batches: If given, stop each epoch after this many batches.
        restore_best_weights: Load the weights from the epoch with the lowest mean loss
            when training ends. Otherwise keep the last epoch's.
        progress: Show a progress bar over epochs.

    Returns:
        Each epoch's mean loss, weighted by batch size.
    """
    weights = torch.as_tensor(dataset.edge_weights, dtype=torch.double)
    sampler = SplitAndMergeWeightedSampler(weights=weights, num_samples=len(dataset))
    steps = math.ceil(len(dataset) / batch_size)
    if num_batches is not None:
        steps = min(steps, num_batches)

    module.to(device)
    module.train()
    optimizer = module.configure_optimizer()

    epoch_iter: range = range(epochs)
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(epoch_iter, desc="Training", unit="epoch")
        epoch_iter = bar  # type: ignore[assignment]

    losses: list[float] = []
    best_loss: torch.Tensor | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for _ in epoch_iter:
        # A DataLoader draws one number from the global generator each time it starts an
        # epoch (its workers' base seed). Draw it too, so the edges sampled next, and so the
        # map, are the same as the Lightning trainer's.
        torch.empty((), dtype=torch.int64).random_()
        order = sampler.sample()

        # Track the epoch loss as Lightning's `self.log(..., on_epoch=True)` does: a float32
        # sum of each batch's loss times its size, divided by the number of edges seen.
        total = torch.zeros((), dtype=torch.float32)
        seen = 0
        for step in range(steps):
            idx = order[step * batch_size : (step + 1) * batch_size]
            edges_to, edges_from = dataset.edge_batch(idx)
            # Blocking copies: the gathered batches are temporaries, and an asynchronous
            # copy from them can read freed memory (seen as NaN weights on MPS).
            edges_to = edges_to.to(device)
            edges_from = edges_from.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = module.training_loss(edges_to, edges_from)
            loss.backward()
            optimizer.step()

            total = total + loss.detach() * idx.numel()
            seen += idx.numel()

        if seen == 0:
            continue

        epoch_loss = total / seen
        losses.append(float(epoch_loss))
        if bar is not None:
            bar.set_postfix(loss=f"{losses[-1]:.4f}")

        if restore_best_weights:
            # Lightning's ModelCheckpoint(monitor=..., mode="min", save_top_k=1): keep the
            # first epoch, then any epoch with a strictly lower loss. NaN counts as +inf.
            if torch.isnan(epoch_loss):
                epoch_loss = torch.tensor(float("inf"), device=epoch_loss.device)
            if best_loss is None or bool(epoch_loss < best_loss):
                best_loss = epoch_loss
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}

    if bar is not None:
        bar.close()

    if restore_best_weights and best_state is not None:
        module.load_state_dict(best_state)

    return losses
