from typing import Any

import psutil
import pytorch_lightning as pl
from torch import Tensor

from ...utils import current_device_memory_bytes


class MemoryLoggerCallback(pl.Callback):
    """Logs per-step memory usage.

    Reports two metrics:
        - ``device_mem_mb``: bytes held by live tensors in PyTorch's allocator
          on the active device (CUDA VRAM, MPS unified RAM). Omitted on CPU.
        - ``rss_mb``: process-wide resident set size, capturing the dataset,
          graph arrays, model state, and Python overhead in addition to
          tracked tensors.

    On MPS the two metrics overlap (unified memory means tensor allocations
    also count toward RSS), so ``rss_mb >= device_mem_mb`` and they don't
    sum to anything meaningful.
    """

    def __init__(self) -> None:
        super().__init__()
        self._process = psutil.Process()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        device_bytes = current_device_memory_bytes(pl_module.device)
        if device_bytes is not None:
            pl_module.log(
                "device_mem_mb",
                device_bytes / 1024**2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        pl_module.log(
            "rss_mb",
            self._process.memory_info().rss / 1024**2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
