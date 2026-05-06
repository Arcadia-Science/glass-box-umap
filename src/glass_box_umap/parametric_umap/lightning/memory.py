from typing import Any

import pytorch_lightning as pl
from torch import Tensor

from ...utils import current_device_memory_bytes


class MemoryLoggerCallback(pl.Callback):
    """Logs per-step device memory usage as ``mem_mb``.

    No-op on CPU, where there is no PyTorch-tracked allocation to report.
    """

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        bytes_alloc = current_device_memory_bytes(pl_module.device)
        if bytes_alloc is None:
            return
        pl_module.log(
            "mem_mb",
            bytes_alloc / 1024**2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
