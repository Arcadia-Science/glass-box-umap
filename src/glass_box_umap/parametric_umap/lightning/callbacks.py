from pathlib import Path

import pytorch_lightning as pl


class PeriodicCheckpoint(pl.Callback):
    """Save checkpoints at regular epoch intervals.

    Args:
        save_dir: Directory to save checkpoint files.
        every_n_epochs: Save a checkpoint every N epochs.
        filename_template: Template for checkpoint filenames. Use {epoch} placeholder.
    """

    def __init__(
        self,
        save_dir: Path,
        every_n_epochs: int = 5,
        filename_template: str = "checkpoint_epoch_{epoch:03d}.ckpt",
    ) -> None:
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.filename_template = filename_template
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            filepath = self.save_dir / self.filename_template.format(epoch=epoch)
            trainer.save_checkpoint(filepath)
