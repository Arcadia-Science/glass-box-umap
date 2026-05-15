import importlib.util
import logging
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

_LIGHTNING_LOGGER_NAMES = (
    "pytorch_lightning",
    "lightning_fabric",
    "lightning_utilities",
)


class _SuppressTrainerFitStopped(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Trainer.fit` stopped" not in record.getMessage()


logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(_SuppressTrainerFitStopped())
warnings.filterwarnings("ignore", message=".*does not have many workers.*")


def _is_notebook() -> bool:
    if importlib.util.find_spec("IPython") is None:
        return False
    from IPython.core.getipython import get_ipython

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


class _EpochFractionProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        super().on_train_epoch_start(trainer, *_)
        bar = self.train_progress_bar
        if bar.total is not None and getattr(bar, "container", None) is not None:
            bar.container.children[1].max = bar.total
        total = trainer.max_epochs or "?"
        bar.set_description(f"Epoch {trainer.current_epoch + 1}/{total}")


def get_progress_bar() -> TQDMProgressBar | None:
    if _is_notebook():
        return _EpochFractionProgressBar()
    return None


@contextmanager
def suppress_lightning_logs() -> Generator[None, None, None]:
    """Temporarily set all Lightning loggers to CRITICAL to suppress info messages."""
    loggers = [logging.getLogger(name) for name in _LIGHTNING_LOGGER_NAMES]
    old_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for logger, level in zip(loggers, old_levels, strict=True):
            logger.setLevel(level)
