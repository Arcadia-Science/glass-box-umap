import logging
import warnings
from collections.abc import Generator
from contextlib import contextmanager

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
