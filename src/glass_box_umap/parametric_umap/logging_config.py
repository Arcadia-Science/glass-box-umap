import logging
import warnings


class _SuppressTrainerFitStopped(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Trainer.fit` stopped" not in record.getMessage()


logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(_SuppressTrainerFitStopped())
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
