from __future__ import annotations
import random
from typing import Literal

import numpy as np
import torch


def set_global_seeds(seed: int):
    """Sets global seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: For the Pytorch Lightning trainer
    # pl.seed_everything(seed)

    # You might also want deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_accelerator() -> Literal["cuda", "mps", "cpu"]:
    """Detect the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
