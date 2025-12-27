from __future__ import annotations
import random

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


def get_default_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
