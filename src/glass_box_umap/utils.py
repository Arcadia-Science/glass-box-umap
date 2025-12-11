from typing import Literal

import torch


def get_accelerator() -> Literal["cuda", "mps", "cpu"]:
    """Detect the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
