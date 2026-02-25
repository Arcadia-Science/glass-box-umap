from typing import cast

import torch
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry


def get_default_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_to_lightning_acceleration_config(
    device: torch.device,
) -> tuple[str | Accelerator, int | list[int] | str]:
    """Translates a PyTorch device into Lightning `accelerator` and `devices` arguments.

    Notes:
        - If you want multi-device acceleration, don't use this function (it's not
          supported).

    Behaviors:
        1. "cuda:n": Restricts training to ONLY that specific GPU
           (returns `devices=[n]`).
        2. "cuda": Defaults to `devices=1` (the first available GPU).
        3. "cpu"/"mps"/others: Defaults to "auto", allowing Lightning to manage
           platform-specific optimizations.

    Args:
        device: The target torch.device.

    Returns:
        A tuple of (accelerator, devices) ready to be unpacked into pl.Trainer.
    """
    accelerator = device.type
    index = cast(int | None, device.index)

    # Fall back to CPU if Lightning says the accelerator isn't available.
    if (
        accelerator in AcceleratorRegistry
        and not AcceleratorRegistry[accelerator]["accelerator"].is_available()
    ):
        accelerator = "cpu"

    # Default to "auto" for CPU, MPS, or other accelerators.
    devices: int | list[int] | str = "auto"

    if accelerator == "cuda":
        if index is not None:
            # Use specified GPU index.
            devices = [index]
        else:
            # Use single-device behavior.
            devices = 1

    return accelerator, devices
