import pytest
import torch
from glass_box_umap.utils import device_to_lightning_acceleration_config


@pytest.mark.parametrize(
    ("device", "expected_accelerator", "expected_devices"),
    [
        (torch.device("cpu"), "cpu", "auto"),
        (torch.device("mps"), "mps", "auto"),
        (torch.device("cuda"), "cuda", 1),
        (torch.device("cuda:0"), "cuda", [0]),
        (torch.device("cuda:1"), "cuda", [1]),
        (torch.device("cuda:3"), "cuda", [3]),
    ],
)
def test_device_to_lightning_acceleration_config(
    device: torch.device,
    expected_accelerator: str,
    expected_devices: int | list[int] | str,
):
    accelerator, devices = device_to_lightning_acceleration_config(device)
    assert accelerator == expected_accelerator
    assert devices == expected_devices
