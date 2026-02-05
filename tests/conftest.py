from pathlib import Path

import pytest
import torch
from torch import Tensor

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mnist_images() -> Tensor:
    return torch.load(FIXTURES_DIR / "mnist_images.pt", weights_only=True)


@pytest.fixture
def mnist_labels() -> Tensor:
    return torch.load(FIXTURES_DIR / "mnist_labels.pt", weights_only=True)
