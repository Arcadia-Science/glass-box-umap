from pathlib import Path
from typing import Callable

import pytest
import torch
from glass_box_umap.parametric_umap import ParametricUMAP
from torch import Tensor


def _fit_and_save(mnist_images: Tensor, tmp_path: Path) -> Path:
    model = ParametricUMAP(random_state=0, epochs=1, pca_components=8, quiet=True)
    model.fit(mnist_images)
    p = tmp_path / "m.pt"
    model.save(p)
    return p


def test_save_load_round_trip_equates(mnist_images: Tensor, tmp_path: Path) -> None:
    p = _fit_and_save(mnist_images, tmp_path)
    a = ParametricUMAP.load(p)
    b = ParametricUMAP.load(p).to("cpu")
    assert a == b


@pytest.mark.parametrize(
    "field, mutate, should_remain_equal",
    [
        ("quiet", lambda v: not v, True),
        ("num_workers", lambda v: v + 4, True),
        ("checkpoint_dir", lambda v: Path("/tmp/elsewhere"), True),
        ("extra_callbacks", lambda v: [object()], True),
        ("lr", lambda v: v + 1.0, False),
        ("epochs", lambda v: v + 1, False),
        ("random_state", lambda v: 999, False),
        ("n_neighbors", lambda v: v + 1, False),
        ("pca_components", lambda v: v - 1, False),
    ],
)
def test_eq_field_extent(
    mnist_images: Tensor,
    tmp_path: Path,
    field: str,
    mutate: Callable[[object], object],
    should_remain_equal: bool,
) -> None:
    p = _fit_and_save(mnist_images, tmp_path)
    a = ParametricUMAP.load(p)
    b = ParametricUMAP.load(p)
    setattr(b, field, mutate(getattr(b, field)))
    assert (a == b) is should_remain_equal


def test_eq_breaks_on_weight_mutation(mnist_images: Tensor, tmp_path: Path) -> None:
    p = _fit_and_save(mnist_images, tmp_path)
    a = ParametricUMAP.load(p)
    b = ParametricUMAP.load(p)
    assert a == b

    with torch.no_grad():
        next(b._fitted_model.parameters()).add_(1.0)
    assert a != b
