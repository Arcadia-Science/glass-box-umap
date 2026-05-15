import pytest
import torch
from glass_box_umap.parametric_umap.lightning import data as data_module
from glass_box_umap.parametric_umap.lightning.data import SplitAndMergeWeightedSampler


def test_multiple_chunks_when_over_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(data_module, "MULTINOMIAL_CATEGORY_LIMIT", 100)
    weights = torch.rand(250, dtype=torch.double)
    sampler = SplitAndMergeWeightedSampler(weights, num_samples=240)
    assert sampler._chunk_starts == [0, 100, 200]
    assert sampler._chunk_ends == [100, 200, 250]
    out = list(sampler)
    assert len(out) == 240
    assert min(out) >= 0
    assert max(out) < 250


def test_intra_chunk_weighting_respected():
    torch.manual_seed(0)
    weights = torch.full((1000,), 0.01, dtype=torch.double)
    weights[42] = 100.0
    sampler = SplitAndMergeWeightedSampler(weights, num_samples=10_000)
    counts: dict[int, int] = {}
    for i in sampler:
        counts[i] = counts.get(i, 0) + 1
    assert counts[42] > 8000


def test_each_chunk_contributes_its_allocated_share(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(data_module, "MULTINOMIAL_CATEGORY_LIMIT", 100)
    weights = torch.ones(250, dtype=torch.double)
    sampler = SplitAndMergeWeightedSampler(weights, num_samples=300)
    out = list(sampler)
    chunk_0 = sum(1 for i in out if 0 <= i < 100)
    chunk_1 = sum(1 for i in out if 100 <= i < 200)
    chunk_2 = sum(1 for i in out if 200 <= i < 250)
    assert chunk_0 == 100
    assert chunk_1 == 100
    assert chunk_2 == 100


def test_uneven_division_distributes_remainder_to_first_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(data_module, "MULTINOMIAL_CATEGORY_LIMIT", 100)
    weights = torch.rand(250, dtype=torch.double)
    sampler = SplitAndMergeWeightedSampler(weights, num_samples=242)
    assert sampler._samples_per_chunk == [81, 81, 80]
    assert sum(sampler._samples_per_chunk) == 242
    assert len(list(sampler)) == 242
