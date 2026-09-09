import numpy as np
import pytest
import torch
from glass_box_umap.parametric_umap.data import UMAPDataset
from glass_box_umap.parametric_umap.lightning import data as data_module
from glass_box_umap.parametric_umap.lightning.data import (
    SplitAndMergeWeightedBatchSampler,
    SplitAndMergeWeightedSampler,
    UMAPDataModule,
)
from scipy.sparse import csr_matrix


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


def test_tensor_batch_sampler_matches_scalar_draws():
    weights = torch.linspace(0.1, 1.0, 25, dtype=torch.double)
    torch.manual_seed(42)
    scalar = list(SplitAndMergeWeightedSampler(weights, num_samples=53))
    torch.manual_seed(42)
    batched = list(SplitAndMergeWeightedBatchSampler(weights, num_samples=53, batch_size=8))
    assert torch.cat(batched).tolist() == scalar
    assert [len(batch) for batch in batched] == [8, 8, 8, 8, 8, 8, 5]


def test_vectorized_dataset_batch_matches_scalar_lookup():
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    rows = np.arange(20)
    cols = np.roll(rows, -1)
    graph = csr_matrix((np.ones(20), (rows, cols)), shape=(20, 20))
    dataset = UMAPDataset(X, graph, edge_pruning_factor=0.0, random_state=42)
    indices = torch.tensor([0, 3, 7, 11])
    expected_a, expected_b = zip(*(dataset[int(index)] for index in indices), strict=True)
    actual_a, actual_b = dataset.__getitems__(indices)
    torch.testing.assert_close(actual_a, torch.stack(expected_a))
    torch.testing.assert_close(actual_b, torch.stack(expected_b))


def test_data_module_uses_vectorized_fetch(monkeypatch):
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    rows = np.arange(20)
    graph = csr_matrix((np.ones(20), (rows, np.roll(rows, -1))), shape=(20, 20))
    dataset = UMAPDataset(X, graph, edge_pruning_factor=0.0, random_state=42)
    monkeypatch.setattr(dataset, "__getitem__", lambda _: pytest.fail("scalar fetch used"))
    loader = UMAPDataModule(dataset, batch_size=8, num_workers=0).train_dataloader()
    left, right = next(iter(loader))
    assert left.shape == right.shape == (8, 3)
