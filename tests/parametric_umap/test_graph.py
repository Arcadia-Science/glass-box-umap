"""Exact nearest neighbors and the NumPy fuzzy graph (``neighbor_search="exact"``)."""

from typing import cast

import numpy as np
import pytest
from glass_box_umap import GlassBoxUMAP
from glass_box_umap.parametric_umap import graph as graph_module
from glass_box_umap.parametric_umap.graph import (
    EXACT_SEARCH_MAX_SAMPLES,
    exact_nearest_neighbors,
    fuzzy_simplicial_set_numpy,
    get_umap_graph,
    resolve_neighbor_search,
)
from scipy.sparse import csr_matrix


def _umap_graph(result: object) -> csr_matrix:
    return csr_matrix(cast(tuple, result)[0])


def _data(n: int = 300, d: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(5, d))
    return (centers[rng.integers(5, size=n)] + rng.normal(size=(n, d))).astype(np.float32)


def test_exact_neighbors_match_brute_force() -> None:
    X = _data()
    idx, dist = exact_nearest_neighbors(X, 10)
    full = np.sqrt(((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(full, -1.0)
    expected = np.argsort(full, axis=1, kind="stable")[:, :10]
    assert idx.dtype == np.int32 and dist.dtype == np.float32
    assert (idx[:, 0] == np.arange(len(X))).all()
    assert (dist[:, 0] == 0.0).all()
    # Same neighbor sets; distances to float32 rounding.
    assert all(set(a) == set(b) for a, b in zip(idx, expected, strict=True))
    np.testing.assert_allclose(dist[:, 1:], np.sort(full, axis=1)[:, 1:10], rtol=1e-5)


def test_exact_neighbors_put_each_sample_first_despite_duplicates() -> None:
    X = np.repeat(_data(20), 3, axis=0)
    idx, dist = exact_nearest_neighbors(X, 4)
    assert (idx[:, 0] == np.arange(len(X))).all()
    assert (dist[:, :3] == 0.0).all()


def test_exact_neighbors_blocks_give_the_same_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    X = _data(257)
    whole = exact_nearest_neighbors(X, 7)
    monkeypatch.setattr(graph_module, "_EXACT_BLOCK_ELEMENTS", 257 * 10)  # 10 rows per block
    blocked = exact_nearest_neighbors(X, 7)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_array_equal(whole[1], blocked[1])


def test_exact_neighbors_rejects_more_neighbors_than_samples() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        exact_nearest_neighbors(_data(5), 6)


@pytest.mark.parametrize("n_neighbors", [2, 4, 15])
def test_numpy_fuzzy_graph_matches_umap_learn(n_neighbors: int) -> None:
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set

    X = _data(400)
    knn_idx, knn_dist = exact_nearest_neighbors(X, n_neighbors)
    expected = _umap_graph(
        fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            metric="euclidean",
            random_state=check_random_state(0),
            knn_indices=knn_idx,
            knn_dists=knn_dist,
        )
    )
    got = fuzzy_simplicial_set_numpy(knn_idx, knn_dist, n_neighbors)
    assert got.dtype == np.float32
    assert got.shape == expected.shape
    assert abs(got - expected).max() < 1e-6
    np.testing.assert_array_equal(got.toarray() != 0, expected.toarray() != 0)


def test_numpy_fuzzy_graph_matches_umap_learn_with_duplicates() -> None:
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set

    X = np.concatenate([_data(60), _data(60)[:20]])
    knn_idx, knn_dist = exact_nearest_neighbors(X, 6)
    expected = _umap_graph(
        fuzzy_simplicial_set(
            X=X,
            n_neighbors=6,
            metric="euclidean",
            random_state=check_random_state(0),
            knn_indices=knn_idx,
            knn_dists=knn_dist,
        )
    )
    got = fuzzy_simplicial_set_numpy(knn_idx, knn_dist, 6)
    assert abs(got - expected).max() < 1e-6


def test_resolve_neighbor_search() -> None:
    assert resolve_neighbor_search("nndescent", 10, "euclidean") == "nndescent"
    assert resolve_neighbor_search("exact", 10**6, "euclidean") == "exact"
    assert resolve_neighbor_search("auto", EXACT_SEARCH_MAX_SAMPLES - 1, "euclidean") == "exact"
    assert resolve_neighbor_search("auto", EXACT_SEARCH_MAX_SAMPLES, "euclidean") == "nndescent"
    assert resolve_neighbor_search("auto", 10, "cosine") == "nndescent"
    with pytest.raises(ValueError, match="supports the metrics"):
        resolve_neighbor_search("exact", 10, "cosine")
    with pytest.raises(ValueError, match="neighbor_search must be"):
        resolve_neighbor_search("annoy", 10, "euclidean")  # type: ignore[arg-type]


def test_exact_graph_is_close_to_nndescent_graph() -> None:
    """NNDescent finds the exact neighbors of small, low-dimensional data, so the graphs agree."""
    X = _data(200)
    approximate = get_umap_graph(X, n_neighbors=10, random_state=0, quiet=True)
    exact = get_umap_graph(X, n_neighbors=10, neighbor_search="exact")
    assert abs(approximate - exact).max() < 1e-5


@pytest.mark.parametrize("neighbor_search", ["exact", "auto"])
def test_glass_box_with_exact_neighbors(neighbor_search: str) -> None:
    X = _data(120, d=11)
    model = GlassBoxUMAP(
        random_state=0,
        quiet=True,
        epochs=5,
        n_neighbors=10,
        neighbor_search=neighbor_search,  # type: ignore[arg-type]
    ).to("cpu")
    Y = model.fit_transform(X)
    C = model.compute_contributions(X)
    assert Y.shape == (120, 2)
    assert np.abs(C.sum(-1) - Y).max() < 1e-4
    again = GlassBoxUMAP(
        random_state=0,
        quiet=True,
        epochs=5,
        n_neighbors=10,
        neighbor_search=neighbor_search,  # type: ignore[arg-type]
    ).to("cpu")
    np.testing.assert_array_equal(again.fit_transform(X), Y)


def test_tiny_input_with_exact_neighbors() -> None:
    """The smallest useful input: 5 samples, 4 neighbors."""
    X = _data(5, d=11)
    model = GlassBoxUMAP(
        random_state=0, quiet=True, epochs=3, n_neighbors=4, neighbor_search="exact"
    )
    Y = model.to("cpu").fit_transform(X)
    assert np.isfinite(Y).all()
