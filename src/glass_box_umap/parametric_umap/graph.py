from typing import Literal, cast

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

GraphElements = tuple[
    coo_matrix,
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float32],
    int,
]

NeighborSearch = Literal["nndescent", "exact", "auto"]

# umap-learn computes exact neighbors below this many samples, and uses NNDescent above.
EXACT_SEARCH_MAX_SAMPLES = 4096

# Metrics the exact search supports.
_EXACT_METRICS = frozenset({"euclidean"})

# Upper bound on the number of float32 distances held at once by the exact search (256 MB).
_EXACT_BLOCK_ELEMENTS = 2**26

# umap-learn's smooth_knn_dist constants.
_SMOOTH_K_TOLERANCE = 1e-5
_MIN_K_DIST_SCALE = 1e-3


def resolve_neighbor_search(
    neighbor_search: NeighborSearch, n_samples: int, metric: str
) -> Literal["nndescent", "exact"]:
    """Decide which nearest-neighbor search a fit uses.

    Args:
        neighbor_search: ``"nndescent"``, ``"exact"`` or ``"auto"``. ``"auto"`` picks
            ``"exact"`` below ``EXACT_SEARCH_MAX_SAMPLES`` samples, as umap-learn does, when
            the metric is one the exact search supports, and ``"nndescent"`` otherwise.
        n_samples: Number of samples in the fit.
        metric: Distance metric.

    Returns:
        ``"nndescent"`` or ``"exact"``.
    """
    if neighbor_search == "auto":
        if n_samples < EXACT_SEARCH_MAX_SAMPLES and metric in _EXACT_METRICS:
            return "exact"
        return "nndescent"
    if neighbor_search == "exact":
        if metric not in _EXACT_METRICS:
            raise ValueError(
                f"neighbor_search='exact' supports the metrics {sorted(_EXACT_METRICS)}, "
                f"not {metric!r}. Use neighbor_search='nndescent'."
            )
        return "exact"
    if neighbor_search == "nndescent":
        return "nndescent"
    raise ValueError(
        f"neighbor_search must be 'nndescent', 'exact' or 'auto', not {neighbor_search!r}."
    )


def get_umap_graph(
    X: NDArray[np.floating],
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: RandomState | int | None = None,
    quiet: bool = False,
    neighbor_search: NeighborSearch = "nndescent",
) -> csr_matrix:
    """Build a UMAP graph from input data.

    Constructs the fuzzy simplicial set representation used by UMAP, which
    captures local neighborhood structure in the data.

    Args:
        X: Input data array of shape (n_samples, n_features).
        n_neighbors: Number of neighbors to use for graph construction.
        metric: Distance metric for neighbor search.
        random_state: Random state for reproducibility.
        quiet: If ``True``, NNDescent prints nothing.
        neighbor_search: How to find each sample's nearest neighbors.
            ``"nndescent"`` (the default) uses pynndescent's approximate search and
            umap-learn's fuzzy graph. ``"exact"`` computes exact neighbors with PyTorch
            and builds the same fuzzy graph in NumPy, so nothing is compiled with numba;
            the graph can differ slightly from ``"nndescent"``'s because approximate and
            exact neighbors differ. ``"auto"`` uses ``"exact"`` below
            ``EXACT_SEARCH_MAX_SAMPLES`` samples and ``"nndescent"`` above.

    Returns:
        Sparse CSR matrix representing the UMAP graph with edge weights.
    """
    method = resolve_neighbor_search(neighbor_search, X.shape[0], metric)
    if method == "exact":
        knn_indices, knn_dists = exact_nearest_neighbors(X, n_neighbors)
        return fuzzy_simplicial_set_numpy(knn_indices, knn_dists, n_neighbors)

    # Imported here because importing pynndescent and umap-learn takes seconds.
    from pynndescent import NNDescent
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set

    rng = check_random_state(random_state)
    n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    nnd = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=not quiet,
    )

    assert nnd.neighbor_graph is not None
    knn_indices, knn_dists = nnd.neighbor_graph

    umap_graph = cast(
        csr_matrix,
        fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=rng,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )[0],
    )

    return umap_graph


def exact_nearest_neighbors(
    X: NDArray[np.floating], n_neighbors: int
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Exact Euclidean nearest neighbors, laid out as NNDescent returns them.

    Each row lists a sample's ``n_neighbors`` nearest samples, nearest first, starting with
    the sample itself at distance 0. Distances are computed directly, not through the
    ``|a|² + |b|² - 2ab`` expansion, so they are accurate for near neighbors. Memory stays
    bounded by computing the distance matrix a block of rows at a time.

    Args:
        X: Data of shape (n_samples, n_features).
        n_neighbors: Neighbors per sample, including the sample itself.

    Returns:
        ``(indices, distances)``, each of shape (n_samples, n_neighbors).
    """
    import torch

    n = X.shape[0]
    if not 1 <= n_neighbors <= n:
        raise ValueError(
            f"n_neighbors must be between 1 and the number of samples ({n}), not {n_neighbors}."
        )
    data = torch.from_numpy(np.ascontiguousarray(X.reshape(n, -1), dtype=np.float32))
    indices = np.empty((n, n_neighbors), dtype=np.int32)
    distances = np.empty((n, n_neighbors), dtype=np.float32)
    rows_per_block = max(1, _EXACT_BLOCK_ELEMENTS // n)
    for start in range(0, n, rows_per_block):
        stop = min(start + rows_per_block, n)
        d = torch.cdist(data[start:stop], data, compute_mode="donot_use_mm_for_euclid_dist")
        # Put each sample first among its own neighbors, even when it has duplicates.
        rows = torch.arange(stop - start)
        d[rows, rows + start] = -1.0
        vals, idx = torch.topk(d, n_neighbors, dim=1, largest=False, sorted=True)
        vals[:, 0] = 0.0
        indices[start:stop] = idx.numpy()
        distances[start:stop] = vals.numpy()
    return indices, distances


def smooth_knn_dist_numpy(
    distances: NDArray[np.float32], k: float, n_iter: int = 64
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """umap-learn's ``smooth_knn_dist`` (local_connectivity=1, bandwidth=1) in NumPy.

    For each row, finds ``rho``, the distance to the nearest neighbor at a positive
    distance, and ``sigma`` such that the neighbors' memberships sum to ``log2(k)``,
    by the same binary search as umap-learn, run on all rows at once.

    Args:
        distances: Each sample's nearest-neighbor distances, shape (n_samples, n_neighbors),
            with the sample itself first.
        k: The number of neighbors.
        n_iter: Binary search steps.

    Returns:
        ``(sigmas, rhos)``, each of shape (n_samples,).
    """
    n = distances.shape[0]
    target = np.log2(k)

    positive = distances > 0.0
    has_positive = positive.any(axis=1)
    first_positive = np.argmax(positive, axis=1)
    rho = np.where(has_positive, distances[np.arange(n), first_positive], 0.0).astype(np.float32)

    shifted = (distances[:, 1:] - rho[:, None]).astype(np.float64)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    mid = np.ones(n)
    active = np.ones(n, dtype=bool)
    for _ in range(n_iter):
        if not active.any():
            break
        rows = np.flatnonzero(active)
        d = shifted[rows]
        with np.errstate(over="ignore"):
            psum = np.where(d > 0, np.exp(-(d / mid[rows, None])), 1.0).sum(axis=1)
        done = np.abs(psum - target) < _SMOOTH_K_TOLERANCE
        active[rows[done]] = False

        update = rows[~done]
        too_high = psum[~done] > target
        down = update[too_high]
        hi[down] = mid[down]
        mid[down] = (lo[down] + hi[down]) / 2.0
        up = update[~too_high]
        lo[up] = mid[up]
        unbounded = np.isinf(hi[up])
        mid[up[unbounded]] *= 2
        bounded = up[~unbounded]
        mid[bounded] = (lo[bounded] + hi[bounded]) / 2.0

    sigma = mid.astype(np.float32)
    floor = np.where(
        rho > 0.0,
        _MIN_K_DIST_SCALE * distances.mean(axis=1),
        _MIN_K_DIST_SCALE * np.mean(distances),
    ).astype(np.float32)
    sigma = np.where(sigma < floor, floor, sigma).astype(np.float32)
    return sigma, rho


def fuzzy_simplicial_set_numpy(
    knn_indices: NDArray[np.integer], knn_dists: NDArray[np.floating], n_neighbors: int
) -> csr_matrix:
    """umap-learn's ``fuzzy_simplicial_set`` from precomputed neighbors, in NumPy.

    Uses umap-learn's defaults: ``local_connectivity=1`` and ``set_op_mix_ratio=1`` (the
    fuzzy union). The weights match umap-learn's to float32 rounding.

    Args:
        knn_indices: Each sample's neighbors, shape (n_samples, n_neighbors), itself first.
        knn_dists: The matching distances.
        n_neighbors: The number of neighbors.

    Returns:
        The symmetric fuzzy graph, shape (n_samples, n_samples).
    """
    n = knn_indices.shape[0]
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist_numpy(knn_dists, float(n_neighbors))

    shifted = knn_dists - rhos[:, None]
    safe_sigma = np.where(sigmas == 0.0, np.float32(1.0), sigmas)[:, None]
    with np.errstate(over="ignore"):
        vals = np.where(
            (shifted <= 0.0) | (sigmas[:, None] == 0.0),
            np.float32(1.0),
            np.exp(-(shifted / safe_sigma)),
        ).astype(np.float32)
    vals[knn_indices == np.arange(n)[:, None]] = 0.0
    vals[knn_indices == -1] = 0.0

    rows = np.repeat(np.arange(n), knn_indices.shape[1])
    cols = np.where(knn_indices == -1, 0, knn_indices).ravel()
    membership = coo_matrix((vals.ravel(), (rows, cols)), shape=(n, n))
    membership.eliminate_zeros()

    transpose = membership.transpose()
    graph = membership + transpose - membership.multiply(transpose)
    graph = csr_matrix(graph)
    graph.eliminate_zeros()
    return graph


def get_graph_elements(graph: csr_matrix, edge_pruning_factor: float) -> GraphElements:
    """Extract and prune edges from a sparse UMAP graph.

    Removes weak edges below a relative weight threshold and returns the
    remaining edge indices, weights, and vertex counts.

    Args:
        graph: Sparse CSR matrix representing the UMAP graph with edge weights.
        edge_pruning_factor: Relative threshold for discarding weak edges. Edges with
            weight less than ``max_weight * edge_pruning_factor`` are removed.

    Returns:
        A tuple containing:
            - graph: The COO format graph with low-weight edges removed
            - vertices_a: Source vertex indices for each edge (rows in COO matrix)
            - vertices_b: Destination vertex indices for each edge (cols in COO matrix)
            - edge_weights: Edge weights
            - num_vertices: Total number of vertices in the graph
    """
    graph_coo = cast(coo_matrix, graph.tocoo())
    graph_coo.sum_duplicates()

    num_vertices = int(graph_coo.get_shape()[0])

    graph_coo.data[graph_coo.data < (graph_coo.data.max() * float(edge_pruning_factor))] = 0.0
    graph_coo.eliminate_zeros()

    vertices_a = cast(NDArray[np.int32], graph_coo.row.astype(np.int32))
    vertices_b = cast(NDArray[np.int32], graph_coo.col.astype(np.int32))
    edge_weights = cast(NDArray[np.float32], graph_coo.data.astype(np.float32))

    return graph_coo, vertices_a, vertices_b, edge_weights, num_vertices
