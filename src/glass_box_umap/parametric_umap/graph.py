from typing import cast

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from pynndescent import NNDescent
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set

GraphElements = tuple[
    coo_matrix,
    NDArray[np.float32],  # sampling_weights
    NDArray[np.intp],  # head
    NDArray[np.intp],  # tail
    NDArray[np.float64],  # weight  <-- was np.floating
    int,
]


def get_umap_graph(
    X: NDArray[np.floating],
    n_neighbors: int = 10,
    metric: str = "euclidean",
    random_state: RandomState | int | None = None,
) -> csr_matrix:
    """Build a UMAP graph from input data using nearest neighbor descent.

    Constructs the fuzzy simplicial set representation used by UMAP, which
    captures local neighborhood structure in the data.

    Args:
        X: Input data array of shape (n_samples, n_features).
        n_neighbors: Number of neighbors to use for graph construction.
        metric: Distance metric for neighbor search.
        random_state: Random state for reproducibility.

    Returns:
        Sparse CSR matrix representing the UMAP graph with edge weights.
    """
    rng = check_random_state(random_state)
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    nnd = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True,
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


def get_graph_elements(graph: csr_matrix, edge_pruning_factor: float) -> GraphElements:
    """Extract and prune edges from a sparse UMAP graph.

    Removes weak edges below a relative weight threshold and returns the
    remaining edge indices, weights, and sampling weights.

    Args:
        graph: Sparse CSR matrix representing the UMAP graph with edge weights.
        edge_pruning_factor: Relative threshold for discarding weak edges. Edges with
            weight less than ``max_weight * edge_pruning_factor`` are removed.

    Returns:
        A tuple containing:
            - graph: The COO format graph with low-weight edges removed
            - sampling_weights: Edge weights scaled by the inverse of the pruning factor
            - head: Source vertex indices for each edge
            - tail: Target vertex indices for each edge
            - weight: Edge weights
            - n_vertices: Total number of vertices in the graph
    """
    graph_coo = cast(coo_matrix, graph.tocoo())
    graph_coo.sum_duplicates()

    n_vertices = graph_coo.get_shape()[0]

    graph_coo.data[graph_coo.data < (graph_coo.data.max() * float(edge_pruning_factor))] = 0.0
    graph_coo.eliminate_zeros()

    sampling_weights = (graph_coo.data / edge_pruning_factor).astype(np.float32)
    head = graph_coo.row.astype(np.intp)
    tail = graph_coo.col.astype(np.intp)
    weight = graph_coo.data.astype(np.float64)

    return graph_coo, sampling_weights, head, tail, weight, int(n_vertices)
