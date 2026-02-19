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
    NDArray[np.floating],
    NDArray[np.intp],
    NDArray[np.intp],
    NDArray[np.floating],
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


def get_graph_elements(graph: csr_matrix, n_epochs: int) -> GraphElements:
    """Extract graph elements from a sparse UMAP graph for edge sampling.

    Converts a sparse graph representation into arrays of edge indices and weights
    suitable for training a parametric UMAP model.

    Args:
        graph: Sparse CSR matrix representing the UMAP graph with edge weights.
        n_epochs: Number of training epochs, used to determine sampling frequency.

    Returns:
        A tuple containing:
            - graph: The COO format graph with low-probability edges removed
            - epochs_per_sample: Number of times each edge should be sampled per epoch
            - head: Source vertex indices for each edge
            - tail: Target vertex indices for each edge
            - weight: Edge weights
            - n_vertices: Total number of vertices in the graph
    """
    graph_coo = cast(coo_matrix, graph.tocoo())
    graph_coo.sum_duplicates()

    n_vertices = graph_coo.get_shape()[0]

    graph_coo.data[graph_coo.data < (graph_coo.data.max() / float(n_epochs))] = 0.0
    graph_coo.eliminate_zeros()

    epochs_per_sample = (n_epochs * graph_coo.data).astype(np.float32)
    head = graph_coo.row
    tail = graph_coo.col
    weight = graph_coo.data

    return graph_coo, epochs_per_sample, head, tail, weight, n_vertices
