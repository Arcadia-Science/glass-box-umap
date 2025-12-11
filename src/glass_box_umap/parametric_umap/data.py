from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


GraphElements = tuple[
    coo_matrix,
    NDArray[np.floating],
    NDArray[np.intp],
    NDArray[np.intp],
    NDArray[np.floating],
    int,
]


def get_graph_elements(graph: "csr_matrix", n_epochs: int) -> GraphElements:
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
    graph_coo = graph.tocoo()
    graph_coo.sum_duplicates()
    n_vertices = graph_coo.shape[1]

    graph_coo.data[graph_coo.data < (graph_coo.data.max() / float(n_epochs))] = 0.0
    graph_coo.eliminate_zeros()

    epochs_per_sample = n_epochs * graph_coo.data
    head = graph_coo.row
    tail = graph_coo.col
    weight = graph_coo.data

    return graph_coo, epochs_per_sample, head, tail, weight, n_vertices


class UMAPDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset for UMAP edge-based training.

    Generates pairs of data points connected by edges in the UMAP graph,
    used for training the encoder to preserve local structure.

    Args:
        data: Input data array of shape (n_samples, ...).
        graph: Sparse UMAP graph representing neighborhood relationships.
        n_epochs: Number of training epochs for computing edge sampling frequency.
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        graph: "csr_matrix",
        n_epochs: int = 200,
    ) -> None:
        _, epochs_per_sample, head, tail, _, _ = get_graph_elements(graph, n_epochs)

        edges_to_exp = np.repeat(head, epochs_per_sample.astype(np.intp))
        edges_from_exp = np.repeat(tail, epochs_per_sample.astype(np.intp))

        shuffle_mask = np.random.permutation(np.arange(len(edges_to_exp)))
        self.edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return edges_to_exp, edges_from_exp


class MatchDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset for matching parametric embeddings to non-parametric ones.

    Used when training a parametric model to reproduce embeddings from a
    pre-computed non-parametric UMAP.

    Args:
        data: Input data array of shape (n_samples, ...).
        embeddings: Pre-computed UMAP embeddings of shape (n_samples, n_components).
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        embeddings: NDArray[np.floating],
    ) -> None:
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.data[index], self.embeddings[index]
