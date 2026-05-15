import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.utils.data import Dataset

from .graph import get_graph_elements


class UMAPDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset for UMAP edge-based training.

    Generates pairs of data points connected by edges in the UMAP graph,
    used for training the encoder to preserve local structure. The dataset
    length is the number of edges remaining after pruning, not the number
    of input samples.

    Args:
        data: Input data array of shape (n_samples, ...).
        graph: Sparse UMAP graph representing neighborhood relationships.
        edge_pruning_factor: Relative threshold for discarding weak edges from the graph.

    Attributes:
        vertices_a:
            Vertex indices for one endpoint (COO matrix rows) of each edge, shape
            (n_edges,).
        vertices_b:
            Vertex indices for the other endpoint (COO matrix columns) of each edge,
            shape (n_edges,).
        edge_weights:
            Edge weights, shape (n_edges,).
        data:
            Input feature vectors, shape (n_samples, ...).
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        graph: csr_matrix,
        edge_pruning_factor: float = 0.025,
        random_state: int | None = None,
    ) -> None:
        _, vertices_a, vertices_b, edge_weights, _ = get_graph_elements(graph, edge_pruning_factor)
        # Permute edges so that any contiguous slice is a random sample of
        # the graph. Required by SplitAndMergeWeightedSampler to keep
        # cross-chunk allocation unbiased when num_edges > 2**24.
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(vertices_a.shape[0])
        self.vertices_a = vertices_a[perm]
        self.vertices_b = vertices_b[perm]
        self.edge_weights = edge_weights[perm]
        self.data = torch.as_tensor(data, dtype=torch.float32)

    @property
    def num_edges(self) -> int:
        return self.vertices_a.shape[0]

    @property
    def num_samples(self) -> int:
        return self.data.shape[0]

    def __len__(self) -> int:
        return self.num_edges

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        vertex_a_data = self.data[self.vertices_a[index]]
        vertex_b_data = self.data[self.vertices_b[index]]
        return vertex_a_data, vertex_b_data
