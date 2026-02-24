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
    used for training the encoder to preserve local structure.

    Args:
        data: Input data array of shape (n_samples, ...).
        graph: Sparse UMAP graph representing neighborhood relationships.
        n_epochs: Number of training epochs for computing edge sampling frequency.
    """

    # def __init__(
    #     self,
    #     data: NDArray[np.floating],
    #     graph: csr_matrix,
    #     n_epochs: int = 200,
    # ) -> None:
    #     _, epochs_per_sample, head, tail, _, _ = get_graph_elements(graph, n_epochs)

    #     edges_to_exp = np.repeat(head, epochs_per_sample.astype(np.intp))
    #     edges_from_exp = np.repeat(tail, epochs_per_sample.astype(np.intp))

    def __init__(
        self,
        data: NDArray[np.floating],
        graph: csr_matrix,
        n_epochs: int = 40,#200,#
    ) -> None:
        _, epochs_per_sample, head, tail, _, _ = get_graph_elements(graph, n_epochs)
        print("n_epochs: ", n_epochs)
        print("Graph len: ", len(head),len(tail))
        edges_to_exp = np.repeat(head,1)# epochs_per_sample.astype(np.intp))
        edges_from_exp = np.repeat(tail, 1)#epochs_per_sample.astype(np.intp))

        shuffle_mask = np.random.permutation(np.arange(len(edges_to_exp)))
        self.edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        # return self.data.shape[0]
        return self.edges_to_exp.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return edges_to_exp, edges_from_exp
