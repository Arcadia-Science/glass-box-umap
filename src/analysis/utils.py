
#@title UMAP utils
from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch

import torch
from torch import nn
import torch.nn.functional as F

def convert_distance_to_log_probability(distances, a=1.0, b=1.0):
    """
     convert distance representation into log probability,
        as a function of a, b params

    Parameters
    ----------
    distances : array
        euclidean distance between two points in embedding
    a : float, optional
        parameter based on min_dist, by default 1.0
    b : float, optional
        parameter based on min_dist, by default 1.0

    Returns
    -------
    float
        log probability in embedding space
    """
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, log_probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability

    Parameters
    ----------
    probabilities_graph : array
        high dimensional probabilities
    log_probabilities_distance : array
        low dimensional log probabilities
    EPS : float, optional
        offset to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: float
        attraction term for cross entropy loss
    repellant_term: float
        repellent term for cross entropy loss
    cross_entropy: float
        cross entropy umap loss

    """
    # cross entropy
    m = nn.LogSigmoid()
    attraction_term = -probabilities_graph * m(log_probabilities_distance)
    # use numerically stable repellent term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (m(log_probabilities_distance) - log_probabilities_distance)
        * repulsion_strength
    )

    # balance the expected losses between attraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(
    embedding_to,
    embedding_from,
    _a,
    _b,
    batch_size,
    negative_sample_rate=5,
    repulsion_strength=1.0,
):
    """
    Corrected UMAP loss function.
    
    Args:
        embedding_to (torch.Tensor): The 'to' embeddings of positive pairs.
        embedding_from (torch.Tensor): The 'from' embeddings of positive pairs.
        _a (float): The 'a' parameter of the UMAP low-dimensional curve.
        _b (float): The 'b' parameter of the UMAP low-dimensional curve.
        batch_size (int): The number of positive pairs in the batch.
        negative_sample_rate (int): The number of negative samples per positive pair.
        repulsion_strength (float): The weight of the repulsion term.

    Returns:
        torch.Tensor: The final UMAP loss.
    """
    # 1. Create positive and negative sample pairs
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]

    # 2. Calculate distances for positive and negative pairs
    dist_pos = (embedding_to - embedding_from).norm(dim=1)
    dist_neg = (embedding_neg_to - embedding_neg_from).norm(dim=1)

    # 3. Calculate the attraction term (for positive pairs)
    # The term is log(Q_ij), where Q_ij = 1 / (1 + a*d^(2b))
    # This simplifies to -log(1 + a*d^(2b))
    # We use torch.log1p for numerical stability: log(1 + x)
    loss_attraction = torch.log1p(_a * dist_pos.pow(2 * _b))

    # 4. Calculate the repulsion term (for negative pairs)
    # The term is log(1 - Q_ij), where Q_ij is the similarity for negative pairs
    # This simplifies to log(1 + 1/(a*d^(2b)))
    # We use .clamp(min=1e-8) to prevent division by zero if distance is zero
    loss_repulsion = torch.log1p(
        1.0 / (_a * dist_neg.pow(2 * _b).clamp(min=1e-8))
    )

    # 5. Combine and average the losses
    # The UMAP loss is the sum of attraction and repulsion terms.
    # We take the mean of each and apply the repulsion strength.
    loss = torch.mean(loss_attraction) + repulsion_strength * torch.mean(loss_repulsion)
    
    return loss


# def convert_distance_to_probability(distances, a=1.0, b=1.0):
#     #return 1.0 / (1.0 + a * distances ** (2 * b))
#     return -torch.log1p(a * distances ** (2 * b))

# def compute_cross_entropy(
#     probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
# ):
#     # cross entropy
#     attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
#         probabilities_distance
#     )
#     repellant_term = (
#         -(1.0 - probabilities_graph)
#         * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
#         * repulsion_strength
#     )

#     # balance the expected losses between atrraction and repel
#     CE = attraction_term + repellant_term
#     return attraction_term, repellant_term, CE

# def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
#     # get negative samples by randomly shuffling the batch
#     embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
#     repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
#     embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
#     distance_embedding = torch.cat((
#         (embedding_to - embedding_from).norm(dim=1),
#         (embedding_neg_to - embedding_neg_from).norm(dim=1)
#     ), dim=0)

#     # convert probabilities to distances
#     probabilities_distance = convert_distance_to_log_probability(
#         distance_embedding, _a, _b
#     )
#     # set true probabilities based on negative sampling
#     probabilities_graph = torch.cat(
#         (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
#     )

#     # compute cross entropy
#     (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
#         probabilities_graph.cuda(),
#         probabilities_distance.cuda(),
#     )
#     loss = torch.mean(ce_loss)
#     return loss

def get_umap_graph(X, n_neighbors=15, metric="precomputed", random_state=None):
    random_state = check_random_state(None) if random_state == None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    if metric == "precomputed":

        @numba.njit(parallel=True)
        def _extract_and_sort(data, indices, indptr, n_samples, k):
            """
            Numba-jitted function to perform the extraction and sorting efficiently.
            For each row, it finds its neighbors and sorts them by distance.
            """
            # Create placeholder arrays for the results
            knn_indices = np.zeros((n_samples, k), dtype=np.int64)
            knn_dists = np.zeros((n_samples, k), dtype=np.float32)
        
            # Use prange for a parallel loop
            for i in numba.prange(n_samples):
                # Get the slice corresponding to the current row
                start = indptr[i]
                end = indptr[i+1]
                
                # Get the distances and indices for the neighbors in this row
                row_dists = data[start:end]
                row_indices = indices[start:end]
                
                # Sort the neighbors by distance (important!)
                sort_order = np.argsort(row_dists)
                
                # Store the sorted results
                knn_dists[i, :] = row_dists[sort_order]
                knn_indices[i, :] = row_indices[sort_order]
                
            return knn_indices, knn_dists
        def extract_knn_from_sparse(sparse_matrix, k):
            """
            Extracts sorted k-NN indices and distances from a sparse distance matrix.
        
            Args:
                sparse_matrix (scipy.sparse.csr_matrix): A sparse matrix where non-zero entries
                                                         are the distances to k-nearest neighbors.
                k (int): The number of neighbors stored in the matrix.
        
            Returns:
                tuple: A tuple containing:
                    - knn_indices (np.ndarray): Shape (n_samples, k)
                    - knn_dists (np.ndarray): Shape (n_samples, k)
            """
            # Ensure the matrix is in CSR format for efficient row slicing
            sparse_matrix_csr = sparse_matrix.tocsr()
            
            n_samples, _ = sparse_matrix_csr.shape
            
            # Call the fast Numba function to do the work
            knn_indices, knn_dists = _extract_and_sort(
                sparse_matrix_csr.data,
                sparse_matrix_csr.indices,
                sparse_matrix_csr.indptr,
                n_samples,
                k
            )
            
            return knn_indices, knn_dists

        
        knn_indices, knn_dists = extract_knn_from_sparse(sparse_knn_graph, k=n_neighbors)
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1

        knn_search_index = None
    # if metric == "precomputed":
    #     print("PRECOMPUTED graph, finding NNs")
    #     from umap.utils import (
    #         submatrix,
    #         ts,
    #         csr_unique,
    #         fast_knn_indices,
    #     )
        # # Note that this does not support sparse distance matrices yet ...
        # # Compute indices of n nearest neighbors
        # knn_indices = fast_knn_indices(X, n_neighbors)
        # # knn_indices = np.argsort(X)[:, :n_neighbors]
        # # Compute the nearest neighbor distances
        # #   (equivalent to np.sort(X)[:,:n_neighbors])
        # knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # # Prune any nearest neighbours that are infinite distance apart.
        # disconnected_index = knn_dists == np.inf
        # knn_indices[disconnected_index] = -1

        # knn_search_index = None
        # # print("Shape: ", knn_dists.shape)
        # # print(knn_indices)
    else:

        # get nearest neighbors
        nnd = NNDescent(
            X,#.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=160,
            n_jobs=1,
            verbose=True
        )
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X = X,
        n_neighbors = n_neighbors,
        metric = metric,
        random_state = random_state,
        knn_indices= knn_indices,
        knn_dists = knn_dists,
    )

    return umap_graph

#@title Load UMAP
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

class UMAPDataset(Dataset):
    def __init__(self, data, graph_, n_epochs=200):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph_, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.Tensor(data)

    def __len__(self):
        # return int(self.edges_to_exp.shape[0]/1)

        return int(self.data.shape[0])

    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)

#@title UMAP data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

class UMAPDataset(Dataset):
    def __init__(self, data, graph_, n_epochs=200):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph_, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )
        print(self.edges_to_exp.shape)
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.Tensor(data)

    def __len__(self):
        return int(self.data.shape[0])

        # return int(self.edges_to_exp.shape[0]/1)
    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)

class MatchDataset(Dataset):
    def __init__(self, data, embeddings):
        self.embeddings = torch.Tensor(embeddings)
        self.data = data
    def __len__(self):
        return int(self.data.shape[0])
    def __getitem__(self, index):
        return self.data[index], self.embeddings[index]

#@title UMAP modules
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from umap.umap_ import find_ab_params


""" Model """
class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        match_nonparametric_umap=False,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss


""" Datamodule """

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

# min_dist=0.5, spread=1.0, n_components=2, maxiter=None, alpha=1.0, gamma=1.0, negative_sample_rate=5, init_pos='spectral', random_state=0, a=None, b=None, method='umap', neighbors_key='neighbors', copy=False

class PUMAP():
    def __init__(
        self,
        encoder,
        decoder=None,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        lr=1e-3,
        epochs=30,
        batch_size=256*2,
        num_workers=20,
        random_state='random',
        match_nonparametric_umap=False,
        non_parametric_embeddings=None
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.match_nonparametric_umap = match_nonparametric_umap
        self.non_parametric_embeddings = non_parametric_embeddings

    def fit(self, X, precomputed_graph=None, gradient_clip_val=4.0):
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs, gradient_clip_val=gradient_clip_val )
        # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        self.model = Model(self.lr, self.encoder, min_dist=self.min_dist, match_nonparametric_umap=self.match_nonparametric_umap)
        if precomputed_graph is None:
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
        else:
            graph = get_umap_graph(precomputed_graph, n_neighbors=self.n_neighbors, metric="precomputed", random_state=self.random_state)

        if self.non_parametric_embeddings is None:
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
                )
        else:
            print("Fitting Non parametric Umap")
            # non_parametric_umap = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric, n_components=self.n_components, random_state=self.random_state, verbose=True)
            # non_parametric_embeddings = non_parametric_umap.fit_transform(torch.flatten(X, 1, -1).numpy())
            # self.model = Model(self.lr, encoder, decoder, beta=self.beta, reconstruction_loss=self.reconstruction_loss, match_nonparametric_umap=self.match_nonparametric_umap)
            print("Training NN to match embeddings")
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(MatchDataset(X, self.non_parametric_embeddings), self.batch_size, self.num_workers)
            )


    @torch.no_grad()
    def transform(self, X):
        self.embedding_ = self.model.encoder(X).detach().cpu().numpy()
        return self.embedding_

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()
