import matplotlib.pyplot as plt
import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
# Keep fuzzy_simplicial_set and find_ab_params from umap.umap_
from umap.umap_ import find_ab_params, fuzzy_simplicial_set
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss # Still needed if you use it elsewhere, but not for main training loss now
import torch.nn.functional as F

# --- Additional imports required for the new spectral functions ---
import warnings
from warnings import warn
import scipy.sparse
import scipy.sparse.csgraph
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import _VALID_METRICS as SKLEARN_PAIRWISE_VALID_METRICS
from umap.distances import pairwise_special_metric, SPECIAL_METRICS
from umap.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances


# --- Helper functions and Dataset classes from user's context ---
def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """
    Converts distances to probabilities based on UMAP's sigmoid-like curve.
    This function is used in the UMAP loss calculation.
    """
    # Original UMAP uses 1.0 / (1.0 + a * distances ** (2 * b)) for probability.
    # The provided code uses -torch.log1p(a * distances ** (2 * b)) which is
    # related to the negative log-sigmoid for attraction.
    return -torch.log1p(a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Computes the cross-entropy loss components for UMAP.
    Includes attraction and repulsion terms.
    """
    # Attraction term: -P_ij * log(Q_ij) where Q_ij is the probability from distance
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    # Repulsion term: -(1 - P_ij) * log(1 - Q_ij)
    # The provided code uses log(sigmoid(x)) - x for log(1-sigmoid(x))
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
        * repulsion_strength
    )

    # Total cross-entropy loss
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    """
    Calculates the UMAP loss for a batch of embeddings.
    Combines positive and negative samples.
    """
    # Generate negative samples by shuffling embeddings
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]

    # Calculate distances for positive and negative pairs
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1), # Positive pair distances
        (embedding_neg_to - embedding_neg_from).norm(dim=1) # Negative pair distances
    ), dim=0)

    # Convert distances to probabilities
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )

    # Define true probabilities: 1 for positive, 0 for negative
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # Compute cross entropy components and total loss
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda(), # Ensure tensors are on the same device
        probabilities_distance.cuda(),
    )
    loss = torch.mean(ce_loss)
    return loss

def get_umap_graph(X, n_neighbors=15, metric="precomputed", random_state=None):
    """
    Constructs the UMAP fuzzy simplicial set (graph) from input data.
    Supports both precomputed distance matrices and other metrics.
    """
    random_state = check_random_state(None) if random_state == None else random_state
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    if metric == "precomputed":
        print("PRECOMPUTED graph, finding NNs")
        from umap.utils import (
            fast_knn_indices,
        )
        # Compute indices of n nearest neighbors for precomputed distances
        knn_indices = fast_knn_indices(X, n_neighbors)
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # Prune any nearest neighbours that are infinite distance apart.
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1
    else:
        # Use NNDescent for approximate nearest neighbors
        nnd = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=160,
            n_jobs=1,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph

    # Build the fuzzy simplicial set (UMAP graph)
    # This should be from umap.umap_
    umap_graph, sigmas, rhos = fuzzy_simplicial_set( # Using fuzzy_simplicial_set from umap.umap_
        X = X,
        n_neighbors = n_neighbors,
        metric = metric,
        random_state = random_state,
        knn_indices= knn_indices,
        knn_dists = knn_dists,
    )

    return umap_graph

def get_graph_elements(graph_, n_epochs):
    """
    Prepares graph elements (edges, weights, epochs per sample) for UMAP training.
    """
    graph = graph_.tocoo()
    graph.sum_duplicates() # Eliminate duplicate entries by summing them
    n_vertices = graph.shape[1]

    if n_epochs is None:
        n_epochs = 500 if graph.shape[0] <= 10000 else 200

    # Remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

class UMAPDataset(Dataset):
    """
    Dataset for training a UMAP model with the UMAP loss.
    Provides pairs of connected points from the UMAP graph.
    """
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
        return int(self.data.shape[0])

    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)

class MatchDataset(Dataset):
    """
    Dataset for training a model to match pre-computed embeddings.
    Provides data points and their corresponding target embeddings.
    (This dataset is no longer used for the primary training loop, but kept for completeness)
    """
    def __init__(self, data, embeddings):
        self.embeddings = torch.Tensor(embeddings)
        self.data = data
    def __len__(self):
        return int(self.data.shape[0])
    def __getitem__(self, index):
        return self.data[index], self.embeddings[index]

# --- Model and Datamodule ---
class Model(pl.LightningModule):
    """
    PyTorch Lightning Model for UMAP training.
    It now always uses the UMAP loss and can be initialized with spectral embeddings.
    """
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        initial_output_embedding=None, # New argument for initialization
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

        # Apply spectral initialization to the encoder's output layer if provided
        # This assumes the encoder has an attribute 'output_layer' which is a nn.Linear
        if initial_output_embedding is not None and hasattr(self.encoder, 'output_layer'):
            print("Initializing encoder's output layer bias with mean of spectral embeddings...")
            # Initialize bias to the mean of the spectral embeddings
            # Ensure the bias data type matches the model's parameters
            self.encoder.output_layer.bias.data.copy_(torch.from_numpy(initial_output_embedding.mean(axis=0)).float())
            # For weights, a common practice is to leave them as default (e.g., Kaiming uniform for ReLU)
            # or use a small random initialization. Directly setting weights to match spectral output
            # for all inputs X is generally not feasible for a single linear layer unless X is simple.
            # The bias helps to center the initial embedding.

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # Always use UMAP loss
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
        encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
        self.log("umap_loss", encoder_loss, prog_bar=True)

        if self.decoder:
            recon = self.decoder(embedding_to)
            recon_loss = self.reconstruction_loss(recon, edges_to_exp)
            self.log("recon_loss", recon_loss, prog_bar=True)
            total_loss = encoder_loss + self.beta * recon_loss
            self.log("total_loss", total_loss, prog_bar=True)
            return total_loss
        else:
            self.log("total_loss", encoder_loss, prog_bar=True)
            return encoder_loss

class Datamodule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle data loading for training.
    """
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

# --- PUMAP class with spectral initialization ---
class PUMAP():
    """
    Parametric UMAP wrapper for training an encoder with PyTorch Lightning.
    Now includes options for spectral initialization.
    """
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
        num_workers=1,
        random_state=100,
        # Removed match_nonparametric_umap from PUMAP __init__ as it's always False now
        non_parametric_embeddings=None, # Still allowed if user wants to provide external embeddings
        init_pos='random', # 'spectral' or 'random'
        n_components=2, # Number of dimensions for the UMAP embedding
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
        self.non_parametric_embeddings = non_parametric_embeddings
        self.init_pos = init_pos
        self.n_components = n_components

    def fit(self, X, precomputed_graph=None, gradient_clip_val=None):
        """
        Fits the parametric UMAP model to the data.
        Handles graph computation and spectral initialization.
        """
        # Initialize PyTorch Lightning Trainer
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs, gradient_clip_val=gradient_clip_val)

        # Determine the graph to use for UMAP (either computed from X or precomputed)
        if precomputed_graph is None:
            # Compute UMAP graph from input data X
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
        else:
            # Use a precomputed graph (e.g., a distance matrix)
            graph = get_umap_graph(precomputed_graph, n_neighbors=self.n_neighbors, metric="precomputed", random_state=self.random_state)

        initial_output_embedding_for_model = None
        if self.init_pos == 'spectral':
            if self.non_parametric_embeddings is not None:
                # If user explicitly provided non_parametric_embeddings, use them directly
                print("Using provided non-parametric embeddings for initialization.")
                initial_output_embedding_for_model = self.non_parametric_embeddings
            else:
                print("Computing spectral initialization for encoder's output...")
                initialisation = spectral_layout( # Using the spectral_layout defined below
                    data=X, # spectral_layout needs data
                    graph=graph,
                    dim=self.n_components,
                    random_state=self.random_state,
                    metric=self.metric, # Pass metric
                    metric_kwds={} # Assuming no specific metric_kwds for spectral_layout here
                )

                # Apply scaling and noise as per init_embedding_from_graph
                expansion = 10.0 / np.abs(initialisation).max()
                rng = check_random_state(self.random_state) # Ensure random_state is a numpy.random.RandomState object
                initial_output_embedding_for_model = (initialisation * expansion).astype(np.float32) + \
                                                rng.normal(scale=0.0001, size=[graph.shape[0], self.n_components]).astype(np.float32)
                print(f"Spectral initialization (with noise and scaling) computed with shape: {initial_output_embedding_for_model.shape}")
        elif self.init_pos == 'random':
            print("Using random initialization for embeddings (encoder's default init).")
            # No specific initial_output_embedding_for_model needed for 'random' as it relies on encoder's default init.
            pass # initial_output_embedding_for_model remains None

        # Initialize the PyTorch Lightning Model
        self.model = Model(
            self.lr,
            self.encoder,
            decoder=self.decoder,
            min_dist=self.min_dist,
            initial_output_embedding=initial_output_embedding_for_model # Pass the initialization
        )

        # Always train with UMAPDataset and umap_loss
        print("Training neural network with UMAP loss.")
        trainer.fit(
            model=self.model,
            datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
        )

    @torch.no_grad()
    def transform(self, X):
        """
        Transforms high-dimensional data into the learned low-dimensional embedding.
        """
        # Ensure X is a tensor before passing to encoder
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        self.embedding_ = self.model.encoder(X).detach().cpu().numpy()
        return self.embedding_

    @torch.no_grad()
    def inverse_transform(self, Z):
        """
        Transforms low-dimensional embeddings back into the original data space (if decoder exists).
        """
        if self.model.decoder:
            # Ensure Z is a tensor before passing to decoder
            if not isinstance(Z, torch.Tensor):
                Z = torch.from_numpy(Z).float()
            return self.model.decoder(Z).detach().cpu().numpy()
        else:
            raise ValueError("Decoder not provided during PUMAP initialization.")

# --- Functions from umap.spectral (provided by user, now directly included) ---
# These functions are now part of this script's global scope.
def component_layout(
    data,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.

    n_components: int
        The number of distinct components to be layed out.

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.

    dim: int
        The chosen embedding dimension.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'

    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """
    if data is None:
        # We don't have data to work with; just guess
        return np.random.random(size=(n_components, dim)) * 10.0

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    if metric == "precomputed":
        # cannot compute centroids from precomputed distances
        # instead, compute centroid distances using linkage
        distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
        linkage = metric_kwds.get("linkage", "average")
        if linkage == "average":
            linkage = np.mean
        elif linkage == "complete":
            linkage = np.max
        elif linkage == "single":
            linkage = np.min
        else:
            raise ValueError(
                "Unrecognized linkage '%s'. Please choose from "
                "'average', 'complete', or 'single'" % linkage
            )
        for c_i in range(n_components):
            dm_i = data[component_labels == c_i]
            for c_j in range(c_i + 1, n_components):
                dist = linkage(dm_i[:, component_labels == c_j])
                distance_matrix[c_i, c_j] = dist
                distance_matrix[c_j, c_i] = dist
    else:
        for label in range(n_components):
            component_centroids[label] = data[component_labels == label].mean(axis=0)

        if scipy.sparse.isspmatrix(component_centroids):
            warn(
                "Forcing component centroids to dense; if you are running out of "
                "memory then consider increasing n_neighbors."
            )
            component_centroids = component_centroids.toarray()

        if metric in SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=metric,
                kwds=metric_kwds,
            )
        elif metric in SPARSE_SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=SPARSE_SPECIAL_METRICS[metric],
                kwds=metric_kwds,
            )
        else:
            if callable(metric) and scipy.sparse.isspmatrix(data):
                function_to_name_mapping = {
                    sparse_named_distances[k]: k
                    for k in set(SKLEARN_PAIRWISE_VALID_METRICS)
                    & set(sparse_named_distances.keys())
                }
                try:
                    metric_name = function_to_name_mapping[metric]
                except KeyError:
                    raise NotImplementedError(
                        "Multicomponent layout for custom "
                        "sparse metrics is not implemented at "
                        "this time."
                    )
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric_name, **metric_kwds
                )
            else:
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric, **metric_kwds
                )

    affinity_matrix = np.exp(-(distance_matrix**2))

    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    tol=0.0,
    maxiter=0,
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.

    graph: sparse matrix
        The adjacency matrix of the graph to be embedded.

    n_components: int
        The number of distinct components to be layed out.

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.

    dim: int
        The chosen embedding dimension.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = _spectral_layout(
                data=None, # Passed None as data is not used in _spectral_layout for graph embedding
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                init=init,
                tol=tol,
                maxiter=maxiter,
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )

    return result


def spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    tol=0.0,
    maxiter=0,
):
    """
    Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="random",
        tol=tol,
        maxiter=maxiter,
    )


def tswspectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    method=None,
    tol=0.0,
    maxiter=0,
):
    """Given a graph, compute the spectral embedding of the graph. This is
    simply the eigenvectors of the Laplacian of the graph. Here we use the
    normalized laplacian and a truncated SVD-based guess of the
    eigenvectors to "warm" up the eigensolver. This function should
    give results of similar accuracy to the spectral_layout function, but
    may converge more quickly for graph Laplacians that cause
    spectral_layout to take an excessive amount of time to complete.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    method: str (optional, default None, values either 'eigsh' or 'lobpcg')
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="tsvd",
        method=method,
        tol=tol,
        maxiter=maxiter,
    )


def _spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    method=None,
    tol=0.0,
    maxiter=0,
):
    """General implementation of the spectral embedding of the graph, derived as
    a subset of the eigenvectors of the normalized Laplacian of the graph. The numerical
    method for computing the eigendecomposition is chosen through heuristics.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    method: string -- either "eigsh" or "lobpcg" -- or None
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collection are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    if not method:
        method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"

    try:
        if init == "random":
            X = gen.normal(size=(L.shape[0], k))
        elif init == "tsvd":
            X = TruncatedSVD(
                n_components=k,
                random_state=random_state,
                # algorithm="arpack"
            ).fit_transform(L)
        else:
            raise ValueError(
                "The init parameter must be either 'random' or 'tsvd': "
                f"{init} is invalid."
            )
        # For such a normalized Laplacian, the first eigenvector is always
        # proportional to sqrt(degrees). We thus replace the first t-SVD guess
        # with the exact value.
        X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

        if method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol or 1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=maxiter or graph.shape[0] * 5,
            )
        elif method == "lobpcg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    L,
                    np.asarray(X),
                    largest=False,
                    tol=tol or 1e-4,
                    maxiter=maxiter or 5 * graph.shape[0],
                )
        else:
            raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning):
        warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))

# --- Example of how you might use the updated PUMAP class (for testing) ---
if __name__ == "__main__":
    # Mock Encoder with a hidden layer and exposed output layer
    class MockEncoder(nn.Module):
        def __init__(self, input_dim=10, output_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), # Added a hidden layer
                nn.ReLU(),
                nn.Linear(64, output_dim) # This is the final output layer
            )
            self.output_layer = self.net[-1] # Expose the final linear layer

        def forward(self, x):
            return self.net(x.float()) # Ensure float type

    class MockDecoder(nn.Module):
        def __init__(self, input_dim=2, output_dim=10):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.linear(x.float()) # Ensure float type

    # Generate some dummy data
    np.random.seed(42)
    dummy_data = np.random.rand(100, 10).astype(np.float32) # 100 samples, 10 features

    # Test with spectral initialization
    print("\n--- Testing PUMAP with Spectral Initialization ---")
    encoder_spectral = MockEncoder(input_dim=10, output_dim=2)
    decoder_spectral = MockDecoder(input_dim=2, output_dim=10)
    pumap_spectral = PUMAP(
        encoder=encoder_spectral,
        decoder=decoder_spectral,
        n_components=2, # Target dimensions for embedding
        init_pos='spectral', # Use spectral initialization
        epochs=5, # Reduced epochs for quick example
        batch_size=32,
        num_workers=0 # Set to 0 for simpler debugging on some systems
    )
    pumap_spectral.fit(dummy_data)
    transformed_data_spectral = pumap_spectral.transform(dummy_data) # Pass numpy array
    print(f"Transformed data shape (spectral init): {transformed_data_spectral.shape}")

    # Test with random initialization (direct UMAP loss)
    print("\n--- Testing PUMAP with Random Initialization (Direct UMAP Loss) ---")
    encoder_random = MockEncoder(input_dim=10, output_dim=2)
    pumap_random = PUMAP(
        encoder=encoder_random,
        n_components=2,
        init_pos='random', # Use random initialization
        epochs=5,
        batch_size=32,
        num_workers=0
    )
    pumap_random.fit(dummy_data)
    transformed_data_random = pumap_random.transform(dummy_data) # Pass numpy array
    print(f"Transformed data shape (random init): {transformed_data_random.shape}")

    # You can now use the `plot_tensorboard_logs` function (from the original script)
    # to visualize the loss curves if you enable the TensorBoardLogger in the Trainer.
    # For this example, we're just demonstrating the PUMAP class usage.
