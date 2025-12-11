import torch
import torch.nn.functional as F
from torch import Tensor


def convert_distance_to_probability(distances: Tensor, a: float = 1.0, b: float = 1.0) -> Tensor:
    """Convert distances to probabilities using UMAP's distance-to-probability function.

    Uses the formula: -log(1 + a * d^(2*b)) which creates a smooth decay from
    1 (at distance 0) to 0 (at large distances).

    Args:
        distances: Tensor of pairwise distances.
        a: UMAP hyperparameter controlling the spread of the distribution.
        b: UMAP hyperparameter controlling the decay rate.

    Returns:
        Tensor of log-probabilities (negative values, closer to 0 means higher probability).
    """
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph: Tensor,
    probabilities_distance: Tensor,
    repulsion_strength: float = 3.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute UMAP's cross-entropy loss between graph and distance probabilities.

    The loss encourages connected points (high graph probability) to be close in
    embedding space, while pushing unconnected points apart.

    Args:
        probabilities_graph: Binary indicators (1 for connected, 0 for unconnected).
        probabilities_distance: Log-probabilities from embedding distances.
        repulsion_strength: Weight for the repulsion term to balance attraction/repulsion.

    Returns:
        A tuple containing:
            - attraction_term: Loss for pulling connected points together
            - repellant_term: Loss for pushing unconnected points apart
            - ce_loss: Combined cross-entropy loss
    """
    attraction_term = -probabilities_graph * F.logsigmoid(probabilities_distance)
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (F.logsigmoid(probabilities_distance) - probabilities_distance)
        * repulsion_strength
    )
    ce_loss = attraction_term + repellant_term
    return attraction_term, repellant_term, ce_loss


def umap_loss(
    embedding_to: Tensor,
    embedding_from: Tensor,
    a: float,
    b: float,
    batch_size: int,
    negative_sample_rate: int = 5,
    device: str = "cuda",
) -> Tensor:
    """Compute UMAP loss with negative sampling.

    Computes the cross-entropy loss between positive pairs (connected in the graph)
    and negative pairs (randomly sampled unconnected points).

    Args:
        embedding_to: Embeddings of target vertices, shape (batch_size, n_components).
        embedding_from: Embeddings of source vertices, shape (batch_size, n_components).
        a: UMAP hyperparameter for probability conversion.
        b: UMAP hyperparameter for probability conversion.
        batch_size: Number of positive pairs in the batch.
        negative_sample_rate: Number of negative samples per positive pair.
        device: Device to place probability tensors on.

    Returns:
        Scalar loss tensor.
    """
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]

    distance_embedding = torch.cat(
        (
            (embedding_to - embedding_from).norm(dim=1),
            (embedding_neg_to - embedding_neg_from).norm(dim=1),
        ),
        dim=0,
    )

    probabilities_distance = convert_distance_to_probability(distance_embedding, a, b)
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)),
        dim=0,
    )

    _, _, ce_loss = compute_cross_entropy(
        probabilities_graph.to(device),
        probabilities_distance.to(device),
    )
    return torch.mean(ce_loss)
