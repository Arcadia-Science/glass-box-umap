from torch import Tensor, nn
from torch.optim import AdamW

from .loss import find_ab_params, umap_loss


class UMAPModule(nn.Module):
    """An encoder together with the UMAP loss settings it is trained with.

    This is the fitted model a ``ParametricUMAP`` holds. Its ``state_dict`` has the same
    keys as the ``UMAPLightningModule`` used by version 0.1.0, so models saved with 0.1.0
    load into it.

    Args:
        lr: Learning rate for the AdamW optimizer.
        encoder: Neural network encoder mapping data to embedding space.
        input_dims: The expected input shape (excluding batch size).
        min_dist: UMAP min_dist parameter for computing a and b.
        negative_sample_rate: Number of negative samples per positive edge.
        repulsion_strength: Weighting of the repulsive term in the loss.
    """

    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        input_dims: tuple[int, ...],
        min_dist: float = 0.1,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 3.0,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.input_dims = input_dims
        self.min_dist = min_dist
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength

        self._a, self._b = find_ab_params(1.0, min_dist)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def configure_optimizer(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.lr)

    def training_loss(self, edges_to: Tensor, edges_from: Tensor) -> Tensor:
        """The UMAP loss for one batch of edges, as ``UMAPLightningModule.training_step``."""
        embedding_to = self.encoder(edges_to)
        embedding_from = self.encoder(edges_from)
        return umap_loss(
            embedding_to,
            embedding_from,
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
        )
