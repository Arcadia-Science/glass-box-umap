import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer
from umap.umap_ import find_ab_params

from ..loss import umap_loss


class UMAPLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training parametric UMAP.

    Supports two training modes:
    1. Standard UMAP training using edge-based loss
    2. Matching mode to reproduce non-parametric UMAP embeddings

    Args:
        lr:
            Learning rate for the AdamW optimizer.
        encoder:
            Neural network encoder mapping data to embedding space.
        input_dims:
            The expected input shape (excluding batch size). Stored for validation
            during inference/refitting.
        min_dist:
            UMAP min_dist parameter for computing a and b.
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
        self.save_hyperparameters(ignore=["encoder"])

        self.lr = lr
        self.encoder = encoder
        self.input_dims = input_dims
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength

        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        edges_to_exp, edges_from_exp = batch
        embedding_to = self.encoder(edges_to_exp)
        embedding_from = self.encoder(edges_from_exp)

        encoder_loss = umap_loss(
            embedding_to,
            embedding_from,
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
        )
        self.log("umap_loss", encoder_loss, prog_bar=True)

        return encoder_loss
