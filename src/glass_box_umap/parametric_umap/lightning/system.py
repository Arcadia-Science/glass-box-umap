from collections.abc import Callable

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from umap.umap_ import find_ab_params

from ..loss import umap_loss


class UMAPLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training parametric UMAP.

    Supports two training modes:
    1. Standard UMAP training using edge-based loss
    2. Matching mode to reproduce non-parametric UMAP embeddings

    Args:
        lr: Learning rate for the AdamW optimizer.
        encoder: Neural network encoder mapping data to embedding space.
        decoder: Optional decoder for reconstruction loss.
        beta: Weight for reconstruction loss term.
        min_dist: UMAP min_dist parameter for computing a and b.
        reconstruction_loss: Loss function for reconstruction (if decoder provided).
    """

    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        beta: float = 1.0,
        min_dist: float = 0.1,
        reconstruction_loss: Callable[[Tensor, Tensor], Tensor] = binary_cross_entropy_with_logits,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
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
            edges_to_exp.shape[0],
            negative_sample_rate=5,
            device=self.device,
        )
        self.log("umap_loss", encoder_loss, prog_bar=True)

        if self.decoder is not None:
            recon = self.decoder(embedding_to)
            recon_loss = self.reconstruction_loss(recon, edges_to_exp)
            self.log("recon_loss", recon_loss, prog_bar=True)
            return encoder_loss + self.beta * recon_loss

        return encoder_loss
