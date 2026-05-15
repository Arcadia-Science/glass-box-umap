import math

import torch
import torch.nn.init as init
from torch import Tensor, nn

from glass_box_umap.components import LayerNormDetached, VmapPReLU


class DeepPReLUNet(nn.Module):
    """A network with PReLU activation and LayerNormDetached."""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        hidden_size: int = 128,
        n_hidden_layers: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        input_size = math.prod(input_dims)
        self.flatten = nn.Flatten()

        layers = []
        for i in range(n_hidden_layers):
            in_dim = input_size if i == 0 else hidden_size

            layers.append(nn.Linear(in_dim, hidden_size, bias=False))
            layers.append(VmapPReLU())

            if i < n_hidden_layers - 1:
                layers.append(LayerNormDetached(hidden_size))

            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, n_components, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.flatten(x))


class ResidualMLPBlock(nn.Module):
    """Residual MLP block: x -> Linear -> Norm -> Act -> Linear -> Norm -> +skip.

    Keeps everything bias-free and piecewise-linear (if Norm has no affine).

    Args:
        dim: Input and output dimension.
        hidden_dim: Inner dimension. Defaults to ``dim``.
        use_norm: Whether to apply layer normalization.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

        # These norms don't introduce a bias (learned offset)
        self.norm1 = LayerNormDetached(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = LayerNormDetached(dim) if use_norm else nn.Identity()

        self.act = VmapPReLU()

        # He/Kaiming init for ReLU-family activations. Note, kaiming_uniform_ doesn't
        # accept a PReLU activation string, so we use `"leaky_relu"`.
        init.kaiming_uniform_(self.fc1.weight, a=0.25, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc2.weight, a=0.25, nonlinearity="leaky_relu")

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return x + h


def _compute_conv_output_size(
    input_dims: tuple[int, ...],
    conv_layers: nn.Module,
) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, *input_dims)
        out = conv_layers(dummy)
        return out.numel()


class ConvEncoder(nn.Module):
    """Convolutional encoder for image data.

    A CNN-based encoder that adapts to input dimensions.

    Args:
        input_dims: Shape of input data as (C, H, W).
        n_components: Dimensionality of the output embedding space.
        hidden_dims: Sizes of hidden layers in the MLP head.
    """

    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        in_channels = input_dims[0]

        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            VmapPReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            VmapPReLU(),
            nn.Flatten(),
        )

        flattened_size = _compute_conv_output_size(input_dims, self.conv_layers)

        mlp_layers: list[nn.Module] = []
        prev_dim = flattened_size
        for dim in hidden_dims:
            mlp_layers.extend([nn.Linear(prev_dim, dim, bias=False), VmapPReLU()])
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, n_components, bias=False))

        self.mlp = nn.Sequential(*mlp_layers)

        # Apply to all sub-modules recursively
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # Since you use PReLU, we set the 'a' parameter (negative slope)
            # to match. Default PReLU starts at 0.25.
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.25)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return self.mlp(x)


_DECODER_HIDDEN_DIMS = [100, 100, 100]


class DefaultDecoder(nn.Module):
    """Default MLP decoder for reconstruction.

    A symmetric decoder that projects from the embedding space back to the
    original data dimensions.

    Args:
        dims: Shape of output data (excluding batch dimension).
        n_components: Dimensionality of the input embedding space.
        hidden_dims: Sizes of hidden layers.

    Notes:
        - Unused and untested.
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        n_components: int,
        hidden_dims: list[int] = _DECODER_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.dims = dims
        output_dim = math.prod(dims)
        layers: list[nn.Module] = []
        prev_dim = n_components
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x).view(x.shape[0], *self.dims)
