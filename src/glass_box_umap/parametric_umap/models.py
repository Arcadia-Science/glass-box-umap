import math

import torch
from torch import Tensor, nn

DEFAULT_HIDDEN_DIMS = [200, 200, 200]


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
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        flattened_size = _compute_conv_output_size(input_dims, self.conv_layers)

        mlp_layers: list[nn.Module] = []
        prev_dim = flattened_size
        for dim in hidden_dims:
            mlp_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, n_components))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return self.mlp(x)


class DefaultEncoder(nn.Module):
    """Default MLP encoder for flattened data.

    A simple feedforward encoder that flattens input and projects through
    hidden layers to the embedding space.

    Args:
        input_dims: Shape of input data (excluding batch dimension).
        n_components: Dimensionality of the output embedding space.
        hidden_dims: Sizes of hidden layers.
    """

    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        hidden_dims: list[int] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = math.prod(input_dims)
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_components))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class DefaultDecoder(nn.Module):
    """Default MLP decoder for reconstruction.

    A symmetric decoder that projects from the embedding space back to the
    original data dimensions.

    Args:
        dims: Shape of output data (excluding batch dimension).
        n_components: Dimensionality of the input embedding space.
        hidden_dims: Sizes of hidden layers.

    Notes:
        - Currently not used in the codebase.
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        n_components: int,
        hidden_dims: list[int] = DEFAULT_HIDDEN_DIMS,
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
