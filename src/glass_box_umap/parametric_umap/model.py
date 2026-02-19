import math

from torch import Tensor, nn


class ConvEncoder(nn.Module):
    """Convolutional encoder for image data.

    A CNN-based encoder designed for single-channel 28x28 images (e.g., MNIST).

    Args:
        n_components: Dimensionality of the output embedding space.
    """

    def __init__(self, n_components: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # FIX: Explicitly set in_channels=1 for MNIST
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),            
            nn.ReLU(),
            # FIX: in_channels must match the previous layer's out_channels (64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),            
            nn.ReLU(),
            nn.Flatten(),
            # 128 channels * 7 * 7 spatial = 6272
            nn.Linear(6272, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, n_components, bias=False),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class DefaultEncoder(nn.Module):
    """Default MLP encoder for flattened data.

    A simple feedforward encoder that flattens input and projects through
    hidden layers to the embedding space.

    Args:
        dims: Shape of input data (excluding batch dimension).
        n_components: Dimensionality of the output embedding space.
    """

    def __init__(self, dims: tuple[int, ...], n_components: int = 2) -> None:
        super().__init__()
        input_dim = math.prod(dims)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, n_components),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class DefaultDecoder(nn.Module):
    """Default MLP decoder for reconstruction.

    A symmetric decoder that projects from the embedding space back to the
    original data dimensions.

    Args:
        dims: Shape of output data (excluding batch dimension).
        n_components: Dimensionality of the input embedding space.

    Notes:
        - Currently not used in the codebase.
    """

    def __init__(self, dims: tuple[int, ...], n_components: int) -> None:
        super().__init__()
        self.dims = dims
        output_dim = math.prod(dims)
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x).view(x.shape[0], *self.dims)
