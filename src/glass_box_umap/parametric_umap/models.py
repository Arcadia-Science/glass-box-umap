import math

import torch
from torch import Tensor, nn

import torch.nn.init as init

from glass_box_umap.components import LayerNormDetached

DEFAULT_HIDDEN_DIMS = [100, 100, 100]


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
        input_dims=1, #: tuple[int, ...],
        n_components=2,#: int = 2,
        hidden_dims=None,#: list[int] | None = None,
    ) -> None:
        super().__init__()
        in_channels = input_dims[0]

        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Flatten(),
        )

        # flattened_size = _compute_conv_output_size(input_dims, self.conv_layers)

        mlp_layers: list[nn.Module] = []
        # prev_dim = flattened_size
        
        mlp_layers.extend([nn.LazyLinear(hidden_dims[0], bias=False), nn.ReLU()])
        for dim in hidden_dims[1:]:
            mlp_layers.extend([nn.LazyLinear(dim, bias=False), nn.ReLU()])
            # prev_dim = dim
        mlp_layers.append(nn.LazyLinear(n_components, bias=False))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return self.mlp(x)


class DefaultEncoder0(nn.Module):
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
            # layers.extend([nn.Linear(prev_dim, dim, bias=False), nn.ReLU()])
            layers.extend([nn.Linear(prev_dim, dim, bias=False)])
            # init.xavier_uniform_(layers[-1].weight)
            init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")

            # layers.extend([nn.ReLU()])
            layers.extend([nn.LeakyReLU(negative_slope=0.01)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_components, bias=False))
        # init.xavier_uniform_(layers[-1].weight)        
        init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")
        

        self.encoder = nn.Sequential(*layers)
        print(self.encoder)
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
import math
import torch
from torch import Tensor, nn
from torch.nn import init


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block: x -> Linear -> (Norm) -> Act -> Linear -> (Norm) -> +skip
    Keeps everything bias-free and piecewise-linear (if Norm has no affine).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: str = "relu",   # or "leaky_relu"
        negative_slope: float = 0.01,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

        # Norm that does NOT introduce an additive learned offset
        # self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False) if use_norm else nn.Identity()
        # self.norm2 = nn.LayerNorm(dim, elementwise_affine=False) if use_norm else nn.Identity()
        self.norm1 = LayerNormDetached(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = LayerNormDetached(dim) if use_norm else nn.Identity()

        if activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=negative_slope)
            a = negative_slope
            nonlin = "leaky_relu"
        else:
            self.act = nn.ReLU()
            a = 0.0
            nonlin = "relu"

        # He/Kaiming init for ReLU-family activations
        init.kaiming_uniform_(self.fc1.weight, a=a, nonlinearity=nonlin)
        init.kaiming_uniform_(self.fc2.weight, a=a, nonlinearity=nonlin)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return x + h


class DefaultEncoder(nn.Module):
    """
    Bias-free residual MLP encoder for tabular data.

    - Mean-centering/scaling done outside the module (recommended).
    - All Linear layers use bias=False to preserve your local Jacobian property.
    - Residual blocks greatly improve trainability for depth.
    """
    def __init__(
        self,
        input_dims: tuple[int, ...],
        n_components: int = 2,
        width: int = 128,
        depth: int = 3,              # number of residual blocks
        mlp_ratio: float = 2.0,      # inner expansion in each block
        activation: str = "leaky_relu",    # or "leaky_relu"
        negative_slope: float = 0.01,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        in_dim = math.prod(input_dims)
        hidden_dim = int(round(width * mlp_ratio))

        self.flatten = nn.Flatten()

        # Input projection (bias-free)
        self.in_proj = nn.Linear(in_dim, width, bias=False)

        # Init
        if activation == "leaky_relu":
            a = negative_slope
            nonlin = "leaky_relu"
        else:
            a = 0.0
            nonlin = "relu"
        init.kaiming_uniform_(self.in_proj.weight, a=a, nonlinearity=nonlin)

        # Residual trunk
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(
                dim=width,
                hidden_dim=hidden_dim,
                activation=activation,
                negative_slope=negative_slope,
                use_norm=use_norm,
            )
            for _ in range(depth)
        ])

        # Optional final activation (keeps piecewise linearity)
        self.out_act = nn.Identity()  # or nn.ReLU() / nn.LeakyReLU(...) if you want

        # Output projection (bias-free!)
        self.out_proj = nn.Linear(width, n_components, bias=False)
        init.kaiming_uniform_(self.out_proj.weight, a=a, nonlinearity=nonlin)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_act(x)
        x = self.out_proj(x)
        return x


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
