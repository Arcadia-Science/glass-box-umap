import pytest
import torch
from glass_box_umap.parametric_umap.models import (
    ConvEncoder,
    DefaultDecoder,
    DefaultEncoder,
)


@pytest.mark.parametrize(
    ("input_dims", "n_components", "batch_size"),
    [
        ((784,), 2, 8),
        ((28, 28), 2, 8),
        ((1, 28, 28), 3, 4),
        ((3, 32, 32), 10, 16),
        ((100,), 5, 1),
    ],
)
def test_default_encoder_output_shape(
    input_dims: tuple[int, ...],
    n_components: int,
    batch_size: int,
):
    encoder = DefaultEncoder(input_dims=input_dims, n_components=n_components)
    x = torch.randn(batch_size, *input_dims)
    output = encoder(x)
    assert output.shape == (batch_size, n_components)


@pytest.mark.parametrize(
    "width",
    [64, 128, 256],
)
def test_default_encoder_custom_width(width: int):
    input_dims = (784,)
    n_components = 2
    batch_size = 4
    encoder = DefaultEncoder(
        input_dims=input_dims,
        n_components=n_components,
        width=width,
    )
    x = torch.randn(batch_size, *input_dims)
    output = encoder(x)
    assert output.shape == (batch_size, n_components)


@pytest.mark.parametrize(
    ("dims", "n_components", "batch_size"),
    [
        ((784,), 2, 8),
        ((28, 28), 2, 8),
        ((1, 28, 28), 3, 4),
        ((3, 32, 32), 10, 16),
    ],
)
def test_default_decoder_output_shape(
    dims: tuple[int, ...],
    n_components: int,
    batch_size: int,
):
    decoder = DefaultDecoder(dims=dims, n_components=n_components)
    x = torch.randn(batch_size, n_components)
    output = decoder(x)
    assert output.shape == (batch_size, *dims)


@pytest.mark.parametrize(
    "hidden_dims",
    [
        [64],
        [64, 128],
        [64, 128, 256],
    ],
)
def test_default_decoder_custom_hidden_dims(hidden_dims: list[int]):
    dims = (28, 28)
    n_components = 2
    batch_size = 4
    decoder = DefaultDecoder(
        dims=dims,
        n_components=n_components,
        hidden_dims=hidden_dims,
    )
    x = torch.randn(batch_size, n_components)
    output = decoder(x)
    assert output.shape == (batch_size, *dims)


@pytest.mark.parametrize(
    ("input_dims", "n_components", "batch_size"),
    [
        ((1, 28, 28), 2, 8),
        ((1, 32, 32), 2, 4),
        ((3, 32, 32), 10, 4),
        ((3, 64, 64), 5, 2),
        ((1, 28, 28), 128, 8),
    ],
)
def test_conv_encoder_output_shape(
    input_dims: tuple[int, ...],
    n_components: int,
    batch_size: int,
):
    encoder = ConvEncoder(input_dims=input_dims, n_components=n_components)
    x = torch.randn(batch_size, *input_dims)
    output = encoder(x)
    assert output.shape == (batch_size, n_components)


@pytest.mark.parametrize(
    "hidden_dims",
    [
        [256],
        [512, 256],
        [512, 512, 256],
    ],
)
def test_conv_encoder_custom_hidden_dims(hidden_dims: list[int]):
    input_dims = (1, 28, 28)
    n_components = 2
    batch_size = 4
    encoder = ConvEncoder(
        input_dims=input_dims,
        n_components=n_components,
        hidden_dims=hidden_dims,
    )
    x = torch.randn(batch_size, *input_dims)
    output = encoder(x)
    assert output.shape == (batch_size, n_components)
