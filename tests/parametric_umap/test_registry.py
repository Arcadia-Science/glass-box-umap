import pytest
import torch
from glass_box_umap.parametric_umap.registry import (
    ENCODER_REGISTRY,
    create_encoder,
    register_encoder,
)
from torch import nn


@pytest.fixture(autouse=True)
def clean_registry():
    original = ENCODER_REGISTRY.copy()
    yield
    ENCODER_REGISTRY.clear()
    ENCODER_REGISTRY.update(original)


class DummyEncoder(nn.Module):
    def __init__(self, input_dims: tuple[int, ...], n_components: int):
        super().__init__()
        self.input_dims = input_dims
        self.n_components = n_components
        self.linear = nn.Linear(input_dims[0], n_components)

    def forward(self, x):
        return self.linear(x)


class DummyEncoderWithKwargs(nn.Module):
    def __init__(self, input_dims: tuple[int, ...], n_components: int, hidden_size: int):
        super().__init__()
        self.input_dims = input_dims
        self.n_components = n_components
        self.hidden_size = hidden_size

    def forward(self, x):
        return x


class BadEncoder(nn.Module):
    def __init__(self, wrong_param: int):
        super().__init__()

    def forward(self, x):
        return x


def test_register_encoder():
    @register_encoder("test_encoder")
    class TestEncoder(DummyEncoder):
        pass

    assert "test_encoder" in ENCODER_REGISTRY
    assert ENCODER_REGISTRY["test_encoder"] is TestEncoder


def test_register_encoder_duplicate_raises():
    @register_encoder("duplicate")
    class First(DummyEncoder):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_encoder("duplicate")
        class Second(DummyEncoder):
            pass


def test_create_encoder():
    register_encoder("create_test")(DummyEncoder)

    encoder = create_encoder(
        name="create_test",
        input_dims=(784,),
        n_components=2,
        encoder_kwargs={},
    )

    assert isinstance(encoder, DummyEncoder)
    assert encoder.input_dims == (784,)
    assert encoder.n_components == 2


def test_create_encoder_with_kwargs():
    register_encoder("kwargs_test")(DummyEncoderWithKwargs)

    encoder = create_encoder(
        name="kwargs_test",
        input_dims=(784,),
        n_components=2,
        encoder_kwargs={"hidden_size": 128},
    )

    assert isinstance(encoder, DummyEncoderWithKwargs)
    assert encoder.hidden_size == 128


def test_create_encoder_not_found():
    with pytest.raises(ValueError, match="not found"):
        create_encoder(
            name="nonexistent",
            input_dims=(784,),
            n_components=2,
            encoder_kwargs={},
        )


def test_create_encoder_bad_signature():
    register_encoder("bad")(BadEncoder)

    with pytest.raises(TypeError, match="Failed to instantiate"):
        create_encoder(
            name="bad",
            input_dims=(784,),
            n_components=2,
            encoder_kwargs={},
        )


def test_default_encoder_registered():
    assert "default" in ENCODER_REGISTRY


@pytest.mark.parametrize(
    "input_dims",
    [
        (784,),
        (28, 28),
        (1, 28, 28),
        (3, 32, 32),
        (100,),
        (10, 10, 10),
    ],
)
def test_create_encoder_various_input_dims(input_dims: tuple[int, ...]):
    n_components = 2
    encoder = create_encoder(
        name="default",
        input_dims=input_dims,
        n_components=n_components,
        encoder_kwargs={},
    )

    batch_size = 4
    x = torch.randn(batch_size, *input_dims)
    output = encoder(x)

    assert output.shape == (batch_size, n_components)
