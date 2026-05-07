import copy
from collections.abc import Callable
from typing import Any

from torch import nn

from .models import ConvEncoder, DeepPReLUNet

DEFAULT_ENCODER = "default"

ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    DEFAULT_ENCODER: DeepPReLUNet,
    "default_conv": ConvEncoder,
}


def view_registry() -> dict[str, type[nn.Module]]:
    return copy.copy(ENCODER_REGISTRY)


def register_encoder(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Decorator to register a custom encoder class."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered.")
        ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def create_encoder(
    name: str,
    input_dims: tuple[int, ...],
    n_components: int,
    encoder_kwargs: dict[str, Any],
) -> nn.Module:
    """Factory function to instantiate an encoder from the registry.

    This function enforces the contract that all registered encoders must
    accept `input_dims` and `n_components` in their constructor.
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder '{name}' not found. Available encoders: {list(ENCODER_REGISTRY.keys())}"
        )

    cls = ENCODER_REGISTRY[name]

    try:
        return cls(input_dims=input_dims, n_components=n_components, **encoder_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate encoder '{name}'. \n"
            f"Ensure your encoder class accepts 'input_dims' and 'n_components' "
            f"as arguments, or use an adapter wrapper.\n"
            f"Original error: {e}"
        ) from e
