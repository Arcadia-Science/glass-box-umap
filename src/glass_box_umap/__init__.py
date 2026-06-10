import importlib.metadata

from .core import GlassBoxUMAP
from .parametric_umap import ParametricUMAP
from .parametric_umap import logging_config as logging_config


def __getattr__(name: str) -> str:
    if name == "__version__":
        return importlib.metadata.version("glass-box-umap")
    raise AttributeError(name)


__all__ = [
    "GlassBoxUMAP",
    "ParametricUMAP",
]
