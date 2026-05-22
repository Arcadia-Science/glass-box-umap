import importlib.metadata

from .core import GlassBoxUMAP
from .parametric_umap import ParametricUMAP
from .parametric_umap import logging_config as logging_config

__version__ = importlib.metadata.version("glass-box-umap")

__all__ = [
    "GlassBoxUMAP",
    "ParametricUMAP",
    "__version__",
]
