import importlib.metadata

from .batch_size import BatchSizeCandidate, BatchSizeRecommendation, recommend_batch_size
from .core import GlassBoxUMAP
from .parametric_umap import ParametricUMAP
from .parametric_umap import logging_config as logging_config


def __getattr__(name: str) -> str:
    if name == "__version__":
        return importlib.metadata.version("glass-box-umap")
    raise AttributeError(name)


__all__ = [
    "BatchSizeCandidate",
    "BatchSizeRecommendation",
    "GlassBoxUMAP",
    "ParametricUMAP",
    "recommend_batch_size",
]