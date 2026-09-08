from .check import check_package, get_plotting_requirements

for req in get_plotting_requirements():
    check_package(req)

from bokeh.io import output_notebook, show

from .bokeh import HierarchyLevel, HierarchySpec, LiveEmbeddingCallback, plot_embedding
from .mpl import plot_embedding_static

__all__ = [
    "HierarchyLevel",
    "HierarchySpec",
    "LiveEmbeddingCallback",
    "output_notebook",
    "plot_embedding",
    "plot_embedding_static",
    "show",
]
