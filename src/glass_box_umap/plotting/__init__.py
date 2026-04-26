from .check import check_package, get_plotting_requirements

for req in get_plotting_requirements():
    check_package(req)

from bokeh.io import output_notebook, show

from .bokeh import (
    plot_embedding_by_feature_gradient,
    plot_embedding_by_group,
    plot_embedding_by_top_feature,
)
from .mpl import plot_embedding

__all__ = [
    "output_notebook",
    "plot_embedding",
    "plot_embedding_by_feature_gradient",
    "plot_embedding_by_group",
    "plot_embedding_by_top_feature",
    "show",
]
