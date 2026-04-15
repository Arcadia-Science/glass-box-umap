from .check import check_package, get_plotting_requirements

for req in get_plotting_requirements():
    check_package(req)

from .mpl import plot_embedding

__all__ = [
    "plot_embedding",
]
