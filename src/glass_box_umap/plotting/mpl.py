import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.pyplot import get_cmap
from numpy.typing import NDArray


def make_cmap(n_colors: int = 40, seed: int = 42) -> ListedColormap:
    """Create a shuffled qualitative colormap by tiling tab20, tab20b, and tab20c.

    There are 60 unique colors across the three colormaps. If n_colors exceeds 60,
    colors are tiled (repeated) to reach the requested count, then shuffled.
    """
    base_colors = np.vstack(
        (
            get_cmap("tab20").colors,  # type: ignore
            get_cmap("tab20b").colors,  # type: ignore
            get_cmap("tab20c").colors,  # type: ignore
        )
    )
    tiles = (n_colors + len(base_colors) - 1) // len(base_colors)
    colors = np.tile(base_colors, (tiles, 1))[:n_colors]
    rng = np.random.default_rng(seed)
    colors = colors[rng.permutation(n_colors)]
    return ListedColormap(colors)


def plot_embedding(
    Z: NDArray[np.floating],
    group_ids: NDArray[np.integer] | None = None,
    group_names: list[str] | None = None,
    cmap: ListedColormap | None = None,
) -> Figure:
    """Scatter plot of a 2D embedding, optionally colored by group.

    Args:
        Z:
            Embedding coordinates with shape (n_samples, 2).
        group_ids:
            Integer group ID per point with shape (n_samples,). If None, points are
            uncolored.
        group_names:
            Human-readable name for each group, indexed by group ID. If provided,
            annotate group centroids. Requires group_ids.
        cmap:
            Colormap for the scatter plot. If None and group_ids are provided, a
            colormap is generated with one color per unique group.
    """
    if group_ids is not None and cmap is None:
        cmap = make_cmap(len(np.unique(group_ids)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=group_ids,
        cmap=cmap,
        s=0.1,
        marker=MarkerStyle("o", fillstyle="full"),
        alpha=0.5,
        rasterized=True,
    )

    if group_names is not None and group_ids is not None:
        texts = []
        for gid in np.unique(group_ids):
            mask = group_ids == gid
            cx, cy = np.median(Z[mask, 0]), np.median(Z[mask, 1])
            color = cmap(gid) if cmap is not None else "gray"
            texts.append(
                ax.text(
                    cx,
                    cy,
                    group_names[gid],
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
                )
            )
        adjust_text(texts, ax=ax)

    return fig
