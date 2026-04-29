from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from bokeh.layouts import column, row
from bokeh.models.layouts import LayoutDOM
from numpy.typing import NDArray

from ._bars import build_bars
from ._controls import build_controls
from ._data import (
    TOP_K_DISPLAY,
    compute_bar_views,
    precompute_top_features,
    select_top_features,
    validate_shapes,
)
from ._hover import HoverTooltips, resolve_hover
from ._scatter import build_scatter, make_scatter_source


def plot_embedding(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    *,
    group_names: Sequence[Any] | NDArray | None = None,
    feature_names: list[str] | None = None,
    feature_values: NDArray[np.floating] | None = None,
    top_k_global: int = 200,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Interactive 2D embedding scatter linked to a feature-contribution bar chart.

    A single radio toggle above the scatter chooses how to color the points:

    - ``Group`` (only available when ``group_names`` is provided): categorical
      coloring by user-supplied labels.
    - ``Feature``: a Viridis gradient over the L2-reduced contribution of one
      feature, picked via an autocomplete input that appears below the toggle
      (substring match, case-insensitive).
    - ``Top feature``: each sample is colored by the kept feature with its
      largest L2-reduced contribution. A slider lets the user choose the top-N
      most-frequent top features to colorize; samples whose top feature isn't
      in that set are drawn in gray underneath the colored points.

    Lasso- or box-selecting points in the scatter updates the linked bar chart
    on the right (which has its own ``L2 | normed L2 | Dim 1 | Dim 2`` view
    toggle); with no selection the bars summarize all samples.

    Args:
        Z:
            Embedding coordinates of shape ``(n_samples, 2)``.
        contributions:
            Per-feature contributions of shape ``(n_samples, 2, n_features)``.
            Typically the output of
            :meth:`~glass_box_umap.GlassBoxUMAP.compute_contributions` with
            ``reduction=None``.
        group_names:
            Group label per sample. Any sequence of length ``n_samples``;
            elements are stringified before use. When provided, the ``Group``
            color mode is added to the radio and used as the default; when
            ``None`` (default), the radio shows only ``Feature`` / ``Top
            feature`` and starts in ``Feature`` mode.
        feature_names:
            Human-readable name per feature; length must equal
            ``contributions.shape[2]``. Defaults to ``"Feature {i}"``
            (0-indexed).
        feature_values:
            Per-sample feature values of shape
            ``(n_samples, n_features)``. When provided, the default tooltip
            for ``Feature`` mode adds ``value: <X>`` (the picker-selected
            feature's value), and the default tooltip for ``Top feature``
            mode adds ``value: <X>`` (the top feature's value). Whatever
            scaling the caller passes is what the tooltip displays — pass
            raw values for human-readable tooltips, or the same
            standardized array fed to the embedder for consistency with
            contributions space. Ignored when ``hover_tooltips`` is set.
        top_k_global:
            How many features to ship to the browser, ranked by global L2
            importance. Caps everything: the bar chart, the feature-picker
            autocomplete, and the candidate set for top-feature ranking.
        hover_images:
            Per-sample uint8 image array of shape ``(n_samples, H, W)`` or
            ``(n_samples, H, W, 3 | 4)``. When set, each tooltip shows the
            sample's image above the default index/group text. Mutually
            exclusive with ``hover_tooltips`` and ``hover_data``.
        hover_tooltips:
            Bokeh tooltip HTML template that fully replaces the default. May
            reference ``@index``, ``@group`` (when ``group_names`` is
            provided), and any keys from ``hover_data``.
        hover_data:
            Extra columns merged into the scatter ``ColumnDataSource`` for
            reference from ``hover_tooltips``. Each value must have length
            ``n_samples``. Keys must not collide with the reserved columns
            ``x``, ``y``, ``index``, ``group``, ``color_value``,
            ``top_feature_group``, ``top_feature_name``, ``top_data_value``,
            ``picker_data_value``, ``sample_rank``.

    Returns:
        A Bokeh layout — color-by controls + scatter on the left, linked bar
        chart with view toggle on the right. Pass it to :func:`bokeh.io.show`
        or :func:`bokeh.io.save`.
    """
    validate_shapes(
        Z,
        contributions,
        feature_names=feature_names,
        group_names=group_names,
        feature_values=feature_values,
    )
    n_samples = Z.shape[0]

    top = select_top_features(contributions, feature_names, top_k_global, TOP_K_DISPLAY)
    views = compute_bar_views(contributions, top)
    top_feature_names_by_rank, sample_rank, top_kept_idx = precompute_top_features(
        views.l2, top.kept_names
    )
    n_distinct = len(top_feature_names_by_rank)

    has_groups = group_names is not None
    has_values = feature_values is not None
    color_modes = (["Group"] if has_groups else []) + ["Feature", "Top feature"]
    initial_mode = color_modes[0]

    initial_t = min(20, n_distinct)
    initial_top_group = np.asarray(
        [top_feature_names_by_rank[r] if r < initial_t else "(other)" for r in sample_rank]
    )
    initial_gradient = views.l2[:, 0].astype(np.float32).copy()
    top_feature_name = np.asarray([top.kept_names[int(k)] for k in top_kept_idx])

    extras: dict[str, NDArray[Any]] = {
        "color_value": initial_gradient,
        "top_feature_group": initial_top_group,
        "top_feature_name": top_feature_name,
        "sample_rank": sample_rank,
    }
    feature_values_kept: NDArray[np.floating] | None = None
    if has_values:
        feature_values_kept = feature_values[:, top.keep_idx].astype(np.float32)
        extras["top_data_value"] = feature_values_kept[np.arange(n_samples), top_kept_idx]
        extras["picker_data_value"] = feature_values_kept[:, 0].copy()
    if has_groups:
        extras["group"] = np.asarray([str(g) for g in group_names])

    base_body = "index: @index"
    if has_groups:
        base_body += " &nbsp;&middot;&nbsp; group: @group"
    sep = " &nbsp;&middot;&nbsp; "
    feature_body = base_body
    top_body = base_body
    if has_values:
        feature_body += sep + "value: @picker_data_value{0.000}"
        top_body += sep + "value: @top_data_value{0.000}"
    top_body += sep + "feature: @top_feature_name"
    default_bodies = HoverTooltips(group=base_body, feature=feature_body, top=top_body)
    tooltips, hover_extras = resolve_hover(
        default_bodies=default_bodies,
        hover_images=hover_images,
        hover_tooltips=hover_tooltips,
        hover_data=hover_data,
        n_samples=n_samples,
        occupied_keys=set(extras.keys()) | {"x", "y", "index"},
    )
    extras.update(hover_extras)

    scatter_source = make_scatter_source(Z, extras)

    scatter = build_scatter(
        scatter_source=scatter_source,
        tooltips=tooltips,
        top_feature_names_by_rank=top_feature_names_by_rank,
        n_distinct=n_distinct,
        initial_gradient=initial_gradient,
        initial_mode=initial_mode,
        group_names=group_names,
    )

    controls = build_controls(
        color_modes=color_modes,
        initial_mode=initial_mode,
        initial_t=initial_t,
        n_distinct=n_distinct,
        top=top,
        views=views,
        feature_values_kept=feature_values_kept,
        top_feature_names_by_rank=top_feature_names_by_rank,
        scatter_source=scatter_source,
        scatter=scatter,
    )

    bars = build_bars(
        views=views,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
    )

    return row(
        column(
            row(controls.color_by_prefix, controls.color_by_widget),
            controls.feature_picker,
            controls.top_n_slider,
            scatter.p_scatter,
            sizing_mode="stretch_both",
            styles={
                "background-color": "white",
                "flex": "0 0 60%",
                "min-width": "0",
            },
        ),
        bars,
        sizing_mode="stretch_both",
        styles={
            "max-width": "1100px",
            "aspect-ratio": "1100 / 720",
            "min-height": "500px",
            "max-height": "800px",
        },
    )
