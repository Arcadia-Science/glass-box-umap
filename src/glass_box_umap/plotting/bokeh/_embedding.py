import warnings
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models.layouts import LayoutDOM
from numpy.typing import NDArray

from ._bars import build_bars
from ._controls import build_controls
from ._data import (
    TOP_K_DISPLAY,
    collapse_position_features,
    compute_bar_views,
    precompute_top_features,
    select_top_features,
    validate_shapes,
)
from ._hierarchy import HierarchySpec, validate_hierarchies
from ._hover import HoverTooltips, resolve_hover
from ._scatter import OutputBackend, build_scatter, make_scatter_source

_LARGE_DATASET_WARN_THRESHOLD = 100_000


def plot_embedding(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    *,
    group_names: Sequence[Any] | NDArray | None = None,
    label_sets: Mapping[str, Sequence[Any] | NDArray] | None = None,
    feature_names: list[str] | None = None,
    feature_values: NDArray[np.floating] | None = None,
<<<<<<< Updated upstream
    feature_color_views: Mapping[str, NDArray[np.floating]] | None = None,
=======
<<<<<<< Updated upstream
=======
    feature_color_views: Mapping[str, NDArray[np.floating]] | None = None,
    hierarchies: Sequence[HierarchySpec] | None = None,
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    top_k_global: int = 200,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
    output_backend: OutputBackend = "webgl",
) -> LayoutDOM:
    """Interactive 2D embedding scatter linked to a feature-contribution bar chart.

    A single radio toggle above the scatter chooses how to color the points:

    - Named categorical labels (available through ``group_names`` or
      ``label_sets``): coloring by user-supplied labels.
    - ``Feature``: a Viridis gradient over the L2-reduced contribution of one
      feature, picked via an autocomplete input that appears below the toggle
      (substring match, case-insensitive).
    - ``Top feature``: each sample is colored by the feature with its largest
      L2-reduced contribution. A slider lets the user choose the top-N
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
        label_sets:
            Optional named categorical label sets. Each mapping value is a
            sequence of length ``n_samples``. Each name becomes a separate
            color mode, allowing the same embedding to be colored by several
            annotations. ``group_names`` remains supported as a legacy
            ``"Group"`` label set.
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
<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
        feature_color_views:
            Optional named ``(n_samples, n_features)`` reduced-contribution
            matrices for the Feature color map. The default ``Raw`` view is
            always available. These views do not change the exact bar chart
            or per-point contribution values.
<<<<<<< Updated upstream
=======
        hierarchies:
            Optional named hierarchical categorical views. Each hierarchy
            supplies ordered cuts that are selected with a discrete slider.
            Hierarchy coloring does not alter coordinates or contribution bars.
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        top_k_global:
            How many full contribution columns to ship to the browser, ranked
            by global L2 importance. This caps the bar chart and feature-picker
            autocomplete; compact Top feature winner IDs still consider every
            input feature.
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
        output_backend:
            Bokeh rendering backend for the scatter. Defaults to ``"webgl"``,
            which offloads rendering to the GPU and stays smooth at high
            sample counts. Switch to ``"canvas"`` if the GPU/driver/browser
            combination renders the plot incorrectly (e.g. blank canvas,
            wrong-sized points, or color banding) — canvas is slower but
            uses CPU rasterization and works on any setup that supports
            Bokeh at all.

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
        label_sets=label_sets,
        feature_values=feature_values,
    )
    n_samples = Z.shape[0]
<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
=======
    validate_hierarchies(hierarchies, n_samples)
>>>>>>> Stashed changes
    if feature_color_views is not None:
        for name, view in feature_color_views.items():
            if view.shape != (n_samples, contributions.shape[2]):
                raise ValueError(
                    f"feature color view {name!r} has shape {view.shape}; expected "
                    f"{(n_samples, contributions.shape[2])}."
                )
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    if n_samples >= _LARGE_DATASET_WARN_THRESHOLD:
        warnings.warn(
            f"plot_embedding received {n_samples:,} samples; at this size browser "
            f"memory and lasso/box-select responsiveness may suffer. Consider "
            f"lowering top_k_global or downsampling Z.",
            UserWarning,
            stacklevel=2,
        )

    resolved_feature_names = (
        feature_names
        if feature_names is not None
        else [f"Feature {i}" for i in range(contributions.shape[2])]
    )
    top = select_top_features(contributions, feature_names, top_k_global, TOP_K_DISPLAY)
    views = compute_bar_views(contributions, top)
    top_feature_names_by_rank, sample_rank, top_feature_idx = precompute_top_features(
        top.reduced, resolved_feature_names
    )
    n_distinct = len(top_feature_names_by_rank)

<<<<<<< Updated upstream
    named_labels: dict[str, Sequence[Any] | NDArray] = {}
    if group_names is not None:
        named_labels["Group"] = group_names
    if label_sets is not None:
        overlap = set(named_labels).intersection(label_sets)
        if overlap:
            raise ValueError(f"label_sets duplicates reserved label names: {sorted(overlap)}")
        named_labels.update(label_sets)
    label_fields = {name: f"label_set_{i}" for i, name in enumerate(named_labels)}
    has_groups = bool(named_labels)
    has_values = feature_values is not None
    color_modes = (["Label"] if has_groups else []) + ["Feature", "Top feature"]
=======
<<<<<<< Updated upstream
    has_groups = group_names is not None
    has_values = feature_values is not None
    color_modes = (["Group"] if has_groups else []) + ["Feature", "Top feature"]
=======
    collapsed = collapse_position_features(contributions, resolved_feature_names, feature_values)
    collapsed_reduced = np.sqrt(np.square(collapsed.contributions).sum(axis=1, dtype=np.float32))
    (
        collapsed_names_by_rank,
        collapsed_sample_rank,
        collapsed_top_feature_idx,
    ) = precompute_top_features(collapsed_reduced, collapsed.names)
    top_factor_names = list(dict.fromkeys([*top_feature_names_by_rank, *collapsed_names_by_rank]))

    named_labels: dict[str, Sequence[Any] | NDArray] = {}
    if group_names is not None:
        named_labels["Group"] = group_names
    if label_sets is not None:
        overlap = set(named_labels).intersection(label_sets)
        if overlap:
            raise ValueError(f"label_sets duplicates reserved label names: {sorted(overlap)}")
        named_labels.update(label_sets)
    label_fields = {name: f"label_set_{i}" for i, name in enumerate(named_labels)}
    has_groups = bool(named_labels)
    has_values = feature_values is not None
    color_modes = (["Label"] if has_groups else []) + ["Feature", "Top feature"]
    has_hierarchy = bool(hierarchies)
    if has_hierarchy:
        color_modes.append("Hierarchy")
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    initial_mode = color_modes[0]

    initial_t = min(20, n_distinct)
    names_with_other = np.asarray([*top_feature_names_by_rank, "(other)"])
    clipped_rank = np.where(sample_rank < initial_t, sample_rank, len(top_feature_names_by_rank))
    initial_top_group = names_with_other[clipped_rank]
    initial_gradient = views.l2[:, 0].astype(np.float32).copy()
    top_feature_name = np.asarray(resolved_feature_names)[top_feature_idx]

    extras: dict[str, NDArray[Any]] = {
        "color_value": initial_gradient,
        "top_feature_group": initial_top_group,
        "top_feature_name": top_feature_name,
        "sample_rank": sample_rank,
        # Updated by the categorical checkboxes. All non-label renderers and
        # linked summaries use this mask, so a label subset persists when the
        # user switches color modes.
        "subset_visible": np.ones(n_samples, dtype=np.uint8),
    }
    hierarchy_sources: dict[str, ColumnDataSource] = {}
    hierarchy_level_names: dict[str, list[str]] = {}
    hierarchy_colors: dict[str, list[dict[str, str]]] = {}
    hierarchy_metadata: dict[str, list[dict[str, dict[str, Any]]]] = {}
    hierarchy_key_html = ""
    if hierarchies:
        for hierarchy in hierarchies:
            hierarchy_sources[hierarchy.name] = ColumnDataSource(
                {
                    f"level_{i}": np.asarray(level.labels).astype(str)
                    for i, level in enumerate(hierarchy.levels)
                }
            )
            hierarchy_level_names[hierarchy.name] = [level.name for level in hierarchy.levels]
            hierarchy_colors[hierarchy.name] = [dict(level.colors) for level in hierarchy.levels]
            hierarchy_metadata[hierarchy.name] = [
                {str(label): dict(metadata) for label, metadata in level.metadata.items()}
                for level in hierarchy.levels
            ]

        initial_hierarchy = hierarchies[0]
        initial_level = initial_hierarchy.levels[0]
        initial_labels = np.asarray(initial_level.labels).astype(str)
        extras["hierarchy_cluster"] = initial_labels.copy()
        extras["hierarchy_color"] = np.asarray(
            [initial_level.colors[label] for label in initial_labels]
        )
        extras["hierarchy_size"] = np.asarray(
            [str(initial_level.metadata[label].get("size", "")) for label in initial_labels]
        )
        extras["hierarchy_top_families"] = np.asarray(
            [str(initial_level.metadata[label].get("top_families", "")) for label in initial_labels]
        )
        key_rows = []
        for label, metadata in initial_level.metadata.items():
            key_rows.append(
                '<div style="margin:3px 0">'
                f'<span style="display:inline-block;width:10px;height:10px;background:'
                f'{escape(initial_level.colors[label])};margin-right:5px"></span>'
                f"<b>{escape(label)}</b> ({escape(str(metadata.get('size', '')))}) — "
                f"{escape(str(metadata.get('top_families', '')))}</div>"
            )
        hierarchy_key_html = (
            f"<b>{escape(initial_hierarchy.name)} — {escape(initial_level.name)}</b>"
            '<div style="max-height:150px;overflow-y:auto;margin-top:4px">'
            + "".join(key_rows)
            + "</div>"
        )
    feature_values_kept: NDArray[np.floating] | None = None
    if has_values:
        feature_values_kept = feature_values[:, top.keep_idx].astype(np.float32)
        extras["top_data_value"] = feature_values[np.arange(n_samples), top_feature_idx]
        extras["picker_data_value"] = feature_values_kept[:, 0].copy()
    if has_groups:
        # ``group`` preserves the legacy tooltip field; named sets also get
        # independent source columns so their glyph filters can switch live.
        extras["group"] = np.asarray(next(iter(named_labels.values()))).astype(str)
        for name, labels in named_labels.items():
            extras[label_fields[name]] = np.asarray(labels).astype(str)

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

    top_mode_sources = {
        "Expanded": ColumnDataSource(
            {
                "top_idx": top_feature_idx.astype(np.int32),
                "sample_rank": sample_rank.astype(np.int32),
                **(
                    {"top_value": feature_values[np.arange(n_samples), top_feature_idx]}
                    if feature_values is not None
                    else {}
                ),
            }
        ),
        "Collapsed": ColumnDataSource(
            {
                "top_idx": collapsed_top_feature_idx.astype(np.int32),
                "sample_rank": collapsed_sample_rank.astype(np.int32),
                **(
                    {"top_value": collapsed.values[np.arange(n_samples), collapsed_top_feature_idx]}
                    if collapsed.values is not None
                    else {}
                ),
            }
        ),
    }
    top_rank_sources = {
        "Expanded": ColumnDataSource({"name": top_feature_names_by_rank}),
        "Collapsed": ColumnDataSource({"name": collapsed_names_by_rank}),
    }

    l2_source = ColumnDataSource({f"c{k}": views.l2[:, k] for k in range(top.n_kept)})
    feature_view_sources = {"Raw": l2_source}
    if feature_color_views is not None:
        for name, view in feature_color_views.items():
            if name == "Raw":
                raise ValueError("feature_color_views must not redefine the reserved 'Raw' view.")
            feature_view_sources[name] = ColumnDataSource(
                {f"c{k}": view[:, top.keep_idx].astype(np.float32) for k in range(top.n_kept)}
            )
    feature_values_source: ColumnDataSource | None = None
    if feature_values_kept is not None:
        feature_values_source = ColumnDataSource(
            {f"c{k}": feature_values_kept[:, k] for k in range(top.n_kept)}
        )

    scatter = build_scatter(
        scatter_source=scatter_source,
        tooltips=tooltips,
        top_feature_names_by_rank=top_factor_names,
        n_distinct=len(top_factor_names),
        initial_gradient=initial_gradient,
        initial_mode=initial_mode,
<<<<<<< Updated upstream
        initial_label_mode=next(iter(named_labels), None),
        label_sets={name: (label_fields[name], labels) for name, labels in named_labels.items()},
=======
<<<<<<< Updated upstream
        group_names=group_names,
=======
        initial_label_mode=next(iter(named_labels), None),
        label_sets={name: (label_fields[name], labels) for name, labels in named_labels.items()},
        has_hierarchy=has_hierarchy,
        hierarchy_tooltip=(
            tooltips.group
            + "<br><b>Hierarchy cluster</b>: @hierarchy_cluster"
            + "<br>cluster size: @hierarchy_size"
            + "<br>top families: @hierarchy_top_families"
            if has_hierarchy
            else tooltips.group
        ),
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        output_backend=output_backend,
    )

    controls = build_controls(
        color_modes=color_modes,
        initial_mode=initial_mode,
        initial_t=initial_t,
<<<<<<< Updated upstream
        label_modes=list(named_labels),
        label_factors={
            name: sorted({str(value) for value in labels})
            for name, labels in named_labels.items()
        },
=======
<<<<<<< Updated upstream
=======
        label_modes=list(named_labels),
        label_factors={
            name: sorted({str(value) for value in labels}) for name, labels in named_labels.items()
        },
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        n_distinct=n_distinct,
        top=top,
        l2_source=l2_source,
        feature_view_sources=feature_view_sources,
        feature_values_source=feature_values_source,
        top_feature_names_by_rank=top_feature_names_by_rank,
        collapsed_top_feature_names_by_rank=collapsed_names_by_rank,
        top_mode_sources=top_mode_sources,
        top_rank_sources=top_rank_sources,
        expanded_top_names=resolved_feature_names,
        collapsed_top_names=collapsed.names,
        scatter_source=scatter_source,
        scatter=scatter,
        hierarchy_sources=hierarchy_sources,
        hierarchy_level_names=hierarchy_level_names,
        hierarchy_colors=hierarchy_colors,
        hierarchy_metadata=hierarchy_metadata,
        hierarchy_key_html=hierarchy_key_html,
    )

    bars = build_bars(
        views=views,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
        l2_source=l2_source,
    )

    return row(
        column(
            row(controls.color_by_prefix, controls.color_by_widget),
            controls.feature_picker,
<<<<<<< Updated upstream
            controls.feature_view_picker,
            controls.label_picker,
            controls.label_class_filter,
            controls.top_n_slider,
            row(controls.point_size_slider, controls.point_alpha_slider),
=======
<<<<<<< Updated upstream
            controls.top_n_slider,
=======
            controls.feature_view_picker,
            controls.label_picker,
            controls.label_class_filter,
            controls.select_none_checkbox,
            controls.top_n_slider,
            controls.collapse_checkbox,
            controls.color_key,
            *(
                (
                    controls.hierarchy_source_picker,
                    controls.hierarchy_level_slider,
                    controls.hierarchy_key,
                )
                if has_hierarchy
                else ()
            ),
            row(controls.point_size_slider, controls.point_alpha_slider),
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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
            "width": "100%",
            "height": "100vh",
            "min-height": "600px",
        },
    )
