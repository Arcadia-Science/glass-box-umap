from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bokeh.models import (
    CategoricalColorMapper,
    CDSView,
    ColorBar,
    ColumnDataSource,
    CustomJSFilter,
    GroupFilter,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from numpy.typing import NDArray

from ._colors import nondegenerate_range, pick_palette
from ._hover import HoverTooltips
from ._js import NAMED_FILTER


@dataclass(frozen=True)
class ScatterArtifacts:
    """Bokeh objects produced by :func:`build_scatter` that downstream layers reference.

    Each field exists because at least one CustomJS callback (in the bar
    chart, or the color-by toggle, or the feature picker) needs to reach into
    the scatter to mutate visibility or update a color mapper. Bundling them
    keeps the orchestrator's call site narrow.

    Attributes:
        p_scatter: The scatter figure itself.
        top_other_glyph: Gray "(other)" points drawn under the named ones in
            "Top feature" mode.
        top_named_glyph: Colored points whose top feature is among the
            top-N most-frequent in "Top feature" mode.
        gradient_glyph: The single Viridis-mapped glyph used in "Feature" mode.
        color_bar: ColorBar paired with ``gradient_glyph``; visibility tracks
            the "Feature" mode toggle.
        group_glyphs: One glyph per unique group label (empty when no
            ``group_names`` were passed); visibility tracks the "Group" mode
            toggle.
        gradient_mapper: ``LinearColorMapper`` driving ``gradient_glyph``;
            mutated by the feature picker callback when the user changes
            features.
    """

    p_scatter: figure
    top_other_glyph: Any
    top_named_glyph: Any
    gradient_glyph: Any
    color_bar: ColorBar
    group_glyphs: list[Any]
    gradient_mapper: LinearColorMapper


def _base_figure() -> figure:
    tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select"
    return figure(
        title="Embedding — lasso or box-select to filter",
        sizing_mode="stretch_both",
        tools=tools,
    )


def make_scatter_source(
    Z: NDArray[np.floating],
    extras: dict[str, NDArray[Any]],
) -> ColumnDataSource:
    data: dict[str, Any] = {
        "x": Z[:, 0].astype(np.float32),
        "y": Z[:, 1].astype(np.float32),
        "index": np.arange(Z.shape[0], dtype=np.int32),
        **extras,
    }
    return ColumnDataSource(data)


def build_scatter(
    scatter_source: ColumnDataSource,
    tooltips: HoverTooltips,
    top_feature_names_by_rank: list[str],
    n_distinct: int,
    initial_gradient: NDArray[np.floating],
    initial_mode: str,
    group_names: Sequence[Any] | NDArray | None,
) -> ScatterArtifacts:
    """Assemble the scatter figure with all glyphs, color mapper, color bar, and hover tools.

    Three glyph layers map onto the three "Color by" modes — top-other and
    top-named for "Top feature", gradient for "Feature", per-group glyphs
    for "Group". Visibility is set so only the layers for ``initial_mode``
    are visible at first paint; the color-by toggle in :mod:`._controls`
    flips visibility on user interaction.
    """
    p_scatter = _base_figure()

    other_view = CDSView(
        filter=GroupFilter(column_name="top_feature_group", group="(other)"),
    )
    top_other_glyph = p_scatter.scatter(
        "x",
        "y",
        source=scatter_source,
        view=other_view,
        size=5,
        alpha=0.5,
        nonselection_alpha=0.1,
        color="#cccccc",
        visible=(initial_mode == "Top feature"),
    )

    named_filter = CustomJSFilter(code=NAMED_FILTER)
    named_view = CDSView(filter=named_filter)
    palette = pick_palette(max(n_distinct, 1))
    top_color_mapper = CategoricalColorMapper(
        factors=[*top_feature_names_by_rank, "(other)"],
        palette=[*palette, "#cccccc"],
    )
    top_named_glyph = p_scatter.scatter(
        "x",
        "y",
        source=scatter_source,
        view=named_view,
        size=5,
        alpha=0.6,
        nonselection_alpha=0.1,
        color={"field": "top_feature_group", "transform": top_color_mapper},
        visible=(initial_mode == "Top feature"),
    )

    has_groups = group_names is not None
    group_glyphs: list[Any] = []
    if has_groups:
        factors = sorted({str(g) for g in group_names})
        group_palette = pick_palette(len(factors))
        for factor, color in zip(factors, group_palette, strict=False):
            view = CDSView(filter=GroupFilter(column_name="group", group=factor))
            group_glyphs.append(
                p_scatter.scatter(
                    "x",
                    "y",
                    source=scatter_source,
                    view=view,
                    size=5,
                    alpha=0.6,
                    nonselection_alpha=0.1,
                    color=color,
                    visible=(initial_mode == "Group"),
                )
            )

    init_lo, init_hi = nondegenerate_range(
        float(initial_gradient.min()), float(initial_gradient.max())
    )
    gradient_mapper = LinearColorMapper(palette=Viridis256, low=init_lo, high=init_hi)
    gradient_glyph = p_scatter.scatter(
        "x",
        "y",
        source=scatter_source,
        size=5,
        alpha=0.6,
        nonselection_alpha=0.1,
        color={"field": "color_value", "transform": gradient_mapper},
        visible=(initial_mode == "Feature"),
    )
    color_bar = ColorBar(
        color_mapper=gradient_mapper,
        location=(0, 0),
        visible=(initial_mode == "Feature"),
    )
    p_scatter.add_layout(color_bar, "right")

    p_scatter.add_tools(HoverTool(tooltips=tooltips.feature, renderers=[gradient_glyph]))
    p_scatter.add_tools(
        HoverTool(tooltips=tooltips.top, renderers=[top_named_glyph, top_other_glyph]),
    )
    if has_groups:
        p_scatter.add_tools(HoverTool(tooltips=tooltips.group, renderers=group_glyphs))

    return ScatterArtifacts(
        p_scatter=p_scatter,
        top_other_glyph=top_other_glyph,
        top_named_glyph=top_named_glyph,
        gradient_glyph=gradient_glyph,
        color_bar=color_bar,
        group_glyphs=group_glyphs,
        gradient_mapper=gradient_mapper,
    )
