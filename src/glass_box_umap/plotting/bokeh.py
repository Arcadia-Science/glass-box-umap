import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, overload

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    CDSView,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    FactorRange,
    GroupFilter,
    HoverTool,
    LinearColorMapper,
    RadioButtonGroup,
)
from bokeh.models.layouts import LayoutDOM
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.plotting import figure
from numpy.typing import NDArray
from PIL import Image

from ..jacobian import groups_from_top_features, reduce_contributions

BAR_COLOR_REDUCED = "#756bb1"

_FIGURE_HEIGHT = 600
_HOVER_IMAGE_LONGEST_SIDE = 64
_RESERVED_HOVER_KEYS = frozenset({"x", "y", "index", "group", "color_value", "__hover_image"})

_VIEW_LABELS = ("L2", "normed L2", "Dim 1", "Dim 2")

_LINKED_BARS_CALLBACK_JS = """
const view = view_widget.active;

let active_data;
if (view === 0) {
    active_data = contrib_sources[0].data;
} else if (view === 1) {
    if (!normed_l2_cache.data) {
        const src = contrib_sources[0].data;
        const totals = new Float64Array(n_samples);
        for (let k = 0; k < n_kept; k++) {
            const col = src["c" + k];
            for (let i = 0; i < n_samples; i++) {
                totals[i] += col[i];
            }
        }
        for (let i = 0; i < n_samples; i++) {
            if (totals[i] < 1e-12) totals[i] = 1.0;
        }
        const cache = {};
        for (let k = 0; k < n_kept; k++) {
            const s = src["c" + k];
            const dst = new Float64Array(n_samples);
            for (let i = 0; i < n_samples; i++) {
                dst[i] = s[i] / totals[i];
            }
            cache["c" + k] = dst;
        }
        normed_l2_cache.data = cache;
    }
    active_data = normed_l2_cache.data;
} else if (view === 2) {
    active_data = contrib_sources[1].data;
} else {
    active_data = contrib_sources[2].data;
}

const sel = scatter_source.selected.indices;
const indices = sel.length
    ? sel
    : Array.from({length: n_samples}, (_, i) => i);
const n = indices.length;

const means = new Float64Array(n_kept);
for (let k = 0; k < n_kept; k++) {
    const col = active_data["c" + k];
    let s = 0.0;
    for (let j = 0; j < n; j++) {
        s += col[indices[j]];
    }
    means[k] = s / n;
}

const scored = new Array(n_kept);
for (let k = 0; k < n_kept; k++) {
    scored[k] = { idx: k, score: Math.abs(means[k]) };
}
scored.sort((a, b) => b.score - a.score);
const top = scored.slice(0, display_k).reverse();

const feat = top.map(t => feature_names[t.idx]);
const vals = top.map(t => means[t.idx]);
bar_source.data = { feature: feat, mean: vals };
bar_range.factors = feat;

heading_div.text = `<b>Mean contribution</b> — ${view_labels[view]}`;
"""


@dataclass(frozen=True)
class TopFeatures:
    """Result of ranking features by global L2 importance and selecting the top slice.

    Attributes:
        kept_names: Feature names for the kept pool, ordered by descending
            global L2 importance.
        keep_idx: Indices into the original feature axis, in the same order
            as ``kept_names``. Length equals ``n_kept``.
        display_k: Number of features to draw in the bar chart at any one
            time. Always ``<= n_kept``.
        reduced: The full ``(n_samples, n_features)`` L2-reduced contributions
            array — not sliced by ``keep_idx``.
        n_kept: (property) Size of the kept pool, i.e. ``len(kept_names)``.
    """

    kept_names: list[str]
    keep_idx: NDArray[np.integer]
    display_k: int
    reduced: NDArray[np.floating]

    @property
    def n_kept(self) -> int:
        return len(self.kept_names)


@dataclass(frozen=True)
class BarViews:
    """Three pre-computed ``(n_samples, n_kept)`` views of the kept-feature pool.

    All three are signed values. The "normed L2" view shown in the bar chart
    is derived JS-side from ``l2`` as ``value / max(Σ_k value, ε)`` per sample
    (so each sample's row sums to 1 across the kept pool), cached after first
    computation.

    Attributes:
        l2: L2-reduced contributions, sliced to ``top.keep_idx``. Always non-negative.
        d0: Signed contributions to embedding dimension 0, sliced to ``top.keep_idx``.
        d1: Signed contributions to embedding dimension 1, sliced to ``top.keep_idx``.
    """

    l2: NDArray[np.floating]
    d0: NDArray[np.floating]
    d1: NDArray[np.floating]


def _pick_palette(n: int) -> list[str]:
    if n <= 10:
        return list(Category10[10])[:n]
    base = list(Category20[20])
    tiles = (n + len(base) - 1) // len(base)
    return (base * tiles)[:n]


def _validate_shapes(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    group_names: Sequence[Any] | NDArray | None = None,
) -> None:
    """Validate shape/length invariants shared by the public plot functions.

    Raises:
        ValueError: If ``Z`` is not ``(n_samples, 2)``, ``contributions`` is
            not ``(n_samples, 2, n_features)`` with ``n_features >= 1``, or
            ``feature_names`` / ``group_names`` (when provided) don't match
            the corresponding axes.
    """
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (n_samples, 2); got {Z.shape}.")
    n_samples = Z.shape[0]

    if contributions.ndim != 3:
        raise ValueError(
            "contributions must have 3 dimensions (n_samples, 2, n_features); "
            f"got shape {contributions.shape}."
        )
    if contributions.shape[0] != n_samples:
        raise ValueError(
            f"contributions.shape[0] ({contributions.shape[0]}) must equal "
            f"Z.shape[0] ({n_samples})."
        )
    if contributions.shape[1] != 2:
        raise ValueError(f"contributions.shape[1] must be 2; got {contributions.shape[1]}.")
    if contributions.shape[2] < 1:
        raise ValueError("contributions must have at least one feature.")

    n_features = contributions.shape[2]
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has length {len(feature_names)}, but contributions "
            f"has {n_features} features."
        )

    if group_names is not None and len(group_names) != n_samples:
        raise ValueError(
            f"group_names has length {len(group_names)}, but Z has {n_samples} samples."
        )


def _select_top_features(
    contributions: NDArray[np.floating],
    feature_names: list[str] | None,
    top_k_global: int,
    top_k_display: int,
) -> TopFeatures:
    """Rank features by global L2 importance and select the top slice for plotting.

    Contributions are first collapsed across embedding dimensions via
    :func:`reduce_contributions` (L2 norm), producing a
    ``(n_samples, n_features)`` array. The per-feature scalar is the mean of
    that across samples.

    The top ``top_k_global`` features by this scalar are retained as the
    pool shipped to the browser; ``top_k_display`` (clipped to the pool
    size) is the number drawn in the bar chart at any one time.

    Args:
        contributions: Array of shape ``(n_samples, n_components, n_features)``.
        feature_names: Human-readable name per feature. When ``None``,
            synthetic ``"Feature {i}"`` names are generated.
        top_k_global: Target size of the kept-feature pool. Clipped to
            ``n_features``.
        top_k_display: Target number of features to display. Clipped to the
            kept pool size.

    Returns:
        A :class:`TopFeatures` with ``kept_names`` and ``keep_idx`` ordered
        by descending global L2 importance, ``display_k`` clipped to the
        kept pool size, and the full L2-reduced array in ``reduced``.
    """
    n_features = contributions.shape[-1]
    names = (
        feature_names if feature_names is not None else [f"Feature {i}" for i in range(n_features)]
    )

    reduced = reduce_contributions(contributions, "l2")
    global_importance = reduced.mean(axis=0)

    n_kept = min(top_k_global, n_features)
    keep_idx = np.argsort(global_importance)[::-1][:n_kept]
    kept_names = [names[i] for i in keep_idx]
    display_k = min(top_k_display, n_kept)
    return TopFeatures(
        kept_names=kept_names,
        keep_idx=keep_idx,
        display_k=display_k,
        reduced=reduced,
    )


def _compute_bar_views(
    contributions: NDArray[np.floating],
    top: TopFeatures,
) -> BarViews:
    """Slice the three absolute bar views for the kept-feature pool."""
    return BarViews(
        l2=top.reduced[:, top.keep_idx].astype(np.float32),
        d0=contributions[:, 0, top.keep_idx].astype(np.float32),
        d1=contributions[:, 1, top.keep_idx].astype(np.float32),
    )


def _base_figure(title: str) -> figure:
    tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select"
    return figure(
        title=title or "Embedding — lasso or box-select to filter",
        width=600,
        height=_FIGURE_HEIGHT,
        tools=tools,
    )


def _make_scatter_source(
    Z: NDArray[np.floating],
    n_samples: int,
    extras: dict[str, NDArray[Any]],
) -> ColumnDataSource:
    data: dict[str, NDArray[Any]] = {
        "x": Z[:, 0].astype(np.float32),
        "y": Z[:, 1].astype(np.float32),
        "index": np.arange(n_samples),
        **extras,
    }
    return ColumnDataSource(data)  # pyright: ignore[reportArgumentType]


def _nondegenerate_range(lo: float, hi: float) -> tuple[float, float]:
    """Widen a degenerate (``hi - lo < 1e-12``) range for ``LinearColorMapper``.

    When every sample has the same color value (e.g. a feature with zero
    variance), a mapper with ``low == high`` renders all points at ``high``
    and breaks tick generation on the color bar. This widens such ranges to a
    small symmetric span around the constant value, so points map to the
    mid-color and the color bar retains a legible axis.
    """
    if hi - lo >= 1e-12:
        return lo, hi
    mid = (lo + hi) / 2
    span = max(abs(mid) * 0.05, 1e-6)
    return mid - span, mid + span


def _to_hover_uri(arr: NDArray[np.uint8]) -> str:
    """Encode an image array as a base64 PNG data URI.

    The image is resized so its longest side is
    ``_HOVER_IMAGE_LONGEST_SIDE`` pixels, preserving aspect ratio.

    Args:
        arr: Uint8 array of shape (H, W) for grayscale or (H, W, 3/4) for
            RGB(A).

    Returns:
        A ``data:image/png;base64,...`` URI suitable for use as the ``src``
        of an ``<img>`` tag inside a Bokeh ``HoverTool`` tooltip.
    """
    img = Image.fromarray(arr)
    w, h = img.size
    scale = _HOVER_IMAGE_LONGEST_SIDE / max(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_hover(
    default_body: str,
    hover_images: NDArray[np.uint8] | None,
    hover_tooltips: str | None,
    hover_data: Mapping[str, Sequence[Any]] | None,
) -> tuple[str, dict[str, NDArray[Any]]]:
    """Resolve hover customization into a tooltip template and CDS extras.

    Three modes, determined by which kwargs are set:

    - None set: returns ``(<div>{default_body}</div>, {})`` — the plot's
      built-in tooltip.
    - ``hover_images`` set: image is PNG-encoded and prepended to the default
      tooltip body.
    - ``hover_tooltips`` and/or ``hover_data`` set: user-supplied template
      fully replaces the default; user columns are merged into the scatter
      ``ColumnDataSource``.

    Args:
        default_body: HTML fragment (no outer ``<div>``) representing the
            plot's built-in tooltip body — used when no customization is
            passed and as the suffix when ``hover_images`` is passed.
        hover_images: Uint8 array of shape (n_samples, H, W) or
            (n_samples, H, W, 3/4). Mutually exclusive with the other two.
        hover_tooltips: Tooltip HTML that fully replaces the default.
        hover_data: Extra CDS columns referenced by ``@field`` in
            ``hover_tooltips``. Keys must not collide with reserved columns.

    Raises:
        ValueError: If ``hover_images`` is combined with the other two, or if
            ``hover_data`` contains a reserved key.
    """
    if hover_images is not None and (hover_tooltips is not None or hover_data is not None):
        raise ValueError("Pass either hover_images or hover_tooltips/hover_data, not both.")

    if hover_images is not None:
        uris = [_to_hover_uri(img) for img in hover_images]
        tooltip = (
            "<div>"
            "<img src='@__hover_image' style='display:block; margin-bottom:4px'/>"
            f"{default_body}"
            "</div>"
        )
        return tooltip, {"__hover_image": np.asarray(uris, dtype=object)}

    extras: dict[str, NDArray[Any]] = {}
    if hover_data is not None:
        collisions = set(hover_data) & _RESERVED_HOVER_KEYS
        if collisions:
            raise ValueError(
                f"hover_data keys collide with reserved CDS columns: {sorted(collisions)}"
            )
        extras = {k: np.asarray(v) for k, v in hover_data.items()}

    tooltip = hover_tooltips if hover_tooltips is not None else f"<div>{default_body}</div>"
    return tooltip, extras


def _build_bars(
    views: BarViews,
    top: TopFeatures,
    n_samples: int,
    scatter_source: ColumnDataSource,
) -> LayoutDOM:
    """Single bar chart with a ``L2 | normed L2 | Dim 1 | Dim 2`` view toggle.

    The chart shows the per-feature mean of contributions over the current
    scatter selection (or all samples if none selected). The ``normed L2`` view
    is computed JS-side on first selection (per-sample fractions of L2) and
    cached.
    """
    contrib_sources = [
        ColumnDataSource({f"c{k}": views.l2[:, k] for k in range(top.n_kept)}),
        ColumnDataSource({f"c{k}": views.d0[:, k] for k in range(top.n_kept)}),
        ColumnDataSource({f"c{k}": views.d1[:, k] for k in range(top.n_kept)}),
    ]

    init_mean = views.l2.mean(axis=0)
    init_top = np.argsort(init_mean)[::-1][: top.display_k][::-1]
    init_feat = [top.kept_names[i] for i in init_top]
    bar_source = ColumnDataSource(
        data=dict(feature=init_feat, mean=[float(init_mean[i]) for i in init_top])
    )

    p_bar = figure(
        width=400,
        sizing_mode="stretch_height",
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar.hbar(y="feature", right="mean", height=0.8, source=bar_source, color=BAR_COLOR_REDUCED)
    p_bar.xgrid.grid_line_color = None

    heading_div = Div(text=f"<b>Mean contribution</b> — {_VIEW_LABELS[0]}")
    view_widget = RadioButtonGroup(labels=list(_VIEW_LABELS), active=0)

    cb = CustomJS(
        args=dict(
            scatter_source=scatter_source,
            contrib_sources=contrib_sources,
            bar_source=bar_source,
            bar_range=p_bar.y_range,
            heading_div=heading_div,
            view_widget=view_widget,
            feature_names=top.kept_names,
            n_kept=top.n_kept,
            n_samples=n_samples,
            display_k=top.display_k,
            view_labels=list(_VIEW_LABELS),
            normed_l2_cache={"data": None},
        ),
        code=_LINKED_BARS_CALLBACK_JS,
    )
    scatter_source.selected.js_on_change("indices", cb)
    view_widget.js_on_change("active", cb)

    return column(
        heading_div,
        view_widget,
        p_bar,
        height=_FIGURE_HEIGHT,
        styles={"background-color": "white"},
    )


@overload
def plot_embedding_by_group(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    group_names: Sequence[Any] | NDArray,
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8],
) -> LayoutDOM: ...


@overload
def plot_embedding_by_group(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    group_names: Sequence[Any] | NDArray,
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM: ...


def plot_embedding_by_group(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    group_names: Sequence[Any] | NDArray,
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding scatter colored by group, linked to a feature-contribution bar chart.

    Renders a 2D scatter of ``Z`` colored categorically by ``group_names``, with a
    bar chart on the right showing the mean per-feature contribution. The chart
    has a single radio toggle ``L2 | normed L2 | Dim 1 | Dim 2``:

    - ``L2``: per-feature L2 norm across embedding dimensions (always non-negative).
    - ``normed L2``: each sample's L2 values divided by their sum across the kept
      pool, so each row sums to 1. Reads as "what share of this sample's overall
      contribution magnitude came from this feature."
    - ``Dim 1`` / ``Dim 2``: signed contributions to embedding dimension 0 and 1
      separately.

    Lasso- or box-selecting points in the scatter updates the bars to summarize
    only the selection; with no selection the bars summarize all samples. Group
    identity is surfaced via the hover tooltip rather than a rendered legend.

    Args:
        Z:
            Embedding coordinates of shape ``(n_samples, 2)``.
        contributions:
            Per-feature contributions to each embedding dimension, of shape
            ``(n_samples, 2, n_features)``. Typically the output of
            :meth:`~glass_box_umap.GlassBoxUMAP.compute_contributions` with
            ``reduction=None``.
        group_names:
            Group label per sample. Any sequence of length ``n_samples``; elements
            are stringified before use. Up to 10 groups use the Category10 palette;
            beyond that a tiled Category20 palette is used.
        feature_names:
            Human-readable name per feature; length must equal
            ``contributions.shape[2]``. Defaults to ``"Feature {i}"``.
        top_k_global:
            How many features to ship to the browser, ranked by global L2
            importance. Features beyond this cap never appear in the bar chart, so
            set it high enough to cover anything worth inspecting but low enough
            that the page stays responsive.
        top_k_display:
            How many features the bar chart shows at any one time. The selected
            features are reordered as the scatter selection or view toggle changes.
        title:
            Scatter-plot title.
        hover_images:
            Per-sample image to display in the tooltip, as a uint8 array of shape
            ``(n_samples, H, W)`` (grayscale) or ``(n_samples, H, W, 3 | 4)``
            (RGB / RGBA). Each image is resized to a small thumbnail and PNG-encoded
            into the tooltip above the default index/group text. Mutually exclusive
            with ``hover_tooltips`` and ``hover_data``.
        hover_tooltips:
            Bokeh tooltip HTML template that fully replaces the default. May
            reference ``@index``, ``@group``, and any keys from ``hover_data``.
        hover_data:
            Extra columns merged into the scatter ``ColumnDataSource`` for reference
            from ``hover_tooltips``. Keys must not collide with the reserved columns
            ``x``, ``y``, ``index``, ``group``, ``color_value``.

    Returns:
        A Bokeh layout — scatter on the left, linked bar chart with a view toggle
        on the right. Pass it to :func:`bokeh.io.show` or :func:`bokeh.io.save`.
    """
    _validate_shapes(Z, contributions, feature_names=feature_names, group_names=group_names)
    n_samples = Z.shape[0]

    top = _select_top_features(contributions, feature_names, top_k_global, top_k_display)
    views = _compute_bar_views(contributions, top)

    tooltip_html, hover_extras = _resolve_hover(
        default_body="index: @index &nbsp;&middot;&nbsp; group: @group",
        hover_images=hover_images,
        hover_tooltips=hover_tooltips,
        hover_data=hover_data,
    )

    scatter_source = _make_scatter_source(
        Z,
        n_samples,
        {"group": np.asarray([str(g) for g in group_names]), **hover_extras},
    )

    p_scatter = _base_figure(title)
    p_scatter.add_tools(HoverTool(tooltips=tooltip_html))

    factors = sorted({str(g) for g in group_names})
    palette = _pick_palette(len(factors))
    for factor, color in zip(factors, palette, strict=False):
        view = CDSView(filter=GroupFilter(column_name="group", group=factor))
        p_scatter.scatter(
            "x",
            "y",
            source=scatter_source,
            view=view,
            size=5,
            alpha=0.6,
            nonselection_alpha=0.1,
            color=color,
        )

    bars = _build_bars(
        views=views,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
    )

    return row(p_scatter, bars)


@overload
def plot_embedding_by_feature_gradient(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8],
) -> LayoutDOM: ...


@overload
def plot_embedding_by_feature_gradient(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM: ...


def plot_embedding_by_feature_gradient(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding scatter colored by a single feature's contribution, with picker.

    Renders a 2D scatter of ``Z`` colored by the L2-reduced contribution of one
    feature, using a Viridis gradient. An autocomplete input above the scatter
    selects which feature drives the coloring (substring match against feature
    names, case-insensitive). The bar chart on the right has the same view
    toggle described in :func:`plot_embedding_by_group`; lasso- or box-selecting
    points in the scatter updates the bars to summarize only the selection.

    Args:
        Z:
            Embedding coordinates of shape ``(n_samples, 2)``.
        contributions:
            Per-feature contributions of shape ``(n_samples, 2, n_features)``.
        feature_names:
            Human-readable name per feature; length must equal
            ``contributions.shape[2]``. Populates the autocomplete completion
            list. Defaults to ``"Feature {i}"``.
        top_k_global:
            How many features to ship to the browser, ranked by global L2
            importance. Caps the size of the autocomplete list — features
            beyond this cap cannot be selected for coloring.
        top_k_display:
            How many features the bar chart shows at any one time.
        title:
            Scatter-plot title.
        hover_images:
            See :func:`plot_embedding_by_group`. Mutually exclusive with
            ``hover_tooltips`` and ``hover_data``.
        hover_tooltips:
            Bokeh tooltip HTML template that fully replaces the default. May
            reference ``@index`` plus any keys from ``hover_data``.
        hover_data:
            Extra columns merged into the scatter ``ColumnDataSource`` for
            reference from ``hover_tooltips``. Keys must not collide with the
            reserved columns ``x``, ``y``, ``index``, ``group``, ``color_value``.

    Returns:
        A Bokeh layout — feature picker on top, scatter + linked bar chart with a
        view toggle below. Pass it to :func:`bokeh.io.show` or :func:`bokeh.io.save`.
    """
    _validate_shapes(Z, contributions, feature_names=feature_names)
    n_samples = Z.shape[0]

    top = _select_top_features(contributions, feature_names, top_k_global, top_k_display)
    views = _compute_bar_views(contributions, top)

    reduced_kept = views.l2
    initial_color = reduced_kept[:, 0]

    tooltip_html, hover_extras = _resolve_hover(
        default_body="index: @index",
        hover_images=hover_images,
        hover_tooltips=hover_tooltips,
        hover_data=hover_data,
    )

    scatter_source = _make_scatter_source(
        Z, n_samples, {"color_value": initial_color.copy(), **hover_extras}
    )

    reduced_source = ColumnDataSource({f"f{k}": reduced_kept[:, k] for k in range(top.n_kept)})

    p_scatter = _base_figure(title)
    p_scatter.add_tools(HoverTool(tooltips=tooltip_html))

    init_lo, init_hi = _nondegenerate_range(float(initial_color.min()), float(initial_color.max()))
    gradient_mapper = LinearColorMapper(palette=Viridis256, low=init_lo, high=init_hi)
    p_scatter.scatter(
        "x",
        "y",
        source=scatter_source,
        size=5,
        alpha=0.6,
        nonselection_alpha=0.1,
        color={"field": "color_value", "transform": gradient_mapper},
    )
    color_bar = ColorBar(color_mapper=gradient_mapper, location=(0, 0))
    p_scatter.add_layout(color_bar, "right")

    feature_picker = AutocompleteInput(
        title="feature",
        completions=top.kept_names,
        value=top.kept_names[0],
        placeholder="start typing…",
        search_strategy="includes",
        case_sensitive=False,
        min_characters=0,
        max_completions=15,
        width=260,
    )
    feature_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                reduced_source=reduced_source,
                mapper=gradient_mapper,
                feature_names=top.kept_names,
            ),
            code="""
            const idx = feature_names.indexOf(cb_obj.value);
            if (idx < 0) { return; }
            const col = reduced_source.data["f" + idx];
            const copy = new Float64Array(col.length);
            let lo = Infinity, hi = -Infinity;
            for (let i = 0; i < col.length; i++) {
                const v = col[i];
                copy[i] = v;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (hi - lo < 1e-12) {
                const mid = (lo + hi) / 2;
                const span = Math.max(Math.abs(mid) * 0.05, 1e-6);
                lo = mid - span;
                hi = mid + span;
            }
            scatter_source.data["color_value"] = copy;
            scatter_source.change.emit();
            mapper.low = lo;
            mapper.high = hi;
            """,
        ),
    )

    bars = _build_bars(
        views=views,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
    )

    return column(feature_picker, row(p_scatter, bars))


@overload
def plot_embedding_by_top_feature(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_n: int = 20,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8],
) -> LayoutDOM: ...


@overload
def plot_embedding_by_top_feature(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_n: int = 20,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM: ...


def plot_embedding_by_top_feature(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    top_n: int = 20,
    top_k_global: int = 200,
    top_k_display: int = 20,
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding scatter colored by each sample's most-contributing feature.

    For every sample, the feature with the largest L2-reduced contribution is
    chosen as that sample's group. Samples whose top feature is not among the
    ``top_n`` most-frequently-occurring top features across the dataset are
    dropped — this keeps the color count manageable when many features each
    "win" for only a handful of samples. The remaining samples are rendered
    via :func:`plot_embedding_by_group`, so all interactions and arguments
    there apply here.

    Args:
        Z:
            Embedding coordinates of shape ``(n_samples, 2)``.
        contributions:
            Per-feature contributions of shape ``(n_samples, 2, n_features)``.
        feature_names:
            Human-readable name per feature; length must equal
            ``contributions.shape[2]``. Used both to identify each sample's
            top feature and as the group label. Defaults to ``"Feature {i}"``.
        top_n:
            Cap on the number of distinct top-feature groups in the plot.
            Samples whose top-contributing feature falls outside the most-common
            ``top_n`` are filtered out. Reducing ``top_n`` hides rare top
            features and shrinks the dataset rendered.
        top_k_global:
            Forwarded to :func:`plot_embedding_by_group`.
        top_k_display:
            Forwarded to :func:`plot_embedding_by_group`.
        title:
            Scatter-plot title; defaults to ``"Top-feature groups"``.
        hover_images:
            See :func:`plot_embedding_by_group`. Filtered alongside ``Z`` so
            the hover image matches each kept point.
        hover_tooltips:
            See :func:`plot_embedding_by_group`.
        hover_data:
            See :func:`plot_embedding_by_group`. Each column is filtered
            alongside ``Z``.

    Returns:
        The Bokeh layout produced by :func:`plot_embedding_by_group` — scatter
        on the left, linked bar chart with a view toggle on the right. Pass it to
        :func:`bokeh.io.show` or :func:`bokeh.io.save`.
    """
    _validate_shapes(Z, contributions, feature_names=feature_names)

    n_features = contributions.shape[-1]
    names_arr = (
        np.array(feature_names)
        if feature_names is not None
        else np.array([f"Feature {i}" for i in range(n_features)])
    )
    group_ids, group_name_list, mask = groups_from_top_features(
        contributions, feature_names=names_arr, top_n=top_n
    )
    kept_group_names = [group_name_list[gid] for gid in group_ids]

    shared_kwargs: dict[str, Any] = dict(
        Z=Z[mask],
        contributions=contributions[mask],
        group_names=kept_group_names,
        feature_names=list(names_arr),
        top_k_global=top_k_global,
        top_k_display=top_k_display,
        title=title or "Top-feature groups",
    )

    if hover_images is not None:
        return plot_embedding_by_group(**shared_kwargs, hover_images=hover_images[mask])

    masked_data = (
        {k: np.asarray(v)[mask].tolist() for k, v in hover_data.items()}
        if hover_data is not None
        else None
    )
    return plot_embedding_by_group(
        **shared_kwargs,
        hover_tooltips=hover_tooltips,
        hover_data=masked_data,
    )


