import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal, overload

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    CDSView,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    GroupFilter,
    HoverTool,
    LinearColorMapper,
    Range,
)
from bokeh.models.layouts import LayoutDOM
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.plotting import figure
from numpy.typing import NDArray
from PIL import Image

from ..jacobian import groups_from_top_features, reduce_contributions

BAR_COLOR_REDUCED = "#756bb1"
BAR_COLOR_DIM_X = "#2b8cbe"
BAR_COLOR_DIM_Y = "#e34a33"

_HOVER_IMAGE_LONGEST_SIDE = 64
_RESERVED_HOVER_KEYS = frozenset({"x", "y", "index", "group", "color_value", "__hover_image"})

_LINKED_BARS_CALLBACK_JS = """
const sel = scatter_source.selected.indices;
const indices = sel.length
    ? sel
    : Array.from({length: n_samples}, (_, i) => i);
const n = indices.length;

const n_channels = contrib_sources.length;
const means = [];
for (let c = 0; c < n_channels; c++) {
    means.push(new Float64Array(n_kept));
}

for (let k = 0; k < n_kept; k++) {
    const key = "c" + k;
    for (let c = 0; c < n_channels; c++) {
        const col = contrib_sources[c].data[key];
        let s = 0.0;
        for (let j = 0; j < n; j++) {
            s += col[indices[j]];
        }
        means[c][k] = s / n;
    }
}

const scored = new Array(n_kept);
for (let k = 0; k < n_kept; k++) {
    let score = 0.0;
    for (let c = 0; c < n_channels; c++) {
        const m = Math.abs(means[c][k]);
        if (m > score) score = m;
    }
    scored[k] = { idx: k, score: score };
}
scored.sort((a, b) => b.score - a.score);
const top = scored.slice(0, display_k).reverse();
const feat = top.map(t => feature_names[t.idx]);

for (let c = 0; c < n_channels; c++) {
    const vals = top.map(t => means[c][t.idx]);
    bar_sources[c].data = { feature: feat, mean: vals };
    bar_ranges[c].factors = feat;
}
"""


@dataclass(frozen=True)
class TopFeatures:
    """Result of ranking features by global importance and selecting the top slice.

    Attributes:
        kept_names: Feature names for the kept pool, ordered by descending
            global importance.
        keep_idx: Indices into the original feature axis, in the same order
            as ``kept_names``. Length equals ``n_kept``.
        display_k: Number of features to draw in the bar chart at any one
            time. Always ``<= n_kept``.
        reduced: The ``(n_samples, n_features)`` array produced by the
            chosen reduction (e.g. L2 across embedding dimensions), or
            ``None`` if no reduction was applied. This is the full reduced
            array — not sliced by ``keep_idx``.
        n_kept: (property) Size of the kept pool, i.e. ``len(kept_names)``.
    """

    kept_names: list[str]
    keep_idx: NDArray[np.integer]
    display_k: int
    reduced: NDArray[np.floating] | None

    @property
    def n_kept(self) -> int:
        return len(self.kept_names)


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
    reduction: Literal["l2"] | None,
) -> TopFeatures:
    """Rank features by global importance and select the top slice for plotting.

    Ranking is done on a per-feature scalar derived from ``contributions``:

    - ``reduction=None``: the scalar is ``mean(|contributions|)`` taken over
      both the sample axis and the embedding-dimension axis. The reduced
      array is not computed and ``None`` is returned in its place.
    - ``reduction="l2"``: contributions are first collapsed across embedding
      dimensions via :func:`reduce_contributions`, producing a
      ``(n_samples, n_features)`` array. The scalar is then the per-feature
      mean across samples.

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
        reduction: Controls both how global importance is computed and
            whether the reduced array is materialized. See above.

    Returns:
        A :class:`TopFeatures` with ``kept_names`` and ``keep_idx`` ordered
        by descending global importance, ``display_k`` clipped to the kept
        pool size, and ``reduced`` populated iff ``reduction`` is set.
    """
    n_features = contributions.shape[-1]
    names = (
        feature_names if feature_names is not None else [f"Feature {i}" for i in range(n_features)]
    )

    if reduction is None:
        global_importance = np.abs(contributions).mean(axis=(0, 1))
        reduced: NDArray[np.floating] | None = None
    else:
        reduced = reduce_contributions(contributions, reduction)
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


def _base_figure(title: str) -> figure:
    tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select"
    return figure(
        title=title or "Embedding — lasso or box-select to filter",
        width=600,
        height=600,
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
    contributions: NDArray[np.floating],
    top: TopFeatures,
    reduction: Literal["l2"] | None,
    n_samples: int,
    scatter_source: ColumnDataSource,
) -> LayoutDOM:
    if reduction is None:
        return _build_two_dim_bars(
            contributions=contributions,
            top=top,
            n_samples=n_samples,
            scatter_source=scatter_source,
        )
    assert top.reduced is not None
    return _build_reduced_bars(
        reduced=top.reduced,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
        reduction=reduction,
    )


def _linked_bars_callback(
    scatter_source: ColumnDataSource,
    contrib_sources: list[ColumnDataSource],
    bar_sources: list[ColumnDataSource],
    bar_ranges: list[Range],
    top: TopFeatures,
    n_samples: int,
) -> CustomJS:
    return CustomJS(
        args=dict(
            scatter_source=scatter_source,
            contrib_sources=contrib_sources,
            bar_sources=bar_sources,
            bar_ranges=bar_ranges,
            feature_names=top.kept_names,
            n_kept=top.n_kept,
            n_samples=n_samples,
            display_k=top.display_k,
        ),
        code=_LINKED_BARS_CALLBACK_JS,
    )


@overload
def plot_embedding_by_group(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    group_names: Sequence[Any] | NDArray,
    feature_names: list[str] | None = None,
    top_k_global: int = 200,
    top_k_display: int = 20,
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding colored by group with linked bar charts of feature contributions.

    Points are colored categorically by ``group_names``. Group identity is
    surfaced via the hover tooltip rather than a legend.

    Args:
        Z: Embedding coordinates with shape (n_samples, 2).
        contributions: Array of shape (n_samples, 2, n_features).
        group_names: Label per point. Any sequence of length ``n_samples`` —
            elements are stringified and used as legend entries.
        feature_names: Optional human-readable names for each feature.
        top_k_global: Number of globally most-important features shipped to
            the browser.
        top_k_display: Number of features drawn in the bar chart(s) at a time.
        reduction: Per-sample reduction across embedding dimensions for the
            bar chart. ``"l2"`` renders one bar chart; ``None`` renders two
            (one per dim).
        title: Title of the scatter plot.
        hover_images: Per-sample uint8 image array of shape (n_samples, H, W)
            or (n_samples, H, W, 3/4). When set, each tooltip shows the
            sample's image above the default index/group text. Mutually
            exclusive with ``hover_tooltips`` / ``hover_data``.
        hover_tooltips: Bokeh-style tooltip HTML template that fully replaces
            the default tooltip. May reference built-in columns via
            ``@index`` and ``@group``, plus any keys from ``hover_data``.
        hover_data: Extra columns merged into the scatter ``ColumnDataSource``
            for reference from ``hover_tooltips``. Keys must not collide with
            reserved columns (``x``, ``y``, ``index``, ``group``,
            ``color_value``).
    """
    _validate_shapes(Z, contributions, feature_names=feature_names, group_names=group_names)
    n_samples = Z.shape[0]

    top = _select_top_features(contributions, feature_names, top_k_global, top_k_display, reduction)

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
        contributions=contributions,
        top=top,
        reduction=reduction,
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
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding colored by per-feature contribution with a fuzzy-search picker.

    A single gradient (Viridis) colors points by the L2-reduced contribution
    of the selected feature. An autocomplete input above the scatter lets the
    user pick a feature (substring match, case-insensitive). Bar charts to the
    right work the same as in :func:`plot_embedding_by_group`.

    Note: gradient coloring always uses the per-sample L2 scalar, independent
    of ``reduction``. The ``reduction`` argument only controls the bar charts.

    Args:
        Z: Embedding coordinates with shape (n_samples, 2).
        contributions: Array of shape (n_samples, 2, n_features).
        feature_names: Optional human-readable names for each feature.
        top_k_global: Number of globally most-important features shipped to
            the browser. Caps the size of the autocomplete list.
        top_k_display: Number of features drawn in the bar chart(s) at a time.
        reduction: Per-sample reduction across embedding dimensions for the
            bar chart. See :func:`plot_embedding_by_group`.
        title: Title of the scatter plot.
        hover_images: Per-sample uint8 image array of shape (n_samples, H, W)
            or (n_samples, H, W, 3/4). When set, each tooltip shows the
            sample's image above the default index text. Mutually exclusive
            with ``hover_tooltips`` / ``hover_data``.
        hover_tooltips: Bokeh-style tooltip HTML template that fully replaces
            the default tooltip. May reference ``@index`` plus any keys from
            ``hover_data``.
        hover_data: Extra columns merged into the scatter ``ColumnDataSource``
            for reference from ``hover_tooltips``. Keys must not collide with
            reserved columns (``x``, ``y``, ``index``, ``group``,
            ``color_value``).
    """
    _validate_shapes(Z, contributions, feature_names=feature_names)
    n_samples = Z.shape[0]

    top = _select_top_features(contributions, feature_names, top_k_global, top_k_display, reduction)

    reduced_l2 = (
        top.reduced if top.reduced is not None else reduce_contributions(contributions, "l2")
    )
    reduced_kept = reduced_l2[:, top.keep_idx].astype(np.float32)
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
        contributions=contributions,
        top=top,
        reduction=reduction,
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
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
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
    reduction: Literal["l2"] | None = "l2",
    title: str = "",
    *,
    hover_images: NDArray[np.uint8] | None = None,
    hover_tooltips: str | None = None,
    hover_data: Mapping[str, Sequence[Any]] | None = None,
) -> LayoutDOM:
    """Embedding where each point is colored by its top-contributing feature.

    Points whose top-contributing feature is not among the ``top_n`` most
    common top features are filtered out. Delegates rendering to
    :func:`plot_embedding_by_group`.

    Args:
        Z: Embedding coordinates with shape (n_samples, 2).
        contributions: Array of shape (n_samples, 2, n_features).
        feature_names: Optional human-readable names for each feature.
        top_n: Number of most-common top features to keep.
        top_k_global: Passed through to :func:`plot_embedding_by_group`.
        top_k_display: Passed through to :func:`plot_embedding_by_group`.
        reduction: Must not be ``None``. Top-feature ranking is undefined for
            2D contributions.
        title: Title of the scatter plot.
        hover_images: See :func:`plot_embedding_by_group`. Filtered alongside
            ``Z`` to match the kept points.
        hover_tooltips: See :func:`plot_embedding_by_group`.
        hover_data: See :func:`plot_embedding_by_group`. Each column is
            filtered alongside ``Z`` to match the kept points.
    """
    _validate_shapes(Z, contributions, feature_names=feature_names)

    if reduction is None:
        raise ValueError(
            "plot_embedding_by_top_feature requires reduction to be set "
            "(top-feature ranking is undefined for 2D contributions). "
            "Pass reduction='l2'."
        )

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
        reduction=reduction,
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


def _build_reduced_bars(
    reduced: NDArray[np.floating],
    top: TopFeatures,
    n_samples: int,
    scatter_source: ColumnDataSource,
    reduction: str,
) -> LayoutDOM:
    totals = reduced.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    normalized = reduced / totals
    kept = normalized[:, top.keep_idx].astype(np.float32)
    contrib_source = ColumnDataSource({f"c{k}": kept[:, k] for k in range(top.n_kept)})

    init_mean = kept.mean(axis=0)
    init_top = np.argsort(init_mean)[::-1][: top.display_k][::-1]
    init_feat = [top.kept_names[i] for i in init_top]
    bar_source = ColumnDataSource(
        data=dict(feature=init_feat, mean=[float(init_mean[i]) for i in init_top])
    )

    p_bar = figure(
        title=f"Mean fractional contribution ({reduction})",
        width=400,
        height=600,
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar.hbar(y="feature", right="mean", height=0.8, source=bar_source, color=BAR_COLOR_REDUCED)
    p_bar.xgrid.grid_line_color = None

    cb = _linked_bars_callback(
        scatter_source=scatter_source,
        contrib_sources=[contrib_source],
        bar_sources=[bar_source],
        bar_ranges=[p_bar.y_range],
        top=top,
        n_samples=n_samples,
    )
    scatter_source.selected.js_on_change("indices", cb)

    return p_bar


def _build_two_dim_bars(
    contributions: NDArray[np.floating],
    top: TopFeatures,
    n_samples: int,
    scatter_source: ColumnDataSource,
) -> LayoutDOM:
    kept = contributions[:, :, top.keep_idx].astype(np.float32)
    contrib_x_source = ColumnDataSource({f"c{k}": kept[:, 0, k] for k in range(top.n_kept)})
    contrib_y_source = ColumnDataSource({f"c{k}": kept[:, 1, k] for k in range(top.n_kept)})

    init_mean_x = kept[:, 0, :].mean(axis=0)
    init_mean_y = kept[:, 1, :].mean(axis=0)
    init_score = np.maximum(np.abs(init_mean_x), np.abs(init_mean_y))
    init_top = np.argsort(init_score)[::-1][: top.display_k][::-1]
    init_feat = [top.kept_names[i] for i in init_top]
    bar_x_source = ColumnDataSource(
        data=dict(feature=init_feat, mean=[float(init_mean_x[i]) for i in init_top])
    )
    bar_y_source = ColumnDataSource(
        data=dict(feature=init_feat, mean=[float(init_mean_y[i]) for i in init_top])
    )

    p_bar_x = figure(
        title="Mean contribution — dim 1",
        width=400,
        height=300,
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar_x.hbar(y="feature", right="mean", height=0.8, source=bar_x_source, color=BAR_COLOR_DIM_X)
    p_bar_x.xgrid.grid_line_color = None

    p_bar_y = figure(
        title="Mean contribution — dim 2",
        width=400,
        height=300,
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar_y.hbar(y="feature", right="mean", height=0.8, source=bar_y_source, color=BAR_COLOR_DIM_Y)
    p_bar_y.xgrid.grid_line_color = None

    cb = _linked_bars_callback(
        scatter_source=scatter_source,
        contrib_sources=[contrib_x_source, contrib_y_source],
        bar_sources=[bar_x_source, bar_y_source],
        bar_ranges=[p_bar_x.y_range, p_bar_y.y_range],
        top=top,
        n_samples=n_samples,
    )
    scatter_source.selected.js_on_change("indices", cb)

    return column(p_bar_x, p_bar_y)
