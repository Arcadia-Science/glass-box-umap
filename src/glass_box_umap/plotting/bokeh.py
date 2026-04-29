import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, overload

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    CategoricalColorMapper,
    CDSView,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    CustomJSFilter,
    Div,
    FactorRange,
    GroupFilter,
    HoverTool,
    InlineStyleSheet,
    LinearColorMapper,
    RadioButtonGroup,
    Slider,
)
from bokeh.models.layouts import LayoutDOM
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from numpy.typing import NDArray
from PIL import Image

from ..jacobian import reduce_contributions

BAR_COLOR_REDUCED = "#756bb1"

_TOP_K_DISPLAY = 36
_HOVER_IMAGE_LONGEST_SIDE = 64
_RESERVED_HOVER_KEYS = frozenset(
    {
        "x",
        "y",
        "index",
        "group",
        "color_value",
        "top_feature_group",
        "top_feature_name",
        "top_data_value",
        "picker_data_value",
        "sample_rank",
        "__hover_image",
    }
)

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

heading_div.text = `<b>Mean contribution — ${view_labels[view]}</b>`;
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
class HoverTooltips:
    """Per-color-mode tooltip HTML used by the scatter's HoverTools.

    Each glyph set in the scatter is paired with one of these templates so the
    tooltip surfaces info relevant to the current "Color by" mode. When the
    user passes ``hover_tooltips``, all three fields hold the same override
    string.

    Attributes:
        group: Tooltip used by the per-group glyphs ("Color by" → Group).
        feature: Tooltip used by the gradient glyph ("Color by" → Feature).
        top: Tooltip used by the top-feature glyphs ("Color by" → Top feature).
    """

    group: str
    feature: str
    top: str


@dataclass(frozen=True)
class BarViews:
    """Three pre-computed ``(n_samples, n_kept)`` views of the kept-feature pool.

    Sign convention is per-attribute, not uniform: ``l2`` is non-negative
    (it's an L2 norm); ``d0`` and ``d1`` are signed (raw per-dimension
    contributions). The "normed L2" view shown in the bar chart is derived
    JS-side from ``l2`` as ``value / max(Σ_k value, ε)`` per sample (so each
    sample's row sums to 1 across the kept pool), cached after first
    computation. Note that the scatter's "Top feature" coloring is also
    decided from ``l2`` (argmax over the kept pool), so a sample's "top
    feature" reflects magnitude regardless of which view the user toggles
    in the bar chart.

    Attributes:
        l2: L2-reduced contributions, sliced to ``top.keep_idx``. Always non-negative.
        d0: Signed contributions to embedding dimension 0, sliced to ``top.keep_idx``.
        d1: Signed contributions to embedding dimension 1, sliced to ``top.keep_idx``.
    """

    l2: NDArray[np.floating]
    d0: NDArray[np.floating]
    d1: NDArray[np.floating]


_GLASBEY_CATEGORY10: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#3a0183", "#004301",
    "#0fffa9", "#5e0040", "#bcbcff", "#d8afa2", "#b80080", "#004e53",
    "#6b6500", "#7d0200", "#6126ff", "#ffff9a", "#574964", "#8cb894",
    "#94fcff", "#028268", "#91ff00", "#8300a0", "#ad8944", "#5b3400",
    "#ffc0f3", "#ff6f76",
]


def _pick_palette(n: int) -> list[str]:
    base = _GLASBEY_CATEGORY10
    tiles = (n + len(base) - 1) // len(base)
    return (base * tiles)[:n]


def _validate_shapes(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    group_names: Sequence[Any] | NDArray | None = None,
    feature_values: NDArray[np.floating] | None = None,
) -> None:
    """Validate shape/length invariants shared by the public plot functions.

    Raises:
        ValueError: If ``Z`` is not ``(n_samples, 2)``, ``contributions`` is
            not ``(n_samples, 2, n_features)`` with ``n_features >= 1``, or
            ``feature_names`` / ``group_names`` / ``feature_values`` (when
            provided) don't match the corresponding axes.
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

    if feature_values is not None and feature_values.shape != (n_samples, n_features):
        raise ValueError(
            f"feature_values has shape {feature_values.shape}, but expected "
            f"({n_samples}, {n_features})."
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


def _precompute_top_features(
    kept_l2: NDArray[np.floating],
    kept_names: list[str],
) -> tuple[list[str], NDArray[np.integer], NDArray[np.integer]]:
    """Per-sample top kept feature, ranked by frequency.

    For each sample, the kept feature with the largest L2-reduced contribution
    is its "top feature". Distinct top features are then ranked by how often
    they win across the dataset (most common → rank 0). Restricting argmax to
    the kept pool guarantees every legend label has a corresponding bar in the
    bar chart.

    Args:
        kept_l2: L2-reduced contributions sliced to the kept pool, shape
            ``(n_samples, n_kept)``.
        kept_names: Feature names matching ``kept_l2``'s column axis.

    Returns:
        A ``(top_feature_names_by_rank, sample_rank, top_kept_idx)`` tuple:

        - ``top_feature_names_by_rank``: distinct kept-feature names ordered
          by descending frequency of being a sample's top feature.
        - ``sample_rank``: per-sample integer rank into
          ``top_feature_names_by_rank`` (always ``< len(top_feature_names_by_rank)``).
        - ``top_kept_idx``: per-sample column index into ``kept_l2`` of the
          sample's top feature (i.e. ``kept_l2.argmax(axis=1)``).
    """
    top_kept_idx = kept_l2.argmax(axis=1)
    unique, counts = np.unique(top_kept_idx, return_counts=True)
    rank_order = unique[np.argsort(counts)[::-1]]
    top_feature_names_by_rank = [kept_names[i] for i in rank_order]
    rank_of: dict[int, int] = {int(idx): r for r, idx in enumerate(rank_order)}
    sample_rank = np.array([rank_of[int(i)] for i in top_kept_idx], dtype=np.int64)
    return top_feature_names_by_rank, sample_rank, top_kept_idx


def _base_figure() -> figure:
    tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select"
    return figure(
        title="Embedding — lasso or box-select to filter",
        sizing_mode="stretch_both",
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
    default_bodies: HoverTooltips,
    hover_images: NDArray[np.uint8] | None,
    hover_tooltips: str | None,
    hover_data: Mapping[str, Sequence[Any]] | None,
) -> tuple[HoverTooltips, dict[str, NDArray[Any]]]:
    """Resolve hover customization into per-mode tooltip templates and CDS extras.

    Three modes, determined by which kwargs are set:

    - None set: each ``default_bodies`` field is wrapped in ``<div>...</div>``.
    - ``hover_images`` set: each default body is wrapped in a ``<div>`` with
      a PNG-encoded image prefix.
    - ``hover_tooltips`` and/or ``hover_data`` set: user-supplied template
      fully replaces the defaults (same template for all three modes); user
      columns are merged into the scatter ``ColumnDataSource``.

    Args:
        default_bodies: Per-mode HTML body fragments (no outer ``<div>``)
            representing the plot's built-in tooltip bodies — used when no
            override is passed and as the suffix when ``hover_images`` is set.
        hover_images: Uint8 array of shape (n_samples, H, W) or
            (n_samples, H, W, 3/4). Mutually exclusive with the other two.
        hover_tooltips: Tooltip HTML that fully replaces the defaults across
            all modes.
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
        img_prefix = "<img src='@__hover_image' style='display:block; margin-bottom:4px'/>"
        tooltips = HoverTooltips(
            group=f"<div>{img_prefix}{default_bodies.group}</div>",
            feature=f"<div>{img_prefix}{default_bodies.feature}</div>",
            top=f"<div>{img_prefix}{default_bodies.top}</div>",
        )
        return tooltips, {"__hover_image": np.asarray(uris, dtype=object)}

    extras: dict[str, NDArray[Any]] = {}
    if hover_data is not None:
        collisions = set(hover_data) & _RESERVED_HOVER_KEYS
        if collisions:
            raise ValueError(
                f"hover_data keys collide with reserved CDS columns: {sorted(collisions)}"
            )
        extras = {k: np.asarray(v) for k, v in hover_data.items()}

    if hover_tooltips is not None:
        tooltips = HoverTooltips(group=hover_tooltips, feature=hover_tooltips, top=hover_tooltips)
    else:
        tooltips = HoverTooltips(
            group=f"<div>{default_bodies.group}</div>",
            feature=f"<div>{default_bodies.feature}</div>",
            top=f"<div>{default_bodies.top}</div>",
        )
    return tooltips, extras


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
        sizing_mode="stretch_both",
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar.hbar(y="feature", right="mean", height=0.8, source=bar_source, color=BAR_COLOR_REDUCED)
    p_bar.xgrid.grid_line_color = None

    heading_div = Div(
        text=f"<b>Mean contribution — {_VIEW_LABELS[0]}</b>",
        styles={"color": "#444444", "font-size": "13px"},
    )
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
        view_widget,
        heading_div,
        p_bar,
        sizing_mode="stretch_both",
        styles={
            "background-color": "white",
            "flex": "0 0 40%",
            "min-width": "0",
        },
    )


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
            reference from ``hover_tooltips``. Keys must not collide with the
            reserved columns ``x``, ``y``, ``index``, ``group``,
            ``color_value``, ``top_feature_group``, ``top_feature_name``,
            ``top_data_value``, ``picker_data_value``, ``sample_rank``.

    Returns:
        A Bokeh layout — color-by controls + scatter on the left, linked bar
        chart with view toggle on the right. Pass it to :func:`bokeh.io.show`
        or :func:`bokeh.io.save`.
    """
    _validate_shapes(
        Z,
        contributions,
        feature_names=feature_names,
        group_names=group_names,
        feature_values=feature_values,
    )
    n_samples = Z.shape[0]

    top = _select_top_features(contributions, feature_names, top_k_global, _TOP_K_DISPLAY)
    views = _compute_bar_views(contributions, top)
    top_feature_names_by_rank, sample_rank, top_kept_idx = _precompute_top_features(
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

    if has_values:
        feature_values_kept = feature_values[:, top.keep_idx].astype(np.float32)
        top_data_value = feature_values_kept[np.arange(n_samples), top_kept_idx]
        picker_data_value = feature_values_kept[:, 0].copy()

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
    tooltips, hover_extras = _resolve_hover(
        default_bodies=default_bodies,
        hover_images=hover_images,
        hover_tooltips=hover_tooltips,
        hover_data=hover_data,
    )

    extras: dict[str, NDArray[Any]] = {
        "color_value": initial_gradient,
        "top_feature_group": initial_top_group,
        "top_feature_name": top_feature_name,
        "sample_rank": sample_rank,
        **hover_extras,
    }
    if has_values:
        extras["top_data_value"] = top_data_value
        extras["picker_data_value"] = picker_data_value
    if has_groups:
        extras["group"] = np.asarray([str(g) for g in group_names])

    scatter_source = _make_scatter_source(Z, n_samples, extras)

    p_scatter = _base_figure()

    reduced_kept_source = ColumnDataSource(
        {f"f{k}": views.l2[:, k] for k in range(top.n_kept)}
    )
    feature_values_kept_source: ColumnDataSource | None = None
    if has_values:
        feature_values_kept_source = ColumnDataSource(
            {f"f{k}": feature_values_kept[:, k] for k in range(top.n_kept)}
        )

    other_view = CDSView(
        filter=GroupFilter(column_name="top_feature_group", group="(other)")
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

    named_filter = CustomJSFilter(
        code="""
        const tfg = source.data["top_feature_group"];
        const out = new Array(tfg.length);
        for (let i = 0; i < tfg.length; i++) out[i] = tfg[i] !== "(other)";
        return out;
        """,
    )
    named_view = CDSView(filter=named_filter)
    palette = _pick_palette(max(n_distinct, 1))
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

    group_glyphs: list[Any] = []
    if has_groups:
        factors = sorted({str(g) for g in group_names})
        group_palette = _pick_palette(len(factors))
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

    init_lo, init_hi = _nondegenerate_range(
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
        color_mapper=gradient_mapper, location=(0, 0), visible=(initial_mode == "Feature")
    )
    p_scatter.add_layout(color_bar, "right")

    p_scatter.add_tools(HoverTool(tooltips=tooltips.feature, renderers=[gradient_glyph]))
    p_scatter.add_tools(
        HoverTool(tooltips=tooltips.top, renderers=[top_named_glyph, top_other_glyph])
    )
    if has_groups:
        p_scatter.add_tools(HoverTool(tooltips=tooltips.group, renderers=group_glyphs))

    color_by_widget = RadioButtonGroup(labels=color_modes, active=0)
    color_by_prefix = Div(
        text="<b>Color by:</b>",
        styles={"color": "#444444", "font-size": "13px", "padding-top": "8px"},
    )
    feature_picker = AutocompleteInput(
        title="Search for feature",
        completions=top.kept_names,
        value=top.kept_names[0],
        placeholder="start typing…",
        search_strategy="includes",
        case_sensitive=False,
        min_characters=0,
        max_completions=15,
        width=260,
        visible=(initial_mode == "Feature"),
        styles={"color": "#444444"},
        stylesheets=[InlineStyleSheet(css=".bk-input { color: #444444; }")],
    )
    top_n_slider = Slider(
        start=1,
        end=max(n_distinct, 1),
        value=max(initial_t, 1),
        step=1,
        title="Top features",
        width=260,
        visible=(initial_mode == "Top feature"),
        styles={"color": "#444444"},
    )

    color_by_widget.js_on_change(
        "active",
        CustomJS(
            args=dict(
                color_modes=color_modes,
                group_glyphs=group_glyphs,
                top_other_glyph=top_other_glyph,
                top_named_glyph=top_named_glyph,
                gradient_glyph=gradient_glyph,
                color_bar=color_bar,
                feature_picker=feature_picker,
                top_n_slider=top_n_slider,
            ),
            code="""
            const mode = color_modes[cb_obj.active];
            const is_group = (mode === "Group");
            const is_feature = (mode === "Feature");
            const is_top = (mode === "Top feature");
            for (const g of group_glyphs) g.visible = is_group;
            top_other_glyph.visible = is_top;
            top_named_glyph.visible = is_top;
            gradient_glyph.visible = is_feature;
            color_bar.visible = is_feature;
            feature_picker.visible = is_feature;
            top_n_slider.visible = is_top;
            """,
        ),
    )

    feature_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                reduced_source=reduced_kept_source,
                values_source=feature_values_kept_source,
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
            if (values_source !== null) {
                const vcol = values_source.data["f" + idx];
                const vcopy = new Float64Array(vcol.length);
                for (let i = 0; i < vcol.length; i++) vcopy[i] = vcol[i];
                scatter_source.data["picker_data_value"] = vcopy;
            }
            scatter_source.change.emit();
            mapper.low = lo;
            mapper.high = hi;
            """,
        ),
    )

    top_n_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                names_by_rank=top_feature_names_by_rank,
            ),
            code="""
            const t = cb_obj.value;
            const tfg = scatter_source.data["top_feature_group"];
            const ranks = scatter_source.data["sample_rank"];
            const n = ranks.length;
            for (let i = 0; i < n; i++) {
                tfg[i] = ranks[i] < t ? names_by_rank[ranks[i]] : "(other)";
            }
            scatter_source.change.emit();
            """,
        ),
    )

    bars = _build_bars(
        views=views,
        top=top,
        n_samples=n_samples,
        scatter_source=scatter_source,
    )

    return row(
        column(
            row(color_by_prefix, color_by_widget),
            feature_picker,
            top_n_slider,
            p_scatter,
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
