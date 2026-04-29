from dataclasses import dataclass

import numpy as np
from bokeh.models import (
    AutocompleteInput,
    ColumnDataSource,
    CustomJS,
    Div,
    InlineStyleSheet,
    RadioButtonGroup,
    Slider,
)
from numpy.typing import NDArray

from ._data import BarViews, TopFeatures
from ._js import COLOR_BY_MODE, FEATURE_PICKER, TOP_N_SLIDER
from ._scatter import ScatterArtifacts


@dataclass(frozen=True)
class ControlsArtifacts:
    """Widgets produced by :func:`build_controls` for the orchestrator to lay out.

    Attributes:
        color_by_prefix: The static "Color by:" label rendered next to the
            radio button.
        color_by_widget: Radio button group toggling between the
            ``Group``/``Feature``/``Top feature`` modes.
        feature_picker: Autocomplete input that selects which feature drives
            the gradient glyph (visible only in ``Feature`` mode).
        top_n_slider: Slider controlling how many of the most-frequent top
            features are colorized (visible only in ``Top feature`` mode).
    """

    color_by_prefix: Div
    color_by_widget: RadioButtonGroup
    feature_picker: AutocompleteInput
    top_n_slider: Slider


def build_controls(
    color_modes: list[str],
    initial_mode: str,
    initial_t: int,
    n_distinct: int,
    top: TopFeatures,
    views: BarViews,
    feature_values_kept: NDArray[np.floating] | None,
    top_feature_names_by_rank: list[str],
    scatter_source: ColumnDataSource,
    scatter: ScatterArtifacts,
) -> ControlsArtifacts:
    """Build the color-by widgets and wire their three CustomJS callbacks.

    The mode toggle reaches into ``scatter`` to flip glyph/colorbar visibility
    when the user picks a mode. The feature picker recomputes the gradient
    column on the scatter source and adjusts ``scatter.gradient_mapper``'s
    range. The top-N slider relabels samples whose top-feature rank is
    outside the slider value into the ``"(other)"`` bucket.
    """
    color_by_prefix = Div(
        text="<b>Color by:</b>",
        styles={"color": "#444444", "font-size": "13px", "padding-top": "8px"},
    )
    color_by_widget = RadioButtonGroup(labels=color_modes, active=0)
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
                group_glyphs=scatter.group_glyphs,
                top_other_glyph=scatter.top_other_glyph,
                top_named_glyph=scatter.top_named_glyph,
                gradient_glyph=scatter.gradient_glyph,
                color_bar=scatter.color_bar,
                feature_picker=feature_picker,
                top_n_slider=top_n_slider,
            ),
            code=COLOR_BY_MODE,
        ),
    )

    reduced_kept_source = ColumnDataSource({f"f{k}": views.l2[:, k] for k in range(top.n_kept)})
    feature_values_kept_source: ColumnDataSource | None = None

    if feature_values_kept is not None:
        feature_values_kept_source = ColumnDataSource(
            {f"f{k}": feature_values_kept[:, k] for k in range(top.n_kept)}
        )

    feature_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                reduced_source=reduced_kept_source,
                values_source=feature_values_kept_source,
                mapper=scatter.gradient_mapper,
                feature_names=top.kept_names,
            ),
            code=FEATURE_PICKER,
        ),
    )

    top_n_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                names_by_rank=top_feature_names_by_rank,
            ),
            code=TOP_N_SLIDER,
        ),
    )

    return ControlsArtifacts(
        color_by_prefix=color_by_prefix,
        color_by_widget=color_by_widget,
        feature_picker=feature_picker,
        top_n_slider=top_n_slider,
    )
