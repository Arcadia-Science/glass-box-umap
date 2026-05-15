from dataclasses import dataclass

from bokeh.models import (
    AutocompleteInput,
    ColumnDataSource,
    CustomJS,
    Div,
    InlineStyleSheet,
    RadioButtonGroup,
    Slider,
)

from ._colors import (
    DEGENERATE_RANGE_EPS,
    DEGENERATE_RANGE_FRAC,
    DEGENERATE_RANGE_MIN_SPAN,
    LABEL_COLOR,
    LABEL_FONT_SIZE,
)
from ._data import TopFeatures
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
    l2_source: ColumnDataSource,
    feature_values_source: ColumnDataSource | None,
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
        styles={"color": LABEL_COLOR, "font-size": LABEL_FONT_SIZE, "padding-top": "8px"},
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
        styles={"color": LABEL_COLOR},
        stylesheets=[InlineStyleSheet(css=f".bk-input {{ color: {LABEL_COLOR}; }}")],
    )
    top_n_slider = Slider(
        start=1,
        end=max(n_distinct, 1),
        value=max(initial_t, 1),
        step=1,
        title="Top features",
        width=260,
        visible=(initial_mode == "Top feature"),
        styles={"color": LABEL_COLOR},
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

    feature_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter_source=scatter_source,
                reduced_source=l2_source,
                values_source=feature_values_source,
                mapper=scatter.gradient_mapper,
                feature_names=top.kept_names,
                degenerate_eps=DEGENERATE_RANGE_EPS,
                degenerate_frac=DEGENERATE_RANGE_FRAC,
                degenerate_min_span=DEGENERATE_RANGE_MIN_SPAN,
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
