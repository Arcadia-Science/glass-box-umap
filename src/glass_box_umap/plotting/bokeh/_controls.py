from dataclasses import dataclass

from bokeh.models import (
    AutocompleteInput,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    InlineStyleSheet,
    RadioButtonGroup,
    Select,
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
from ._js import COLOR_BY_MODE, FEATURE_PICKER, FEATURE_VIEW_PICKER, TOP_N_SLIDER
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
    label_picker: Select
    label_class_filter: CheckboxGroup
    feature_view_picker: Select
    point_size_slider: Slider
    point_alpha_slider: Slider


def build_controls(
    color_modes: list[str],
    initial_mode: str,
    initial_t: int,
    label_modes: list[str],
    label_factors: dict[str, list[str]],
    n_distinct: int,
    top: TopFeatures,
    l2_source: ColumnDataSource,
    feature_view_sources: dict[str, ColumnDataSource],
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
    label_picker = Select(
        title="Label",
        options=label_modes,
        value=label_modes[0] if label_modes else "",
        width=260,
        visible=(initial_mode == "Label"),
        styles={"color": LABEL_COLOR},
    )
    initial_label_factors = label_factors[label_modes[0]] if label_modes else []
    label_class_filter = CheckboxGroup(
        labels=initial_label_factors,
        active=list(range(len(initial_label_factors))),
        visible=(initial_mode == "Label"),
        height=150,
        styles={"color": LABEL_COLOR},
        stylesheets=[
            InlineStyleSheet(
                css=".bk-input-group { max-height: 150px; overflow-y: auto; overflow-x: hidden; }"
            )
        ],
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
        styles={"color": LABEL_COLOR},
        stylesheets=[InlineStyleSheet(css=f".bk-input {{ color: {LABEL_COLOR}; }}")],
    )
    feature_view_picker = Select(
        title="Feature map",
        options=list(feature_view_sources),
        value=next(iter(feature_view_sources)),
        width=260,
        visible=(initial_mode == "Feature" and len(feature_view_sources) > 1),
        styles={"color": LABEL_COLOR},
    )
    top_n_slider = Slider(
        start=1,
        # Bokeh rejects a slider whose start and end are equal.
        end=max(n_distinct, 2),
        value=max(initial_t, 1),
        step=1,
        title="Top features",
        width=260,
        visible=(initial_mode == "Top feature"),
        styles={"color": LABEL_COLOR},
    )
    point_size_slider = Slider(
        start=1, end=14, value=5, step=1, title="Point size", width=125
    )
    point_alpha_slider = Slider(
        start=0.1, end=1.0, value=0.6, step=0.05, title="Point opacity", width=125
    )

    color_by_widget.js_on_change(
        "active",
        CustomJS(
            args=dict(
                color_modes=color_modes,
                label_glyphs=scatter.label_glyphs,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
                top_other_glyph=scatter.top_other_glyph,
                top_named_glyph=scatter.top_named_glyph,
                gradient_glyph=scatter.gradient_glyph,
                color_bar=scatter.color_bar,
                feature_picker=feature_picker,
                feature_view_picker=feature_view_picker,
                top_n_slider=top_n_slider,
            ),
            code=COLOR_BY_MODE,
        ),
    )

    feature_view_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                view_sources=feature_view_sources, reduced_source=l2_source,
                scatter_source=scatter_source, mapper=scatter.gradient_mapper,
                feature_picker=feature_picker, feature_names=top.kept_names,
                degenerate_eps=DEGENERATE_RANGE_EPS, degenerate_frac=DEGENERATE_RANGE_FRAC,
                degenerate_min_span=DEGENERATE_RANGE_MIN_SPAN,
            ),
            code=FEATURE_VIEW_PICKER,
        ),
    )

    label_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
                color_by_widget=color_by_widget, color_modes=color_modes,
                label_glyphs=scatter.label_glyphs, label_picker=label_picker,
                label_class_filter=label_class_filter, label_factors=label_factors,
            ),
            code="""
label_class_filter.labels = label_factors[label_picker.value];
label_class_filter.active = Array.from({length: label_class_filter.labels.length}, (_, i) => i);
const is_label = color_modes[color_by_widget.active] === "Label";
const selected = new Set(label_class_filter.active.map(i => label_class_filter.labels[i]));
for (const [label_mode, glyphs] of Object.entries(label_glyphs)) {
  for (const glyph of glyphs) {
    glyph.visible = is_label && label_mode === label_picker.value
      && selected.has(glyph.view.filter.group);
  }
}
label_class_filter.change.emit();
""",
        ),
    )

    label_class_filter.js_on_change(
        "active",
        CustomJS(
            args=dict(
                color_by_widget=color_by_widget, color_modes=color_modes,
                label_glyphs=scatter.label_glyphs, label_picker=label_picker,
                label_class_filter=label_class_filter,
            ),
            code="""
const is_label = color_modes[color_by_widget.active] === "Label";
const selected = new Set(label_class_filter.active.map(i => label_class_filter.labels[i]));
for (const [label_mode, glyphs] of Object.entries(label_glyphs)) {
  for (const glyph of glyphs) {
    glyph.visible = is_label && label_mode === label_picker.value
      && selected.has(glyph.view.filter.group);
  }
}
""",
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

    point_size_slider.js_on_change(
        "value", CustomJS(args=dict(glyphs=scatter.point_glyphs), code="""
for (const renderer of glyphs) renderer.glyph.size = cb_obj.value;
"""),
    )
    point_alpha_slider.js_on_change(
        "value", CustomJS(args=dict(glyphs=scatter.point_glyphs), code="""
for (const renderer of glyphs) {
  renderer.glyph.fill_alpha = cb_obj.value;
  renderer.glyph.line_alpha = cb_obj.value;
}
"""),
    )

    return ControlsArtifacts(
        color_by_prefix=color_by_prefix,
        color_by_widget=color_by_widget,
        feature_picker=feature_picker,
        top_n_slider=top_n_slider,
        label_picker=label_picker,
        label_class_filter=label_class_filter,
        feature_view_picker=feature_view_picker,
        point_size_slider=point_size_slider,
        point_alpha_slider=point_alpha_slider,
    )
