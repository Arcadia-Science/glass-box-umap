from dataclasses import dataclass
from html import escape
from typing import Any

from bokeh.models import (
    AutocompleteInput,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    CustomJSTickFormatter,
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
    pick_palette,
)
from ._data import TopFeatures
<<<<<<< Updated upstream
from ._js import COLOR_BY_MODE, FEATURE_PICKER, FEATURE_VIEW_PICKER, TOP_N_SLIDER
=======
<<<<<<< Updated upstream
from ._js import COLOR_BY_MODE, FEATURE_PICKER, TOP_N_SLIDER
=======
from ._js import (
    COLOR_BY_MODE,
    FEATURE_PICKER,
    FEATURE_VIEW_PICKER,
    HIERARCHY_UPDATE,
    SUBSET_UPDATE,
    TOP_N_SLIDER,
)
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    label_picker: Select
    label_class_filter: CheckboxGroup
    feature_view_picker: Select
    point_size_slider: Slider
    point_alpha_slider: Slider
=======
<<<<<<< Updated upstream
=======
    label_picker: Select
    label_class_filter: CheckboxGroup
    select_none_checkbox: CheckboxGroup
    collapse_checkbox: CheckboxGroup
    color_key: Div
    feature_view_picker: Select
    point_size_slider: Slider
    point_alpha_slider: Slider
    hierarchy_source_picker: Select
    hierarchy_level_slider: Slider
    hierarchy_key: Div
>>>>>>> Stashed changes
>>>>>>> Stashed changes


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
    collapsed_top_feature_names_by_rank: list[str],
    top_mode_sources: dict[str, ColumnDataSource],
    top_rank_sources: dict[str, ColumnDataSource],
    expanded_top_names: list[str],
    collapsed_top_names: list[str],
    scatter_source: ColumnDataSource,
    scatter: ScatterArtifacts,
    hierarchy_sources: dict[str, ColumnDataSource],
    hierarchy_level_names: dict[str, list[str]],
    hierarchy_colors: dict[str, list[dict[str, str]]],
    hierarchy_metadata: dict[str, list[dict[str, dict[str, Any]]]],
    hierarchy_key_html: str,
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
<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
    label_picker = Select(
        title="Label",
        options=label_modes,
        value=label_modes[0] if label_modes else "",
        width=260,
        visible=(initial_mode == "Label"),
        styles={"color": LABEL_COLOR},
    )
    initial_label_factors = label_factors[label_modes[0]] if label_modes else []
<<<<<<< Updated upstream
=======
    label_colors = {
        name: dict(zip(factors, pick_palette(len(factors)), strict=False))
        for name, factors in label_factors.items()
    }
    top_color_names = list(
        dict.fromkeys([*top_feature_names_by_rank, *collapsed_top_feature_names_by_rank])
    )
    top_colors = dict(zip(top_color_names, pick_palette(len(top_color_names)), strict=False))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
    select_none_checkbox = CheckboxGroup(
        labels=["Select none"],
        active=[],
        visible=(initial_mode == "Label"),
        styles={"color": LABEL_COLOR},
    )
    collapse_checkbox = CheckboxGroup(
        labels=["Collapse numbered features across region (max L2)"],
        active=[],
        visible=(initial_mode == "Top feature"),
        styles={"color": LABEL_COLOR},
    )
    initial_label_colors = label_colors[label_modes[0]] if label_modes else {}
    initial_colors = initial_label_colors
    initial_key_rows = (
        "".join(
            '<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0">'
            f'<span style="width:10px;height:10px;background:{escape(initial_colors[factor])};'
            f'margin-right:4px"></span>{escape(factor)}</span>'
            for factor in initial_label_factors
        )
        if label_modes
        else ""
    )
    color_key = Div(
        text=f"<b>Color key</b><div>{initial_key_rows}</div>",
        width=420,
        visible=(initial_mode in {"Label", "Top feature"}),
        styles={"color": LABEL_COLOR, "font-size": LABEL_FONT_SIZE},
    )
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    point_size_slider = Slider(
        start=1, end=14, value=5, step=1, title="Point size", width=125
    )
    point_alpha_slider = Slider(
        start=0.1, end=1.0, value=0.6, step=0.05, title="Point opacity", width=125
    )
=======
<<<<<<< Updated upstream
=======
    point_size_slider = Slider(start=1, end=14, value=5, step=1, title="Point size", width=125)
    point_alpha_slider = Slider(
        start=0.1, end=1.0, value=0.6, step=0.05, title="Point opacity", width=125
    )
    hierarchy_source_names = list(hierarchy_sources)
    initial_hierarchy_source = hierarchy_source_names[0] if hierarchy_source_names else ""
    initial_level_names = hierarchy_level_names.get(initial_hierarchy_source, ["Clusters"])
    hierarchy_source_picker = Select(
        title="Hierarchy source",
        options=hierarchy_source_names,
        value=initial_hierarchy_source,
        width=260,
        visible=(initial_mode == "Hierarchy" and len(hierarchy_source_names) > 1),
        styles={"color": LABEL_COLOR},
    )
    hierarchy_level_slider = Slider(
        start=0,
        # This hidden control still participates in Bokeh validation when no
        # hierarchy is supplied; a 0..0 slider makes the whole document fail
        # validation before it can render.
        end=max(len(initial_level_names) - 1, 1),
        value=0,
        step=1,
        title=f"Cluster depth — {initial_level_names[0]}",
        width=260,
        visible=(initial_mode == "Hierarchy"),
        show_value=False,
        format=CustomJSTickFormatter(
            args=dict(source_picker=hierarchy_source_picker, level_names=hierarchy_level_names),
            code=(
                "const names = level_names[source_picker.value] ?? ['Clusters']; "
                "return names[Math.round(tick)] ?? '';"
            ),
        ),
        styles={"color": LABEL_COLOR},
    )
    hierarchy_key = Div(
        text=hierarchy_key_html,
        width=300,
        visible=(initial_mode == "Hierarchy"),
        styles={"color": LABEL_COLOR, "font-size": LABEL_FONT_SIZE},
    )
>>>>>>> Stashed changes
>>>>>>> Stashed changes

    color_by_widget.js_on_change(
        "active",
        CustomJS(
            args=dict(
                color_modes=color_modes,
<<<<<<< Updated upstream
                label_glyphs=scatter.label_glyphs,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
=======
<<<<<<< Updated upstream
                group_glyphs=scatter.group_glyphs,
=======
                label_glyphs=scatter.label_glyphs,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
                select_none_checkbox=select_none_checkbox,
                hierarchy_glyph=scatter.hierarchy_glyph,
                hierarchy_source_picker=hierarchy_source_picker,
                hierarchy_level_slider=hierarchy_level_slider,
                hierarchy_key=hierarchy_key,
>>>>>>> Stashed changes
>>>>>>> Stashed changes
                top_other_glyph=scatter.top_other_glyph,
                top_named_glyph=scatter.top_named_glyph,
                gradient_glyph=scatter.gradient_glyph,
                color_bar=scatter.color_bar,
                feature_picker=feature_picker,
                feature_view_picker=feature_view_picker,
                top_n_slider=top_n_slider,
                collapse_checkbox=collapse_checkbox,
                color_key=color_key,
            ),
            code=COLOR_BY_MODE,
        ),
    )

<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
    feature_view_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
<<<<<<< Updated upstream
                view_sources=feature_view_sources, reduced_source=l2_source,
                scatter_source=scatter_source, mapper=scatter.gradient_mapper,
                feature_picker=feature_picker, feature_names=top.kept_names,
                degenerate_eps=DEGENERATE_RANGE_EPS, degenerate_frac=DEGENERATE_RANGE_FRAC,
=======
                view_sources=feature_view_sources,
                reduced_source=l2_source,
                scatter_source=scatter_source,
                mapper=scatter.gradient_mapper,
                feature_picker=feature_picker,
                feature_names=top.kept_names,
                degenerate_eps=DEGENERATE_RANGE_EPS,
                degenerate_frac=DEGENERATE_RANGE_FRAC,
>>>>>>> Stashed changes
                degenerate_min_span=DEGENERATE_RANGE_MIN_SPAN,
            ),
            code=FEATURE_VIEW_PICKER,
        ),
    )

    label_picker.js_on_change(
        "value",
        CustomJS(
            args=dict(
<<<<<<< Updated upstream
                color_by_widget=color_by_widget, color_modes=color_modes,
                label_glyphs=scatter.label_glyphs, label_picker=label_picker,
                label_class_filter=label_class_filter, label_factors=label_factors,
=======
                color_by_widget=color_by_widget,
                color_modes=color_modes,
                label_glyphs=scatter.label_glyphs,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
                label_factors=label_factors,
                select_none_checkbox=select_none_checkbox,
>>>>>>> Stashed changes
            ),
            code="""
label_class_filter.labels = label_factors[label_picker.value];
label_class_filter.active = Array.from({length: label_class_filter.labels.length}, (_, i) => i);
<<<<<<< Updated upstream
=======
select_none_checkbox.active = [];
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                color_by_widget=color_by_widget, color_modes=color_modes,
                label_glyphs=scatter.label_glyphs, label_picker=label_picker,
                label_class_filter=label_class_filter,
=======
                color_by_widget=color_by_widget,
                color_modes=color_modes,
                label_glyphs=scatter.label_glyphs,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
                select_none_checkbox=select_none_checkbox,
                label_fields={name: f"label_set_{i}" for i, name in enumerate(label_modes)},
                label_colors=label_colors,
                scatter_source=scatter_source,
                top_mode_sources=top_mode_sources,
                top_rank_sources=top_rank_sources,
                top_names={"Expanded": expanded_top_names, "Collapsed": collapsed_top_names},
                names_by_rank=top_feature_names_by_rank,
                collapsed_names_by_rank=collapsed_top_feature_names_by_rank,
                top_n_slider=top_n_slider,
                collapse_checkbox=collapse_checkbox,
                color_key=color_key,
                hierarchy_level_slider=hierarchy_level_slider,
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
if (label_class_filter.active.length > 0) select_none_checkbox.active = [];
"""
            + SUBSET_UPDATE,
        ),
    )

    select_none_checkbox.js_on_change(
        "active",
        CustomJS(
            args=dict(label_class_filter=label_class_filter),
            code="""
if (cb_obj.active.includes(0) && label_class_filter.active.length > 0) {
  label_class_filter.active = [];
}
>>>>>>> Stashed changes
""",
        ),
    )

<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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
                top_rank_sources=top_rank_sources,
                collapse_checkbox=collapse_checkbox,
                color_key=color_key,
                color_by_widget=color_by_widget,
                color_modes=color_modes,
                top_colors=top_colors,
            ),
            code=TOP_N_SLIDER,
        ),
    )

<<<<<<< Updated upstream
    point_size_slider.js_on_change(
        "value", CustomJS(args=dict(glyphs=scatter.point_glyphs), code="""
for (const renderer of glyphs) renderer.glyph.size = cb_obj.value;
"""),
    )
    point_alpha_slider.js_on_change(
        "value", CustomJS(args=dict(glyphs=scatter.point_glyphs), code="""
=======
<<<<<<< Updated upstream
=======
    collapse_checkbox.js_on_change(
        "active",
        CustomJS(
            args=dict(
                color_by_widget=color_by_widget,
                color_modes=color_modes,
                label_picker=label_picker,
                label_class_filter=label_class_filter,
                select_none_checkbox=select_none_checkbox,
                label_fields={name: f"label_set_{i}" for i, name in enumerate(label_modes)},
                label_colors=label_colors,
                scatter_source=scatter_source,
                top_mode_sources=top_mode_sources,
                top_rank_sources=top_rank_sources,
                top_names={"Expanded": expanded_top_names, "Collapsed": collapsed_top_names},
                names_by_rank=top_feature_names_by_rank,
                collapsed_names_by_rank=collapsed_top_feature_names_by_rank,
                top_n_slider=top_n_slider,
                collapse_checkbox=collapse_checkbox,
                color_key=color_key,
                hierarchy_level_slider=hierarchy_level_slider,
            ),
            code=SUBSET_UPDATE,
        ),
    )

    hierarchy_callback_args = dict(
        source_picker=hierarchy_source_picker,
        level_slider=hierarchy_level_slider,
        hierarchy_sources=hierarchy_sources,
        level_names=hierarchy_level_names,
        hierarchy_colors=hierarchy_colors,
        hierarchy_metadata=hierarchy_metadata,
        hierarchy_key=hierarchy_key,
        scatter_source=scatter_source,
    )
    hierarchy_level_slider.js_on_change(
        "value", CustomJS(args=hierarchy_callback_args, code=HIERARCHY_UPDATE)
    )
    hierarchy_source_picker.js_on_change(
        "value",
        CustomJS(
            args=hierarchy_callback_args,
            code="""
const names = level_names[cb_obj.value];
level_slider.end = Math.max(names.length - 1, 0);
if (level_slider.value > level_slider.end) level_slider.value = level_slider.end;
"""
            + HIERARCHY_UPDATE,
        ),
    )

    point_size_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(glyphs=scatter.point_glyphs),
            code="""
for (const renderer of glyphs) renderer.glyph.size = cb_obj.value;
""",
        ),
    )
    point_alpha_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(glyphs=scatter.point_glyphs),
            code="""
>>>>>>> Stashed changes
for (const renderer of glyphs) {
  renderer.glyph.fill_alpha = cb_obj.value;
  renderer.glyph.line_alpha = cb_obj.value;
}
<<<<<<< Updated upstream
"""),
    )

=======
""",
        ),
    )

>>>>>>> Stashed changes
>>>>>>> Stashed changes
    return ControlsArtifacts(
        color_by_prefix=color_by_prefix,
        color_by_widget=color_by_widget,
        feature_picker=feature_picker,
        top_n_slider=top_n_slider,
<<<<<<< Updated upstream
        label_picker=label_picker,
        label_class_filter=label_class_filter,
        feature_view_picker=feature_view_picker,
        point_size_slider=point_size_slider,
        point_alpha_slider=point_alpha_slider,
=======
<<<<<<< Updated upstream
=======
        label_picker=label_picker,
        label_class_filter=label_class_filter,
        select_none_checkbox=select_none_checkbox,
        collapse_checkbox=collapse_checkbox,
        color_key=color_key,
        feature_view_picker=feature_view_picker,
        point_size_slider=point_size_slider,
        point_alpha_slider=point_alpha_slider,
        hierarchy_source_picker=hierarchy_source_picker,
        hierarchy_level_slider=hierarchy_level_slider,
        hierarchy_key=hierarchy_key,
>>>>>>> Stashed changes
>>>>>>> Stashed changes
    )
