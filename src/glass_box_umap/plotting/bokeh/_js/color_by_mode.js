const mode = color_modes[cb_obj.active];
const is_label = (mode === "Label");
const is_feature = (mode === "Feature");
const is_top = (mode === "Top feature");
<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
for (const g of group_glyphs) g.visible = is_group;
=======
const is_hierarchy = (mode === "Hierarchy");
>>>>>>> Stashed changes
const selected = new Set(label_class_filter.active.map(i => label_class_filter.labels[i]));
for (const [label_mode, glyphs] of Object.entries(label_glyphs)) {
  for (const glyph of glyphs) {
    glyph.visible = is_label && (label_mode === label_picker.value) && selected.has(glyph.view.filter.group);
  }
}
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
top_other_glyph.visible = is_top;
top_named_glyph.visible = is_top;
gradient_glyph.visible = is_feature;
if (hierarchy_glyph !== null) hierarchy_glyph.visible = is_hierarchy;
color_bar.visible = is_feature;
feature_picker.visible = is_feature;
feature_view_picker.visible = is_feature && feature_view_picker.options.length > 1;
top_n_slider.visible = is_top;
<<<<<<< Updated upstream
label_picker.visible = is_label;
label_class_filter.visible = is_label;
=======
<<<<<<< Updated upstream
=======
label_picker.visible = is_label;
label_class_filter.visible = is_label;
select_none_checkbox.visible = is_label;
collapse_checkbox.visible = is_top;
color_key.visible = is_label || is_top;
if (is_top) top_n_slider.change.emit();
hierarchy_source_picker.visible = is_hierarchy && hierarchy_source_picker.options.length > 1;
hierarchy_level_slider.visible = is_hierarchy;
hierarchy_key.visible = is_hierarchy;
>>>>>>> Stashed changes
>>>>>>> Stashed changes
