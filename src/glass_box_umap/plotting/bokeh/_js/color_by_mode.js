const mode = color_modes[cb_obj.active];
const is_label = (mode === "Label");
const is_feature = (mode === "Feature");
const is_top = (mode === "Top feature");
for (const [label_mode, glyphs] of Object.entries(label_glyphs)) {
  for (const glyph of glyphs) glyph.visible = is_label && (label_mode === label_picker.value);
}
top_other_glyph.visible = is_top;
top_named_glyph.visible = is_top;
gradient_glyph.visible = is_feature;
color_bar.visible = is_feature;
feature_picker.visible = is_feature;
top_n_slider.visible = is_top;
label_picker.visible = is_label;
