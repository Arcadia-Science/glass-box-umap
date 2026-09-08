const activeLabels = new Set(
    label_class_filter.active.map(i => label_class_filter.labels[i])
);
const labelField = label_fields[label_picker.value];
const subset = scatter_source.data["subset_visible"];
if (labelField !== undefined) {
    const labels = scatter_source.data[labelField];
    for (let i = 0; i < subset.length; i++) subset[i] = activeLabels.has(labels[i]) ? 1 : 0;
}

const modeName = collapse_checkbox.active.includes(0) ? "Collapsed" : "Expanded";
const modeSource = top_mode_sources[modeName].data;
const featureNames = top_names[modeName];
const topIdx = modeSource["top_idx"];
const counts = new Map();
for (let i = 0; i < topIdx.length; i++) {
    if (subset[i]) counts.set(topIdx[i], (counts.get(topIdx[i]) ?? 0) + 1);
}
const rankedIdx = Array.from(counts.keys()).sort((a, b) => {
    const difference = counts.get(b) - counts.get(a);
    return difference !== 0 ? difference : featureNames[a].localeCompare(featureNames[b]);
});
const rankOf = new Map(rankedIdx.map((index, rank) => [index, rank]));
const rankedNames = rankedIdx.map(index => featureNames[index]);
top_rank_sources[modeName].data = {name: rankedNames};

const ranks = scatter_source.data["sample_rank"];
const groups = scatter_source.data["top_feature_group"];
const hoverNames = scatter_source.data["top_feature_name"];
const hoverValues = scatter_source.data["top_data_value"];
const modeValues = modeSource["top_value"];
const distinct = Math.max(rankedNames.length, 1);
top_n_slider.end = Math.max(distinct, 2);
top_n_slider.value = Math.max(1, Math.min(top_n_slider.value, distinct));
const threshold = top_n_slider.value;
for (let i = 0; i < topIdx.length; i++) {
    const rank = rankOf.has(topIdx[i]) ? rankOf.get(topIdx[i]) : distinct;
    ranks[i] = rank;
    hoverNames[i] = featureNames[topIdx[i]];
    if (hoverValues !== undefined && modeValues !== undefined) hoverValues[i] = modeValues[i];
    groups[i] = subset[i] && rank < threshold ? rankedNames[rank] : "(other)";
}

function escapeHtml(value) {
    return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
if (color_modes[color_by_widget.active] === "Label") {
    const colors = label_colors[label_picker.value];
    let html = "<b>Color key</b><div>";
    for (const label of label_class_filter.labels) {
        if (!activeLabels.has(label)) continue;
        html += '<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0">';
        html += `<span style="width:10px;height:10px;background:${colors[label]};margin-right:4px"></span>`;
        html += `${escapeHtml(label)}</span>`;
    }
    html += "</div>";
    color_key.text = html;
}
scatter_source.change.emit();
top_n_slider.change.emit();
if (hierarchy_level_slider.visible) hierarchy_level_slider.change.emit();
