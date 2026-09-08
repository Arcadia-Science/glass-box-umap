const sourceName = source_picker.value;
const levelIndex = Math.round(level_slider.value);
const hierarchySource = hierarchy_sources[sourceName];
const labels = hierarchySource.data["level_" + levelIndex];
const colors = hierarchy_colors[sourceName][levelIndex];
const metadata = hierarchy_metadata[sourceName][levelIndex];
const cluster = scatter_source.data["hierarchy_cluster"];
const clusterColor = scatter_source.data["hierarchy_color"];
const clusterSize = scatter_source.data["hierarchy_size"];
const topFamilies = scatter_source.data["hierarchy_top_families"];
const subset = scatter_source.data["subset_visible"];
const subsetCounts = {};
for (let i = 0; i < labels.length; i++) {
  if (subset[i]) subsetCounts[labels[i]] = (subsetCounts[labels[i]] ?? 0) + 1;
}

for (let i = 0; i < labels.length; i++) {
  const label = labels[i];
  const meta = metadata[label];
  cluster[i] = label;
  clusterColor[i] = colors[label];
  clusterSize[i] = String(subsetCounts[label] ?? 0);
  topFamilies[i] = meta.top_families;
}
scatter_source.change.emit();

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
const entries = Object.keys(metadata).sort((a, b) => {
  if (a === "Not OOF") return 1;
  if (b === "Not OOF") return -1;
  return a.localeCompare(b, undefined, {numeric: true});
});
let html = `<b>${escapeHtml(sourceName)} — ${escapeHtml(level_names[sourceName][levelIndex])}</b>`;
html += '<div style="max-height:150px;overflow-y:auto;margin-top:4px">';
for (const label of entries) {
  if ((subsetCounts[label] ?? 0) === 0) continue;
  const meta = metadata[label];
  html += `<div style="margin:3px 0"><span style="display:inline-block;width:10px;height:10px;`;
  html += `background:${colors[label]};margin-right:5px"></span><b>${escapeHtml(label)}</b> `;
  html += `(${escapeHtml(subsetCounts[label])} in subset) — ${escapeHtml(meta.top_families)}</div>`;
}
html += "</div>";
hierarchy_key.text = html;
level_slider.title = "Cluster depth — " + level_names[sourceName][levelIndex];
