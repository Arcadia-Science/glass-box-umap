const t = cb_obj.value;
const tfg = scatter_source.data["top_feature_group"];
const ranks = scatter_source.data["sample_rank"];
const modeName = collapse_checkbox.active.includes(0) ? "Collapsed" : "Expanded";
const activeNames = top_rank_sources[modeName].data["name"];
const n = ranks.length;
for (let i = 0; i < n; i++) {
    tfg[i] = ranks[i] < t ? activeNames[ranks[i]] : "(other)";
}
scatter_source.change.emit();

if (color_modes[color_by_widget.active] === "Top feature") {
    function escapeHtml(value) {
        return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
    }
    let html = "<b>Color key</b><div>";
    for (const name of activeNames.slice(0, t)) {
        html += `<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0">`;
        html += `<span style="width:10px;height:10px;background:${top_colors[name]};margin-right:4px"></span>`;
        html += `${escapeHtml(name)}</span>`;
    }
    html += '<span style="display:inline-flex;align-items:center;margin:2px 10px 2px 0">';
    html += '<span style="width:10px;height:10px;background:#cccccc;margin-right:4px"></span>(other)</span></div>';
    color_key.text = html;
}
