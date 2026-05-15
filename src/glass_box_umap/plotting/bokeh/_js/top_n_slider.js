const t = cb_obj.value;
const tfg = scatter_source.data["top_feature_group"];
const ranks = scatter_source.data["sample_rank"];
const n = ranks.length;
for (let i = 0; i < n; i++) {
    tfg[i] = ranks[i] < t ? names_by_rank[ranks[i]] : "(other)";
}
scatter_source.change.emit();
