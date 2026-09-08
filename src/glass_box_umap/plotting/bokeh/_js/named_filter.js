const tfg = source.data["top_feature_group"];
const visible = source.data["subset_visible"];
const out = new Array(tfg.length);
for (let i = 0; i < tfg.length; i++) {
    out[i] = tfg[i] !== "(other)" && Boolean(visible[i]);
}
return out;
