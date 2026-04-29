const tfg = source.data["top_feature_group"];
const out = new Array(tfg.length);
for (let i = 0; i < tfg.length; i++) out[i] = tfg[i] !== "(other)";
return out;
