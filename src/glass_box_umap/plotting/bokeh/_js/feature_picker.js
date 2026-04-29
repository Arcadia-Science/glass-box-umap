const idx = feature_names.indexOf(cb_obj.value);
if (idx < 0) { return; }
const col = reduced_source.data["f" + idx];
const copy = new Float64Array(col.length);
let lo = Infinity, hi = -Infinity;
for (let i = 0; i < col.length; i++) {
    const v = col[i];
    copy[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
}
if (hi - lo < 1e-12) {
    const mid = (lo + hi) / 2;
    const span = Math.max(Math.abs(mid) * 0.05, 1e-6);
    lo = mid - span;
    hi = mid + span;
}
scatter_source.data["color_value"] = copy;
if (values_source !== null) {
    const vcol = values_source.data["f" + idx];
    const vcopy = new Float64Array(vcol.length);
    for (let i = 0; i < vcol.length; i++) vcopy[i] = vcol[i];
    scatter_source.data["picker_data_value"] = vcopy;
}
scatter_source.change.emit();
mapper.low = lo;
mapper.high = hi;
