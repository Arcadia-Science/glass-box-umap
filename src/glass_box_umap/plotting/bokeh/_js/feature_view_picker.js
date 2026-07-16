const source = view_sources[cb_obj.value];
for (const key of Object.keys(source.data)) {
    reduced_source.data[key] = source.data[key].slice();
}
reduced_source.change.emit();

const idx = feature_names.indexOf(feature_picker.value);
if (idx < 0) { return; }
const col = reduced_source.data["c" + idx];
const copy = new Float64Array(col.length);
let lo = Infinity, hi = -Infinity;
for (let i = 0; i < col.length; i++) {
    const value = col[i];
    copy[i] = value;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
}
if (hi - lo < degenerate_eps) {
    const mid = (lo + hi) / 2;
    const span = Math.max(Math.abs(mid) * degenerate_frac, degenerate_min_span);
    lo = mid - span;
    hi = mid + span;
}
scatter_source.data["color_value"] = copy;
scatter_source.change.emit();
mapper.low = lo;
mapper.high = hi;
