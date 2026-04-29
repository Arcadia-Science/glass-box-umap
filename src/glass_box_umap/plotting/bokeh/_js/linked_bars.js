const view = view_widget.active;

let active_data;
if (view === 0) {
    active_data = contrib_sources[0].data;
} else if (view === 1) {
    if (!normed_l2_cache.data) {
        const src = contrib_sources[0].data;
        const totals = new Float64Array(n_samples);
        for (let k = 0; k < n_kept; k++) {
            const col = src["c" + k];
            for (let i = 0; i < n_samples; i++) {
                totals[i] += col[i];
            }
        }
        for (let i = 0; i < n_samples; i++) {
            if (totals[i] < 1e-12) totals[i] = 1.0;
        }
        const cache = {};
        for (let k = 0; k < n_kept; k++) {
            const s = src["c" + k];
            const dst = new Float64Array(n_samples);
            for (let i = 0; i < n_samples; i++) {
                dst[i] = s[i] / totals[i];
            }
            cache["c" + k] = dst;
        }
        normed_l2_cache.data = cache;
    }
    active_data = normed_l2_cache.data;
} else if (view === 2) {
    active_data = contrib_sources[1].data;
} else {
    active_data = contrib_sources[2].data;
}

const sel = scatter_source.selected.indices;
const indices = sel.length
    ? sel
    : Array.from({length: n_samples}, (_, i) => i);
const n = indices.length;

const means = new Float64Array(n_kept);
for (let k = 0; k < n_kept; k++) {
    const col = active_data["c" + k];
    let s = 0.0;
    for (let j = 0; j < n; j++) {
        s += col[indices[j]];
    }
    means[k] = s / n;
}

const scored = new Array(n_kept);
for (let k = 0; k < n_kept; k++) {
    scored[k] = { idx: k, score: Math.abs(means[k]) };
}
scored.sort((a, b) => b.score - a.score);
const top = scored.slice(0, display_k).reverse();

const feat = top.map(t => feature_names[t.idx]);
const vals = top.map(t => means[t.idx]);
bar_source.data = { feature: feat, mean: vals };
bar_range.factors = feat;

heading_div.text = `<b>Mean contribution — ${view_labels[view]}</b>`;
