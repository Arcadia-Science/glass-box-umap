ignore_regex = [
    "*/tests/*",
]

skip_dict: dict[str, list[str]] = {
    "package": [
        "glass_box_umap.parametric_umap",
        "glass_box_umap.plotting.bokeh",
    ],
    "module": [
        "glass_box_umap.core",
        "glass_box_umap.components",
        "glass_box_umap.utils",
        "glass_box_umap.plotting.check",
        "glass_box_umap.plotting.mpl",
    ],
    "function": [],
    "class": [],
    "attribute": [],
    "method": [],
    "property": [],
    "exception": [],
    "data": [],
}

keep_dict: dict[str, list[str]] = {
    "package": [],
    "module": [],
    "function": [],
    "class": [
        "glass_box_umap.GlassBoxUMAP",
        "glass_box_umap.ParametricUMAP",
    ],
    "attribute": [],
    "method": [],
    "property": [],
    "exception": [],
    "data": [],
}
