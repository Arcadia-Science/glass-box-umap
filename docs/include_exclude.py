ignore_regex = [
    "*/tests/*",
]

skip_dict: dict[str, list[str]] = {
    "package": [
        "glass_box_umap.parametric_umap",
    ],
    "module": [
        "glass_box_umap.core",
        "glass_box_umap.components",
        "glass_box_umap.utils",
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
