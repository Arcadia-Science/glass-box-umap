# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "glass-box-umap"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "myst_sphinx_gallery",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_paramlinks",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_togglebutton",
    "custom_skip_members",
    "resolve_missing_references",
    "fix_dataclass_defaults",
    "restructure_class_layout",
    "patch_type_alias_subscript",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# NOTE: Don't use this for excluding python files, use `autoapi_ignore` below
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "**README.md", "autoapi/index.rst"]

# -- Global options ----------------------------------------------------------

# Don't mess with double-dash used in CLI options
smartquotes_action = "qe"

# -- Notebook rendering -------------------------------------------------

nb_execution_mode = "off"
nb_execution_allow_errors = True

suppress_warnings = ["mystnb.unknown_mime_type"]

nb_mime_priority_overrides = [
    ("html", "application/javascript", 5),
    ("html", "application/vnd.bokehjs_load.v0+json", 100),
    ("html", "application/vnd.bokehjs_exec.v0+json", 100),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Atkinson+Hyperlegible+Next:ital,wght@0,200..800;1,200..800&"
    "family=Merriweather:ital,wght@0,300..900;1,300..900&"
    "display=swap"
)
html_theme_options = {
    "light_logo": "lockup.png",
    "dark_logo": "dark_lockup.png",
    "light_css_variables": {
        "font-stack": '"Atkinson Hyperlegible Next", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        "font-stack--monospace": 'Menlo, ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", monospace',
        # Furo doesn't have separate variables for h1 vs h2+; h2+ overridden in css/headings.css
        "font-stack--headings": "Merriweather, Georgia, serif",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_assets"]

# -- Napoleon options
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

napolean_use_param = True  # Each parameter is its own :param: directive
napolean_attr_annotations = True

# -- Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
}

# -- sphinx-tabs options
sphinx_tabs_disable_tab_closing = True

# -- copybutton options
copybutton_exclude = ".linenos, .gp, .go"

# -- myst options
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

togglebutton_hint = ""
togglebutton_hint_hide = ""

# -- autoapi configuration ---------------------------------------------------

add_module_names = False
autodoc_typehints = "both"  # autoapi respects this
autodoc_typehints_format = "short"  # autoapi respects this
autodoc_typehints_description_target = "documented_params"  # autoapi respects this
python_use_unqualified_type_names = True
autodoc_class_signature = "mixed"
autoclass_content = "class"
autodoc_type_aliases = {
    "NDArray": "numpy.typing.NDArray",
    "np.bool_": "numpy.bool_",
    "np.float32": "numpy.float32",
    "np.float64": "numpy.float64",
    "np.floating": "numpy.floating",
    "np.integer": "numpy.integer",
    "np.intp": "numpy.intp",
    "np.str_": "numpy.str_",
    "np.uint8": "numpy.uint8",
}

autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_template_dir = "_templates/autoapi"
autoapi_keep_files = True
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "undoc-members",
]

from include_exclude import ignore_regex

autoapi_ignore = ignore_regex

# Related custom CSS
html_css_files = [
    GOOGLE_FONTS_URL,
    "css/label.css",
    "css/sphinx-togglebutton.css",
    "css/headings.css",
    "css/pub_card.css",
    "css/notebook.css",
    "css/rubric.css",
    "css/sidebar.css",
    "css/admonitions.css",
]
