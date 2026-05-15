from importlib.resources import files

_DIR = files(__package__)

LINKED_BARS = (_DIR / "linked_bars.js").read_text(encoding="utf-8")
NAMED_FILTER = (_DIR / "named_filter.js").read_text(encoding="utf-8")
COLOR_BY_MODE = (_DIR / "color_by_mode.js").read_text(encoding="utf-8")
FEATURE_PICKER = (_DIR / "feature_picker.js").read_text(encoding="utf-8")
TOP_N_SLIDER = (_DIR / "top_n_slider.js").read_text(encoding="utf-8")
