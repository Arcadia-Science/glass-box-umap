BAR_COLOR_REDUCED = "#756bb1"
LABEL_COLOR = "#444444"
LABEL_FONT_SIZE = "13px"

DEGENERATE_RANGE_EPS = 1e-12
DEGENERATE_RANGE_FRAC = 0.05
DEGENERATE_RANGE_MIN_SPAN = 1e-6

_GLASBEY_CATEGORY10: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#3a0183",
    "#004301",
    "#0fffa9",
    "#5e0040",
    "#bcbcff",
    "#d8afa2",
    "#b80080",
    "#004e53",
    "#6b6500",
    "#7d0200",
    "#6126ff",
    "#ffff9a",
    "#574964",
    "#8cb894",
    "#94fcff",
    "#028268",
    "#91ff00",
    "#8300a0",
    "#ad8944",
    "#5b3400",
    "#ffc0f3",
    "#ff6f76",
]


def pick_palette(n: int) -> list[str]:
    base = _GLASBEY_CATEGORY10
    tiles = (n + len(base) - 1) // len(base)
    return (base * tiles)[:n]


def nondegenerate_range(lo: float, hi: float) -> tuple[float, float]:
    """Widen a degenerate (``hi - lo < DEGENERATE_RANGE_EPS``) range for ``LinearColorMapper``.

    When every sample has the same color value (e.g. a feature with zero
    variance), a mapper with ``low == high`` renders all points at ``high``
    and breaks tick generation on the color bar. This widens such ranges to a
    small symmetric span around the constant value, so points map to the
    mid-color and the color bar retains a legible axis.

    The same widening is applied JS-side in ``feature_picker.js`` (constants
    are passed in via CustomJS args from this module).
    """
    if hi - lo >= DEGENERATE_RANGE_EPS:
        return lo, hi

    mid = (lo + hi) / 2
    span = max(abs(mid) * DEGENERATE_RANGE_FRAC, DEGENERATE_RANGE_MIN_SPAN)
    return mid - span, mid + span
