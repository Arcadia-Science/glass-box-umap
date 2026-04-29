import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, FactorRange, RadioButtonGroup
from bokeh.models.layouts import LayoutDOM
from bokeh.plotting import figure

from ._colors import BAR_COLOR_REDUCED, LABEL_COLOR, LABEL_FONT_SIZE
from ._data import BarViews, TopFeatures
from ._js import LINKED_BARS

_VIEW_LABELS = ("L2", "normed L2", "Dim 1", "Dim 2")


def build_bars(
    views: BarViews,
    top: TopFeatures,
    n_samples: int,
    scatter_source: ColumnDataSource,
    l2_source: ColumnDataSource,
) -> LayoutDOM:
    """Single bar chart with a view toggle on top.

    The chart shows the per-feature mean of contributions over the current
    scatter selection (or all samples if none selected). The ``normed L2`` view
    is computed JS-side on first selection (per-sample fractions of L2) and
    cached.

    ``l2_source`` is the ``c{k}``-keyed L2 view shared with the feature picker
    in :mod:`._controls` — a single ``ColumnDataSource`` shipped to the browser
    once.
    """
    contrib_sources = [
        l2_source,
        ColumnDataSource({f"c{k}": views.d0[:, k] for k in range(top.n_kept)}),
        ColumnDataSource({f"c{k}": views.d1[:, k] for k in range(top.n_kept)}),
    ]

    init_mean = views.l2.mean(axis=0)
    init_top = np.argsort(init_mean)[::-1][: top.display_k][::-1]
    init_feat = [top.kept_names[i] for i in init_top]

    bar_source = ColumnDataSource(
        data=dict(
            feature=init_feat,
            mean=[float(init_mean[i]) for i in init_top],
        )
    )

    p_bar = figure(
        sizing_mode="stretch_both",
        y_range=FactorRange(factors=init_feat),
        tools="",
        toolbar_location=None,
    )
    p_bar.hbar(
        y="feature",
        right="mean",
        height=0.8,
        source=bar_source,
        color=BAR_COLOR_REDUCED,
    )
    p_bar.xgrid.grid_line_color = None

    heading_div = Div(
        text=f"<b>Mean contribution — {_VIEW_LABELS[0]}</b>",
        styles={"color": LABEL_COLOR, "font-size": LABEL_FONT_SIZE},
    )
    view_widget = RadioButtonGroup(labels=list(_VIEW_LABELS), active=0)

    cb = CustomJS(
        args=dict(
            scatter_source=scatter_source,
            contrib_sources=contrib_sources,
            bar_source=bar_source,
            bar_range=p_bar.y_range,
            heading_div=heading_div,
            view_widget=view_widget,
            feature_names=top.kept_names,
            n_kept=top.n_kept,
            n_samples=n_samples,
            display_k=top.display_k,
            view_labels=list(_VIEW_LABELS),
            normed_l2_cache={"data": None},
        ),
        code=LINKED_BARS,
    )

    scatter_source.selected.js_on_change("indices", cb)
    view_widget.js_on_change("active", cb)

    return column(
        view_widget,
        heading_div,
        p_bar,
        sizing_mode="stretch_both",
        styles={
            "background-color": "white",
            "flex": "0 0 40%",
            "min-width": "0",
        },
    )
