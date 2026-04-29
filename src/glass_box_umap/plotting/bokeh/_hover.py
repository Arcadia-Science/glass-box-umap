import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_HOVER_IMAGE_LONGEST_SIDE = 64
_RESERVED_HOVER_KEYS = frozenset(
    {
        "x",
        "y",
        "index",
        "group",
        "color_value",
        "top_feature_group",
        "top_feature_name",
        "top_data_value",
        "picker_data_value",
        "sample_rank",
        "__hover_image",
    }
)


@dataclass(frozen=True)
class HoverTooltips:
    """Per-color-mode tooltip HTML used by the scatter's HoverTools.

    Each glyph set in the scatter is paired with one of these templates so the
    tooltip surfaces info relevant to the current "Color by" mode. When the
    user passes ``hover_tooltips``, all three fields hold the same override
    string.

    Attributes:
        group: Tooltip used by the per-group glyphs ("Color by" → Group).
        feature: Tooltip used by the gradient glyph ("Color by" → Feature).
        top: Tooltip used by the top-feature glyphs ("Color by" → Top feature).
    """

    group: str
    feature: str
    top: str


def _to_hover_uri(arr: NDArray[np.uint8]) -> str:
    """Encode an image array as a base64 PNG data URI.

    The image is resized so its longest side is
    ``_HOVER_IMAGE_LONGEST_SIDE`` pixels, preserving aspect ratio.

    Args:
        arr: Uint8 array of shape (H, W) for grayscale or (H, W, 3/4) for
            RGB(A).

    Returns:
        A ``data:image/png;base64,...`` URI suitable for use as the ``src``
        of an ``<img>`` tag inside a Bokeh ``HoverTool`` tooltip.
    """
    img = Image.fromarray(arr)
    w, h = img.size
    scale = _HOVER_IMAGE_LONGEST_SIDE / max(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def resolve_hover(
    default_bodies: HoverTooltips,
    hover_images: NDArray[np.uint8] | None,
    hover_tooltips: str | None,
    hover_data: Mapping[str, Sequence[Any]] | None,
) -> tuple[HoverTooltips, dict[str, NDArray[Any]]]:
    """Resolve hover customization into per-mode tooltip templates and CDS extras.

    Three modes, determined by which kwargs are set:

    - None set: each ``default_bodies`` field is wrapped in ``<div>...</div>``.
    - ``hover_images`` set: each default body is wrapped in a ``<div>`` with
      a PNG-encoded image prefix.
    - ``hover_tooltips`` and/or ``hover_data`` set: user-supplied template
      fully replaces the defaults (same template for all three modes); user
      columns are merged into the scatter ``ColumnDataSource``.

    Args:
        default_bodies: Per-mode HTML body fragments (no outer ``<div>``)
            representing the plot's built-in tooltip bodies — used when no
            override is passed and as the suffix when ``hover_images`` is set.
        hover_images: Uint8 array of shape (n_samples, H, W) or
            (n_samples, H, W, 3/4). Mutually exclusive with the other two.
        hover_tooltips: Tooltip HTML that fully replaces the defaults across
            all modes.
        hover_data: Extra CDS columns referenced by ``@field`` in
            ``hover_tooltips``. Keys must not collide with reserved columns.

    Raises:
        ValueError: If ``hover_images`` is combined with the other two, or if
            ``hover_data`` contains a reserved key.
    """
    if hover_images is not None and (hover_tooltips is not None or hover_data is not None):
        raise ValueError("Pass either hover_images or hover_tooltips/hover_data, not both.")

    if hover_images is not None:
        uris = [_to_hover_uri(img) for img in hover_images]
        img_prefix = "<img src='@__hover_image' style='display:block; margin-bottom:4px'/>"
        tooltips = HoverTooltips(
            group=f"<div>{img_prefix}{default_bodies.group}</div>",
            feature=f"<div>{img_prefix}{default_bodies.feature}</div>",
            top=f"<div>{img_prefix}{default_bodies.top}</div>",
        )
        return tooltips, {"__hover_image": np.asarray(uris, dtype=object)}

    extras: dict[str, NDArray[Any]] = {}
    if hover_data is not None:
        collisions = set(hover_data) & _RESERVED_HOVER_KEYS
        if collisions:
            raise ValueError(
                f"hover_data keys collide with reserved CDS columns: {sorted(collisions)}"
            )
        extras = {k: np.asarray(v) for k, v in hover_data.items()}

    if hover_tooltips is not None:
        tooltips = HoverTooltips(group=hover_tooltips, feature=hover_tooltips, top=hover_tooltips)
    else:
        tooltips = HoverTooltips(
            group=f"<div>{default_bodies.group}</div>",
            feature=f"<div>{default_bodies.feature}</div>",
            top=f"<div>{default_bodies.top}</div>",
        )
    return tooltips, extras
