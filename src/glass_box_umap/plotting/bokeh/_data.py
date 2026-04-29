from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...jacobian import reduce_contributions

TOP_K_DISPLAY = 36


@dataclass(frozen=True)
class TopFeatures:
    """Result of ranking features by global L2 importance and selecting the top slice.

    Attributes:
        kept_names: Feature names for the kept pool, ordered by descending
            global L2 importance.
        keep_idx: Indices into the original feature axis, in the same order
            as ``kept_names``. Length equals ``n_kept``.
        display_k: Number of features to draw in the bar chart at any one
            time. Always ``<= n_kept``.
        reduced: The full ``(n_samples, n_features)`` L2-reduced contributions
            array — not sliced by ``keep_idx``.
        n_kept: (property) Size of the kept pool, i.e. ``len(kept_names)``.
    """

    kept_names: list[str]
    keep_idx: NDArray[np.integer]
    display_k: int
    reduced: NDArray[np.floating]

    @property
    def n_kept(self) -> int:
        return len(self.kept_names)


@dataclass(frozen=True)
class BarViews:
    """Three pre-computed ``(n_samples, n_kept)`` views of the kept-feature pool.

    Sign convention is per-attribute, not uniform: ``l2`` is non-negative
    (it's an L2 norm); ``d0`` and ``d1`` are signed (raw per-dimension
    contributions). The "normed L2" view shown in the bar chart is derived
    JS-side from ``l2`` as ``value / max(Σ_k value, ε)`` per sample (so each
    sample's row sums to 1 across the kept pool), cached after first
    computation. Note that the scatter's "Top feature" coloring is also
    decided from ``l2`` (argmax over the kept pool), so a sample's "top
    feature" reflects magnitude regardless of which view the user toggles
    in the bar chart.

    Attributes:
        l2: L2-reduced contributions, sliced to ``top.keep_idx``. Always non-negative.
        d0: Signed contributions to embedding dimension 0, sliced to ``top.keep_idx``.
        d1: Signed contributions to embedding dimension 1, sliced to ``top.keep_idx``.
    """

    l2: NDArray[np.floating]
    d0: NDArray[np.floating]
    d1: NDArray[np.floating]


def validate_shapes(
    Z: NDArray[np.floating],
    contributions: NDArray[np.floating],
    feature_names: list[str] | None = None,
    group_names: Sequence[Any] | NDArray | None = None,
    feature_values: NDArray[np.floating] | None = None,
) -> None:
    """Validate shape/length invariants shared by the public plot functions.

    Raises:
        ValueError: If ``Z`` is not ``(n_samples, 2)``, ``contributions`` is
            not ``(n_samples, 2, n_features)`` with ``n_features >= 1``, or
            ``feature_names`` / ``group_names`` / ``feature_values`` (when
            provided) don't match the corresponding axes.
    """
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Z must have shape (n_samples, 2); got {Z.shape}.")
    n_samples = Z.shape[0]

    if contributions.ndim != 3:
        raise ValueError(
            "contributions must have 3 dimensions (n_samples, 2, n_features); "
            f"got shape {contributions.shape}."
        )
    if contributions.shape[0] != n_samples:
        raise ValueError(
            f"contributions.shape[0] ({contributions.shape[0]}) must equal "
            f"Z.shape[0] ({n_samples})."
        )
    if contributions.shape[1] != 2:
        raise ValueError(f"contributions.shape[1] must be 2; got {contributions.shape[1]}.")
    if contributions.shape[2] < 1:
        raise ValueError("contributions must have at least one feature.")

    n_features = contributions.shape[2]
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has length {len(feature_names)}, but contributions "
            f"has {n_features} features."
        )

    if group_names is not None and len(group_names) != n_samples:
        raise ValueError(
            f"group_names has length {len(group_names)}, but Z has {n_samples} samples."
        )

    if feature_values is not None and feature_values.shape != (n_samples, n_features):
        raise ValueError(
            f"feature_values has shape {feature_values.shape}, but expected "
            f"({n_samples}, {n_features})."
        )


def select_top_features(
    contributions: NDArray[np.floating],
    feature_names: list[str] | None,
    top_k_global: int,
    top_k_display: int,
) -> TopFeatures:
    """Rank features by global L2 importance and select the top slice for plotting.

    Contributions are first collapsed across embedding dimensions via
    :func:`reduce_contributions` (L2 norm), producing a
    ``(n_samples, n_features)`` array. The per-feature scalar is the mean of
    that across samples.

    The top ``top_k_global`` features by this scalar are retained as the
    pool shipped to the browser; ``top_k_display`` (clipped to the pool
    size) is the number drawn in the bar chart at any one time.

    Args:
        contributions: Array of shape ``(n_samples, n_components, n_features)``.
        feature_names: Human-readable name per feature. When ``None``,
            synthetic ``"Feature {i}"`` names are generated.
        top_k_global: Target size of the kept-feature pool. Clipped to
            ``n_features``.
        top_k_display: Target number of features to display. Clipped to the
            kept pool size.

    Returns:
        A :class:`TopFeatures` with ``kept_names`` and ``keep_idx`` ordered
        by descending global L2 importance, ``display_k`` clipped to the
        kept pool size, and the full L2-reduced array in ``reduced``.
    """
    n_features = contributions.shape[-1]
    names = (
        feature_names if feature_names is not None else [f"Feature {i}" for i in range(n_features)]
    )

    reduced = reduce_contributions(contributions, "l2")
    global_importance = reduced.mean(axis=0)

    n_kept = min(top_k_global, n_features)
    if n_kept == n_features:
        keep_idx = np.argsort(-global_importance)
    else:
        partitioned = np.argpartition(-global_importance, n_kept - 1)[:n_kept]
        keep_idx = partitioned[np.argsort(-global_importance[partitioned])]
    kept_names = [names[i] for i in keep_idx]
    display_k = min(top_k_display, n_kept)
    return TopFeatures(
        kept_names=kept_names,
        keep_idx=keep_idx,
        display_k=display_k,
        reduced=reduced,
    )


def compute_bar_views(
    contributions: NDArray[np.floating],
    top: TopFeatures,
) -> BarViews:
    """Slice the three absolute bar views for the kept-feature pool."""
    return BarViews(
        l2=top.reduced[:, top.keep_idx].astype(np.float32),
        d0=contributions[:, 0, top.keep_idx].astype(np.float32),
        d1=contributions[:, 1, top.keep_idx].astype(np.float32),
    )


def precompute_top_features(
    kept_l2: NDArray[np.floating],
    kept_names: list[str],
) -> tuple[list[str], NDArray[np.integer], NDArray[np.integer]]:
    """Per-sample top kept feature, ranked by frequency.

    For each sample, the kept feature with the largest L2-reduced contribution
    is its "top feature". Distinct top features are then ranked by how often
    they win across the dataset (most common → rank 0). Restricting argmax to
    the kept pool guarantees every legend label has a corresponding bar in the
    bar chart.

    Args:
        kept_l2: L2-reduced contributions sliced to the kept pool, shape
            ``(n_samples, n_kept)``.
        kept_names: Feature names matching ``kept_l2``'s column axis.

    Returns:
        A ``(top_feature_names_by_rank, sample_rank, top_kept_idx)`` tuple:

        - ``top_feature_names_by_rank``: distinct kept-feature names ordered
          by descending frequency of being a sample's top feature.
        - ``sample_rank``: per-sample integer rank into
          ``top_feature_names_by_rank`` (always ``< len(top_feature_names_by_rank)``).
        - ``top_kept_idx``: per-sample column index into ``kept_l2`` of the
          sample's top feature (i.e. ``kept_l2.argmax(axis=1)``).
    """
    top_kept_idx = kept_l2.argmax(axis=1)
    unique, counts = np.unique(top_kept_idx, return_counts=True)
    rank_order = unique[np.argsort(counts)[::-1]]
    top_feature_names_by_rank = [kept_names[i] for i in rank_order]
    rank_of: dict[int, int] = {int(idx): r for r, idx in enumerate(rank_order)}
    sample_rank = np.array([rank_of[int(i)] for i in top_kept_idx], dtype=np.int32)
    return top_feature_names_by_rank, sample_rank, top_kept_idx
