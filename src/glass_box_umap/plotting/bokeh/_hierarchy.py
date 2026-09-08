from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


@dataclass(frozen=True)
class HierarchyLevel:
    """One categorical cut through a hierarchy.

    ``labels`` contains a stable cluster identifier for every plotted sample.
    ``colors`` and ``metadata`` are keyed by those identifiers. Metadata may
    contain arbitrary values; the interactive plot recognizes ``size`` and
    ``top_families`` for its cluster key and hover display.
    """

    name: str
    labels: Sequence[Any] | NDArray
    colors: Mapping[str, str]
    metadata: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class HierarchySpec:
    """A named hierarchy whose ordered levels can be explored in the plot."""

    name: str
    levels: Sequence[HierarchyLevel]


def validate_hierarchies(
    hierarchies: Sequence[HierarchySpec] | None,
    n_samples: int,
) -> None:
    """Validate hierarchy names, levels, labels, colors, and metadata."""
    if hierarchies is None:
        return
    if not hierarchies:
        raise ValueError("hierarchies must contain at least one HierarchySpec.")

    source_names = [hierarchy.name for hierarchy in hierarchies]
    if any(not name for name in source_names):
        raise ValueError("hierarchy names must be non-empty strings.")
    if len(set(source_names)) != len(source_names):
        raise ValueError("hierarchy names must be unique.")

    for hierarchy in hierarchies:
        if not hierarchy.levels:
            raise ValueError(f"hierarchy {hierarchy.name!r} must contain at least one level.")
        level_names = [level.name for level in hierarchy.levels]
        if any(not name for name in level_names):
            raise ValueError(f"hierarchy {hierarchy.name!r} has an empty level name.")
        if len(set(level_names)) != len(level_names):
            raise ValueError(f"hierarchy {hierarchy.name!r} has duplicate level names.")

        for level in hierarchy.levels:
            if len(level.labels) != n_samples:
                raise ValueError(
                    f"hierarchy {hierarchy.name!r} level {level.name!r} has "
                    f"{len(level.labels)} labels, but expected {n_samples}."
                )
            labels = {str(value) for value in level.labels}
            missing_colors = labels - set(level.colors)
            if missing_colors:
                raise ValueError(
                    f"hierarchy {hierarchy.name!r} level {level.name!r} is missing "
                    f"colors for {sorted(missing_colors)[:5]}."
                )
            missing_metadata = labels - set(level.metadata)
            if missing_metadata:
                raise ValueError(
                    f"hierarchy {hierarchy.name!r} level {level.name!r} is missing "
                    f"metadata for {sorted(missing_metadata)[:5]}."
                )
