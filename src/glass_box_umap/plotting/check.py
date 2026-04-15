import importlib.metadata

from packaging.requirements import Requirement
from packaging.version import Version


def get_plotting_requirements() -> list[Requirement]:
    """Get the plotting extra requirements from package metadata."""
    all_reqs = [Requirement(r) for r in importlib.metadata.requires("glass-box-umap") or []]
    return [r for r in all_reqs if r.marker and r.marker.evaluate({"extra": "plotting"})]


def check_package(req: Requirement) -> None:
    """Raises ImportError if a required package is missing or the wrong version."""
    try:
        installed_version = Version(importlib.metadata.version(req.name))
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"glass_box_umap.plotting requires {req}, but it is not installed. "
            f"Install it with: pip install 'glass-box-umap[plotting]'"
        ) from None

    if installed_version not in req.specifier:
        raise ImportError(
            f"glass_box_umap.plotting requires {req}, "
            f"but {installed_version} is installed. "
            f"Upgrade with: pip install 'glass-box-umap[plotting]'"
        )
