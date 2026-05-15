import importlib.metadata
from unittest.mock import patch

import pytest
from glass_box_umap.plotting.check import check_package, get_plotting_requirements
from packaging.requirements import Requirement


def test_get_plotting_requirements():
    reqs = get_plotting_requirements()
    names = [r.name for r in reqs]
    assert "matplotlib" in names
    assert "adjusttext" in names


def test_check_package_missing_raises_import_error():
    req = Requirement("matplotlib>=3.9")
    with patch(
        "glass_box_umap.plotting.check.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("matplotlib"),
    ):
        with pytest.raises(ImportError, match="not installed"):
            check_package(req)


def test_check_package_too_old_raises_import_error():
    req = Requirement("matplotlib>=3.9")
    with patch(
        "glass_box_umap.plotting.check.importlib.metadata.version",
        return_value="3.8.0",
    ):
        with pytest.raises(ImportError, match="3.8.0 is installed"):
            check_package(req)


def test_check_package_too_new_raises_import_error():
    req = Requirement("matplotlib>=3.9,<4.0")
    with patch(
        "glass_box_umap.plotting.check.importlib.metadata.version",
        return_value="11.4.0",
    ):
        with pytest.raises(ImportError, match="11.4.0 is installed"):
            check_package(req)


def test_check_package_valid_passes():
    req = Requirement("matplotlib>=3.9")
    check_package(req)
