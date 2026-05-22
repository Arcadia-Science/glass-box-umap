import re

import glass_box_umap
import pytest


def test_version_is_pep440_string():
    version = glass_box_umap.__version__
    assert isinstance(version, str)
    assert re.match(r"^\d+\.\d+", version)


def test_unknown_module_attribute_raises():
    with pytest.raises(AttributeError):
        glass_box_umap.this_attribute_does_not_exist  # noqa: B018
