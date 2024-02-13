from importlib.metadata import version

import ndimsplinejax as pkg


def test_version() -> None:
    assert version("ndimsplinejax") == pkg.__version__


def test_all() -> None:
    """Test the `ndimsplinejax` package contents."""
    # Test detailed contents (not order)
    assert set(pkg.__all__) == {
        "__version__",
        "__version_tuple__",
        "SplineCoefs_from_GriddedData",
        "SplineInterpolant",
    }
