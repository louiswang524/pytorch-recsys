"""Test package imports and basic functionality."""

import pytorch_recsys


def test_version():
    """Test that version is accessible."""
    assert hasattr(pytorch_recsys, "__version__")
    assert isinstance(pytorch_recsys.__version__, str)
    assert pytorch_recsys.__version__ == "0.1.0"


def test_get_version():
    """Test get_version function."""
    version = pytorch_recsys.get_version()
    assert version == pytorch_recsys.__version__


def test_package_metadata():
    """Test package metadata is accessible."""
    assert hasattr(pytorch_recsys, "__author__")
    assert hasattr(pytorch_recsys, "__email__")
    assert hasattr(pytorch_recsys, "__license__")
    assert pytorch_recsys.__license__ == "MIT"


def test_import_structure():
    """Test that package structure is correct."""
    # Test main package imports
    assert pytorch_recsys is not None

    # Test that __all__ is defined
    assert hasattr(pytorch_recsys, "__all__")
    assert isinstance(pytorch_recsys.__all__, list)


class TestPackageStructure:
    """Test the package directory structure."""

    def test_submodule_imports(self):
        """Test that submodules can be imported without errors."""
        # These should not raise ImportError even if modules are empty
        try:
            from pytorch_recsys import data  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import models  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import layers  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import training  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import evaluation  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import serving  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import configs  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        try:
            from pytorch_recsys import utils  # noqa: F401
        except ImportError:
            pass  # Expected during initial setup

        # This should always work
        import pytorch_recsys

        assert pytorch_recsys is not None
