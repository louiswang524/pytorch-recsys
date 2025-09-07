"""Test command-line interface."""

import pytest

from pytorch_recsys.cli import main


def test_cli_help():
    """Test CLI help command."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_version():
    """Test CLI version command."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_cli_info():
    """Test CLI info command."""
    result = main(["info"])
    assert result == 0


def test_cli_no_args():
    """Test CLI with no arguments."""
    result = main([])
    assert result == 0
