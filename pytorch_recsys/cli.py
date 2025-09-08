"""Command-line interface for pytorch-recsys-framework."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from pytorch_recsys import __version__


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog="pytorch-recsys",
        description=(
            "PyTorch-based framework for sequential recommendation systems"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"pytorch-recsys-framework {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show framework information"
    )
    info_parser.set_defaults(func=_info_command)

    # Parse arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args)

    if hasattr(parsed_args, "func"):
        return int(parsed_args.func(parsed_args))
    else:
        parser.print_help()
        return 0


def _info_command(args: argparse.Namespace) -> int:
    """Show framework information."""
    print(f"PyTorch RecSys Framework v{__version__}")
    print(
        "A modular PyTorch-based framework for "
        "sequential recommendation systems"
    )
    print()

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch: Not installed")

    try:
        import pytorch_lightning

        print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")
    except ImportError:
        print("PyTorch Lightning: Not installed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
