"""PyTorch Recommendation Systems Framework.

A modular PyTorch-based framework for sequential recommendation systems
with transformer models and optimized attention mechanisms.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Louis Wang"
__email__ = "louiswang524@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Louis Wang"

# Import main modules when available
try:
    from pytorch_recsys import data
    from pytorch_recsys import models
    from pytorch_recsys import layers
    from pytorch_recsys import training
    from pytorch_recsys import evaluation
    from pytorch_recsys import serving
    from pytorch_recsys import configs
    from pytorch_recsys import utils
except ImportError:
    # Modules not yet implemented, this is expected during initial setup
    pass


# Version information
def get_version() -> str:
    """Get the current version of pytorch-recsys-framework."""
    return __version__


# Public API exports - will be populated as modules are implemented
__all__ = [
    "__version__",
    "get_version",
    # Module names will be added here as they are implemented
    # "data",
    # "models",
    # "layers",
    # "training",
    # "evaluation",
    # "serving",
    # "configs",
    # "utils",
]
