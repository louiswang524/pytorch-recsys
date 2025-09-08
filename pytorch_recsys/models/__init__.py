"""Core recommendation models and architectures.

This module provides:
- Sequential recommendation models
- Transformer-based architectures
- Attention mechanism implementations
- Model utilities and wrappers
"""

from __future__ import annotations

from typing import List

# Import base classes and registry
from .base import BaseSequentialModel, ModelConfig
from .registry import ModelRegistry, ModelMetadata, create_model

# Import mock models (this will trigger registration)
from .mock import MockSequentialModel, SimpleMockModel

__all__: List[str] = [
    # Base interfaces
    "BaseSequentialModel",
    "ModelConfig",
    # Registry system
    "ModelRegistry",
    "ModelMetadata",
    "create_model",
    # Mock implementations
    "MockSequentialModel",
    "SimpleMockModel",
]
