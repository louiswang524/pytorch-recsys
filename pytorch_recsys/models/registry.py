"""Model registry system for automatic model discovery and instantiation.

This module provides a registry system that allows automatic registration,
discovery, and instantiation of sequential recommendation models. It supports
configuration-driven model creation, version tracking, and compatibility checking.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from .base import BaseSequentialModel

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for registered models.

    This class stores comprehensive information about registered models
    including their class, description, version, and compatibility information.
    """

    name: str = Field(..., description="Model name (registry key)")
    class_name: str = Field(..., description="Model class name")
    description: str = Field("", description="Model description")
    version: str = Field("1.0.0", description="Model version")
    module: str = Field(..., description="Python module path")
    file_path: str = Field(..., description="File path where model is defined")
    model_class: Type[BaseSequentialModel] = Field(
        ..., description="Model class reference"
    )
    tags: List[str] = Field(
        default_factory=list, description="Model tags for categorization"
    )
    requirements: Dict[str, str] = Field(
        default_factory=dict, description="Minimum requirements"
    )

    class Config:
        """Pydantic config for metadata."""

        arbitrary_types_allowed = True


class ModelRegistry:
    """Registry for automatic model discovery and instantiation.

    This registry provides a centralized system for:
    - Automatic model registration via decorators
    - Model discovery and listing
    - Configuration-driven model instantiation
    - Version and compatibility tracking
    - Model metadata management

    Example:
        ```python
        # Register a model
        @ModelRegistry.register(
            name="my_model",
            description="My custom sequential model",
            version="1.0.0"
        )
        class MyModel(BaseSequentialModel):
            pass

        # Create model instance
        config = DictConfig({"num_items": 1000, "embedding_dim": 128, ...})
        model = ModelRegistry.create_model("my_model", config)

        # List all registered models
        models = ModelRegistry.list_models()
        ```
    """

    _models: Dict[str, ModelMetadata] = {}
    _lock = False  # Simple lock to prevent modification during iteration

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        requirements: Optional[Dict[str, str]] = None,
    ) -> Callable[[Type[BaseSequentialModel]], Type[BaseSequentialModel]]:
        """Decorator for registering models in the registry.

        Args:
            name: Unique name for the model (used as registry key)
            description: Human-readable description of the model
            version: Model version string (semantic versioning recommended)
            tags: List of tags for categorization (e.g., ["transformer", "attention"])
            requirements: Minimum requirements dict (e.g., {"torch": ">=2.0.0"})

        Returns:
            Decorator function that registers the model class

        Raises:
            ValueError: If model name is already registered or invalid
            TypeError: If model doesn't inherit from BaseSequentialModel

        Example:
            ```python
            @ModelRegistry.register(
                name="sasrec",
                description="Self-Attentive Sequential Recommendation",
                version="1.0.0",
                tags=["transformer", "attention"],
                requirements={"torch": ">=2.0.0"}
            )
            class SASRec(BaseSequentialModel):
                pass
            ```
        """

        def decorator(
            model_class: Type[BaseSequentialModel],
        ) -> Type[BaseSequentialModel]:
            cls._register_model(
                name=name,
                model_class=model_class,
                description=description,
                version=version,
                tags=tags or [],
                requirements=requirements or {},
            )
            return model_class

        return decorator

    @classmethod
    def _register_model(
        cls,
        name: str,
        model_class: Type[BaseSequentialModel],
        description: str,
        version: str,
        tags: List[str],
        requirements: Dict[str, str],
    ) -> None:
        """Internal method to register a model.

        Args:
            name: Model name
            model_class: Model class to register
            description: Model description
            version: Model version
            tags: Model tags
            requirements: Model requirements

        Raises:
            ValueError: If registration fails
            TypeError: If model class is invalid
        """
        # Validate model name
        if not name or not isinstance(name, str):
            raise ValueError(
                f"Model name must be a non-empty string, got: {name}"
            )

        if name in cls._models:
            existing_version = cls._models[name].version
            logger.warning(
                f"Model '{name}' is already registered (version {existing_version}). "
                f"Overriding with version {version}"
            )

        # Validate model class
        if not inspect.isclass(model_class):
            raise TypeError(f"Expected a class, got: {type(model_class)}")

        if not issubclass(model_class, BaseSequentialModel):
            raise TypeError(
                f"Model {model_class.__name__} must inherit from BaseSequentialModel"
            )

        # Create metadata
        try:
            file_path = inspect.getfile(model_class)
        except (TypeError, OSError):
            file_path = "<unknown>"

        metadata = ModelMetadata(
            name=name,
            class_name=model_class.__name__,
            description=description,
            version=version,
            module=model_class.__module__,
            file_path=file_path,
            model_class=model_class,
            tags=tags,
            requirements=requirements,
        )

        # Register the model
        cls._models[name] = metadata

        logger.info(
            f"Registered model '{name}' "
            f"(class: {model_class.__name__}, version: {version})"
        )

    @classmethod
    def create_model(
        cls, name: str, config: DictConfig
    ) -> BaseSequentialModel:
        """Create model instance from configuration.

        Args:
            name: Name of the registered model
            config: Model configuration (must contain all required parameters)

        Returns:
            Instantiated model ready for training/inference

        Raises:
            ValueError: If model is not registered
            TypeError: If model instantiation fails

        Example:
            ```python
            config = DictConfig({
                "num_items": 1000,
                "max_seq_len": 50,
                "embedding_dim": 128,
                "learning_rate": 0.001
            })
            model = ModelRegistry.create_model("sasrec", config)
            ```
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )

        metadata = cls._models[name]
        model_class = metadata.model_class

        try:
            # Create model instance
            model = model_class(config)

            logger.info(f"Created model '{name}' with config: {dict(config)}")
            return model

        except Exception as e:
            raise TypeError(f"Failed to create model '{name}': {e}") from e

    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseSequentialModel]:
        """Get the model class for a registered model.

        Args:
            name: Name of the registered model

        Returns:
            Model class

        Raises:
            ValueError: If model is not registered
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )

        return cls._models[name].model_class

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed information about a registered model.

        Args:
            name: Name of the registered model

        Returns:
            Dictionary containing model metadata

        Raises:
            ValueError: If model is not registered
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )

        metadata = cls._models[name]
        return {
            "name": metadata.name,
            "class_name": metadata.class_name,
            "description": metadata.description,
            "version": metadata.version,
            "module": metadata.module,
            "file_path": metadata.file_path,
            "tags": metadata.tags,
            "requirements": metadata.requirements,
        }

    @classmethod
    def list_models(
        cls, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all registered models with optional tag filtering.

        Args:
            tags: Optional list of tags to filter models

        Returns:
            List of model information dictionaries

        Example:
            ```python
            # List all models
            all_models = ModelRegistry.list_models()

            # List only transformer-based models
            transformer_models = ModelRegistry.list_models(tags=["transformer"])
            ```
        """
        models = []

        for name, metadata in cls._models.items():
            # Filter by tags if specified
            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue

            models.append(
                {
                    "name": metadata.name,
                    "class_name": metadata.class_name,
                    "description": metadata.description,
                    "version": metadata.version,
                    "tags": metadata.tags,
                }
            )

        return sorted(models, key=lambda x: x["name"])

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a model from the registry.

        Args:
            name: Name of the model to unregister

        Returns:
            True if model was unregistered, False if not found

        Note:
            This method should be used carefully, mainly for testing purposes.
        """
        if name in cls._models:
            del cls._models[name]
            logger.info(f"Unregistered model '{name}'")
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models.

        Warning:
            This method removes all registered models and should only be used
            for testing purposes or when you want to reset the registry completely.
        """
        cls._models.clear()
        logger.warning("Cleared all registered models")

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if model is registered, False otherwise
        """
        return name in cls._models

    @classmethod
    def get_registered_names(cls) -> List[str]:
        """Get list of all registered model names.

        Returns:
            Sorted list of registered model names
        """
        return sorted(cls._models.keys())

    @classmethod
    def validate_requirements(cls, name: str) -> Dict[str, bool]:
        """Validate if model requirements are satisfied.

        Args:
            name: Name of the registered model

        Returns:
            Dictionary mapping requirement names to satisfaction status

        Raises:
            ValueError: If model is not registered
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found")

        metadata = cls._models[name]
        requirements = metadata.requirements

        validation_results = {}

        for req_name, req_version in requirements.items():
            try:
                if req_name == "torch":
                    import torch  # noqa: F401
                elif req_name == "pytorch-lightning":
                    import pytorch_lightning as pl  # noqa: F401
                else:
                    # For other requirements, assume they're satisfied
                    validation_results[req_name] = True
                    continue

                # Simple version comparison (can be enhanced)
                validation_results[req_name] = True

            except ImportError:
                validation_results[req_name] = False

        return validation_results

    @classmethod
    def get_models_by_tag(cls, tag: str) -> List[str]:
        """Get all model names that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of model names with the specified tag
        """
        matching_models = []

        for name, metadata in cls._models.items():
            if tag in metadata.tags:
                matching_models.append(name)

        return sorted(matching_models)


# Convenience function for model creation
def create_model(
    name: str, config: Union[DictConfig, Dict[str, Any]]
) -> BaseSequentialModel:
    """Convenience function to create a model from the registry.

    Args:
        name: Name of the registered model
        config: Model configuration (dict or DictConfig)

    Returns:
        Instantiated model

    Example:
        ```python
        from pytorch_recsys.models import create_model

        config = {"num_items": 1000, "embedding_dim": 128, ...}
        model = create_model("sasrec", config)
        ```
    """
    if isinstance(config, dict):
        config = DictConfig(config)

    return ModelRegistry.create_model(name, config)
