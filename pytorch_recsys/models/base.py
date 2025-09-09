"""Base model interface for sequential recommendation systems.

This module provides the abstract base class that all sequential recommendation
models must inherit from, ensuring consistent interfaces for training,
validation, prediction, and inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, Field, validator
from torch import Tensor, nn


class ModelConfig(BaseModel):
    """Base configuration schema for sequential recommendation models.

    This Pydantic model provides validation for essential model parameters
    required by all sequential recommendation models.
    """

    # Core model parameters
    num_items: int = Field(
        ..., gt=0, description="Number of items in the catalog"
    )
    num_users: Optional[int] = Field(
        None, gt=0, description="Number of users (optional)"
    )
    max_seq_len: int = Field(
        ..., gt=0, le=10000, description="Maximum sequence length"
    )
    embedding_dim: int = Field(..., gt=0, description="Embedding dimension")

    # Training parameters
    learning_rate: float = Field(
        0.001, gt=0, le=1.0, description="Learning rate"
    )
    weight_decay: float = Field(
        0.0, ge=0, le=1.0, description="L2 regularization weight"
    )

    # Optional scheduler configuration
    scheduler: Optional[Dict[str, Any]] = Field(
        None, description="Learning rate scheduler config"
    )

    # Loss function parameters
    loss_type: str = Field("cross_entropy", description="Loss function type")
    label_smoothing: float = Field(
        0.0, ge=0, le=1.0, description="Label smoothing factor"
    )

    @validator("max_seq_len")
    def validate_max_seq_len(cls, v):
        if v <= 0:
            raise ValueError("max_seq_len must be positive")
        if v > 10000:
            raise ValueError(
                "max_seq_len cannot exceed 10000 for memory efficiency"
            )
        return v

    @validator("embedding_dim")
    def validate_embedding_dim(cls, v):
        if v % 8 != 0:
            raise ValueError(
                "embedding_dim should be divisible by 8 for optimal performance"
            )
        return v

    class Config:
        """Pydantic config for the model configuration."""

        extra = "allow"  # Allow additional fields for model-specific configs


class BaseSequentialModel(pl.LightningModule, ABC):
    """Abstract base class for sequential recommendation models.

    This class defines the standard interface that all sequential recommendation
    models must implement, including:
    - Forward pass with standardized input/output
    - Prediction and inference methods
    - Training and validation steps
    - Configuration validation and management
    - Model serialization/deserialization

    All concrete models must inherit from this class and implement the abstract methods.

    Args:
        config: Model configuration containing all hyperparameters

    Example:
        ```python
        class MyModel(BaseSequentialModel):
            def forward(self, sequences: Tensor, **kwargs) -> Tensor:
                # Implement forward pass
                pass

            def predict_next_items(self, sequences: Tensor, k: int = 10) -> Tensor:
                # Implement prediction logic
                pass

            def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
                # Implement loss computation
                pass
        ```
    """

    def __init__(self, config: DictConfig):
        """Initialize the base sequential model.

        Args:
            config: Model configuration with all required parameters

        Raises:
            ValueError: If configuration validation fails
        """
        super().__init__()

        # Store and validate configuration
        self.config = config
        self.save_hyperparameters()

        # Validate configuration using Pydantic
        self._validate_config()

        # Extract common parameters
        self.num_items = config.num_items
        self.num_users = config.get("num_users", None)
        self.max_seq_len = config.max_seq_len
        self.embedding_dim = config.embedding_dim

        # Initialize common components
        self._init_embeddings()
        self._init_loss_function()

    def _validate_config(self) -> None:
        """Validate the model configuration using Pydantic.

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Convert OmegaConf to dict for Pydantic validation
            config_dict = dict(self.config)
            ModelConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid model configuration: {e}")

    def _init_embeddings(self) -> None:
        """Initialize item embeddings with padding token."""
        self.item_embeddings = nn.Embedding(
            self.num_items + 1,  # +1 for padding token (index 0)
            self.embedding_dim,
            padding_idx=0,
        )

        # Initialize user embeddings if specified
        if self.num_users is not None:
            self.user_embeddings = nn.Embedding(
                self.num_users + 1,  # +1 for padding token
                self.embedding_dim,
                padding_idx=0,
            )

    def _init_loss_function(self) -> None:
        """Initialize the loss function based on configuration."""
        loss_type = self.config.get("loss_type", "cross_entropy")
        label_smoothing = self.config.get("label_smoothing", 0.0)

        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=0,
                label_smoothing=label_smoothing,  # Ignore padding tokens
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @abstractmethod
    def forward(self, sequences: Tensor, **kwargs) -> Tensor:
        """Forward pass of the model.

        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            **kwargs: Additional model-specific arguments

        Returns:
            Logits tensor of shape (batch_size, seq_len, num_items) or
            (batch_size, num_items) depending on the model
        """
        pass

    @abstractmethod
    def predict_next_items(self, sequences: Tensor, k: int = 10) -> Tensor:
        """Generate top-k item predictions for given sequences.

        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            k: Number of items to predict

        Returns:
            Tensor of shape (batch_size, k) containing top-k item IDs
        """
        pass

    @abstractmethod
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the training loss.

        Args:
            logits: Model predictions of shape (batch_size, seq_len, num_items)
            targets: Target sequences of shape (batch_size, seq_len)

        Returns:
            Scalar loss tensor
        """
        pass

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """Standard training step implementation.

        Args:
            batch: Batch dictionary containing 'sequences' and 'targets'
            batch_idx: Batch index

        Returns:
            Training loss tensor
        """
        sequences = batch["sequences"]
        targets = batch["targets"]

        # Forward pass
        logits = self(sequences)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Log metrics
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        """Standard validation step implementation.

        Args:
            batch: Batch dictionary containing 'sequences' and 'targets'
            batch_idx: Batch index

        Returns:
            Validation loss tensor
        """
        sequences = batch["sequences"]
        targets = batch["targets"]

        # Forward pass
        logits = self(sequences)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Standard test step implementation.

        Args:
            batch: Batch dictionary containing 'sequences' and 'targets'
            batch_idx: Batch index

        Returns:
            Dictionary containing test metrics
        """
        sequences = batch["sequences"]
        targets = batch["targets"]

        # Forward pass
        logits = self(sequences)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Generate predictions for evaluation
        predictions = self.predict_next_items(sequences, k=10)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return {
            "test_loss": loss,
            "predictions": predictions,
            "targets": targets,
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer or dictionary with optimizer and scheduler configuration
        """
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.get("weight_decay", 0.0),
        )

        # Configure scheduler if specified
        scheduler_config = self.config.get("scheduler", None)
        if scheduler_config:
            scheduler = self._create_scheduler(optimizer, scheduler_config)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler from configuration.

        Args:
            optimizer: PyTorch optimizer
            config: Scheduler configuration dictionary

        Returns:
            Learning rate scheduler instance

        Raises:
            ValueError: If scheduler type is not supported
        """
        scheduler_type = config.get("type", "step")

        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 30),
                gamma=config.get("gamma", 0.1),
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("T_max", 100),
                eta_min=config.get("eta_min", 0),
            )
        elif scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("mode", "min"),
                factor=config.get("factor", 0.5),
                patience=config.get("patience", 10),
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def predict(
        self, sequences: Tensor, k: int = 10, return_scores: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Generate predictions for given sequences.

        This method puts the model in evaluation mode and generates predictions.

        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            k: Number of items to predict
            return_scores: Whether to return prediction scores

        Returns:
            Top-k item IDs or tuple of (item_ids, scores) if return_scores=True
        """
        self.eval()
        with torch.no_grad():
            if return_scores:
                logits = self(sequences)
                scores, item_ids = torch.topk(logits, k, dim=-1)
                return item_ids, scores
            else:
                return self.predict_next_items(sequences, k)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.

        Returns:
            Dictionary containing model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "model_name": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": dict(self.config),
            "device": str(next(self.parameters()).device),
            "dtype": str(next(self.parameters()).dtype),
        }

    def save_model(self, path: str) -> None:
        """Save model checkpoint with configuration.

        Args:
            path: Path to save the model checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": dict(self.config),
            "model_class": self.__class__.__name__,
            "model_info": self.get_model_info(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_model(
        cls, path: str, config: Optional[DictConfig] = None
    ) -> "BaseSequentialModel":
        """Load model from checkpoint.

        Args:
            path: Path to the model checkpoint
            config: Optional config override

        Returns:
            Loaded model instance

        Raises:
            ValueError: If checkpoint is invalid or model class mismatch
        """
        checkpoint = torch.load(path, map_location="cpu")

        if config is None:
            config = DictConfig(checkpoint["config"])

        # Create model instance
        model = cls(config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return model
