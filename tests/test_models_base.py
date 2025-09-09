"""Unit tests for the base model interface.

This module contains comprehensive tests for the BaseSequentialModel abstract
base class and its configuration validation system.
"""

from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig

from pytorch_recsys.models.base import BaseSequentialModel, ModelConfig
from pytorch_recsys.models.mock import MockSequentialModel, SimpleMockModel


class TestModelConfig:
    """Test cases for ModelConfig Pydantic validation."""

    def test_valid_config(self):
        """Test that valid configuration passes validation."""
        config_dict = {
            "num_items": 1000,
            "max_seq_len": 50,
            "embedding_dim": 128,
            "learning_rate": 0.001,
        }

        config = ModelConfig(**config_dict)

        assert config.num_items == 1000
        assert config.max_seq_len == 50
        assert config.embedding_dim == 128
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0  # default value

    def test_invalid_num_items(self):
        """Test that invalid num_items raises validation error."""
        with pytest.raises(
            ValueError, match="ensure this value is greater than 0"
        ):
            ModelConfig(num_items=0, max_seq_len=50, embedding_dim=128)

    def test_invalid_max_seq_len(self):
        """Test that invalid max_seq_len raises validation error."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            ModelConfig(num_items=1000, max_seq_len=-1, embedding_dim=128)

    def test_max_seq_len_too_large(self):
        """Test that max_seq_len exceeding limit raises validation error."""
        with pytest.raises(
            ValueError, match="max_seq_len cannot exceed 10000"
        ):
            ModelConfig(num_items=1000, max_seq_len=20000, embedding_dim=128)

    def test_embedding_dim_not_divisible_by_8(self):
        """Test warning when embedding_dim is not divisible by 8."""
        with pytest.raises(
            ValueError, match="embedding_dim should be divisible by 8"
        ):
            ModelConfig(num_items=1000, max_seq_len=50, embedding_dim=127)

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises validation error."""
        with pytest.raises(
            ValueError, match="ensure this value is less than or equal to 1.0"
        ):
            ModelConfig(
                num_items=1000,
                max_seq_len=50,
                embedding_dim=128,
                learning_rate=2.0,
            )

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        config = ModelConfig(
            num_items=1000,
            max_seq_len=50,
            embedding_dim=128,
            num_users=5000,
            weight_decay=0.01,
            scheduler={"type": "step", "step_size": 10},
        )

        assert config.num_users == 5000
        assert config.weight_decay == 0.01
        assert config.scheduler["type"] == "step"


class TestBaseSequentialModel:
    """Test cases for BaseSequentialModel abstract base class."""

    @pytest.fixture
    def valid_config(self):
        """Fixture providing a valid model configuration."""
        return DictConfig(
            {
                "num_items": 1000,
                "max_seq_len": 50,
                "embedding_dim": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            }
        )

    @pytest.fixture
    def mock_model(self, valid_config):
        """Fixture providing a mock model instance."""
        return MockSequentialModel(valid_config)

    def test_cannot_instantiate_abstract_class(self, valid_config):
        """Test that BaseSequentialModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSequentialModel(valid_config)

    def test_model_initialization(self, mock_model, valid_config):
        """Test that mock model initializes correctly."""
        assert mock_model.num_items == valid_config.num_items
        assert mock_model.max_seq_len == valid_config.max_seq_len
        assert mock_model.embedding_dim == valid_config.embedding_dim

        # Check that embeddings are initialized
        assert hasattr(mock_model, "item_embeddings")
        assert (
            mock_model.item_embeddings.num_embeddings
            == valid_config.num_items + 1
        )
        assert (
            mock_model.item_embeddings.embedding_dim
            == valid_config.embedding_dim
        )

    def test_config_validation_failure(self):
        """Test that invalid configuration raises ValueError."""
        invalid_config = DictConfig(
            {
                "num_items": -1,
                "max_seq_len": 50,
                "embedding_dim": 128,
            }  # Invalid
        )

        with pytest.raises(ValueError, match="Invalid model configuration"):
            MockSequentialModel(invalid_config)

    def test_forward_pass(self, mock_model):
        """Test forward pass with valid input."""
        batch_size, seq_len = 4, 20
        sequences = torch.randint(
            1, mock_model.num_items + 1, (batch_size, seq_len)
        )

        logits = mock_model(sequences)

        assert logits.shape == (batch_size, seq_len, mock_model.num_items + 1)
        assert not torch.isnan(logits).any()

    def test_forward_pass_with_padding(self, mock_model):
        """Test forward pass with padding tokens."""
        batch_size, seq_len = 4, 20
        sequences = torch.randint(
            1, mock_model.num_items + 1, (batch_size, seq_len)
        )

        # Add padding tokens (0) to some positions
        sequences[:, -5:] = 0

        logits = mock_model(sequences)

        assert logits.shape == (batch_size, seq_len, mock_model.num_items + 1)
        assert not torch.isnan(logits).any()

    def test_predict_next_items(self, mock_model):
        """Test prediction generation."""
        batch_size, seq_len = 4, 20
        sequences = torch.randint(
            1, mock_model.num_items + 1, (batch_size, seq_len)
        )

        k = 10
        predictions = mock_model.predict_next_items(sequences, k=k)

        assert predictions.shape == (batch_size, k)
        assert (predictions >= 1).all()  # Should not predict padding token
        assert (predictions <= mock_model.num_items).all()

    def test_compute_loss(self, mock_model):
        """Test loss computation."""
        batch_size, seq_len = 4, 20
        sequences = torch.randint(
            1, mock_model.num_items + 1, (batch_size, seq_len)
        )
        targets = torch.randint(
            1, mock_model.num_items + 1, (batch_size, seq_len)
        )

        logits = mock_model(sequences)
        loss = mock_model.compute_loss(logits, targets)

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)

    def test_training_step(self, mock_model):
        """Test training step implementation."""
        batch = {
            "sequences": torch.randint(1, mock_model.num_items + 1, (4, 20)),
            "targets": torch.randint(1, mock_model.num_items + 1, (4, 20)),
        }

        with patch.object(mock_model, "log") as mock_log:
            loss = mock_model.training_step(batch, 0)

            assert loss.dim() == 0  # Scalar loss
            assert loss.item() >= 0
            mock_log.assert_called_once()

    def test_validation_step(self, mock_model):
        """Test validation step implementation."""
        batch = {
            "sequences": torch.randint(1, mock_model.num_items + 1, (4, 20)),
            "targets": torch.randint(1, mock_model.num_items + 1, (4, 20)),
        }

        with patch.object(mock_model, "log") as mock_log:
            loss = mock_model.validation_step(batch, 0)

            assert loss.dim() == 0  # Scalar loss
            assert loss.item() >= 0
            mock_log.assert_called_once()

    def test_test_step(self, mock_model):
        """Test test step implementation."""
        batch = {
            "sequences": torch.randint(1, mock_model.num_items + 1, (4, 20)),
            "targets": torch.randint(1, mock_model.num_items + 1, (4, 20)),
        }

        with patch.object(mock_model, "log") as mock_log:
            result = mock_model.test_step(batch, 0)

            assert "test_loss" in result
            assert "predictions" in result
            assert "targets" in result
            assert result["test_loss"].item() >= 0
            mock_log.assert_called_once()

    def test_configure_optimizers_without_scheduler(self, mock_model):
        """Test optimizer configuration without scheduler."""
        optimizer = mock_model.configure_optimizers()

        assert hasattr(optimizer, "param_groups")
        assert (
            optimizer.param_groups[0]["lr"] == mock_model.config.learning_rate
        )
        assert (
            optimizer.param_groups[0]["weight_decay"]
            == mock_model.config.weight_decay
        )

    def test_configure_optimizers_with_scheduler(self, valid_config):
        """Test optimizer configuration with scheduler."""
        valid_config.scheduler = {
            "type": "step",
            "step_size": 10,
            "gamma": 0.5,
        }

        model = MockSequentialModel(valid_config)
        config = model.configure_optimizers()

        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["monitor"] == "val_loss"

    def test_predict_method(self, mock_model):
        """Test predict method with return_scores option."""
        sequences = torch.randint(1, mock_model.num_items + 1, (2, 10))

        # Test without scores
        predictions = mock_model.predict(sequences, k=5)
        assert predictions.shape == (2, 5)

        # Test with scores
        predictions, scores = mock_model.predict(
            sequences, k=5, return_scores=True
        )
        assert predictions.shape == (2, 5)
        assert scores.shape == (2, 5)

    def test_get_model_info(self, mock_model):
        """Test model information retrieval."""
        info = mock_model.get_model_info()

        assert "model_name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "config" in info
        assert info["model_name"] == "MockSequentialModel"
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0

    def test_save_and_load_model(self, mock_model, tmp_path):
        """Test model saving and loading."""
        model_path = tmp_path / "test_model.pt"

        # Save model
        mock_model.save_model(str(model_path))

        # Load model
        loaded_model = MockSequentialModel.load_model(str(model_path))

        # Verify loaded model
        assert loaded_model.num_items == mock_model.num_items
        assert loaded_model.embedding_dim == mock_model.embedding_dim

        # Test that loaded model produces same output
        sequences = torch.randint(1, mock_model.num_items + 1, (2, 10))

        with torch.no_grad():
            original_output = mock_model(sequences)
            loaded_output = loaded_model(sequences)

            assert torch.allclose(original_output, loaded_output, atol=1e-6)


class TestSimpleMockModel:
    """Test cases for SimpleMockModel."""

    @pytest.fixture
    def valid_config(self):
        """Fixture providing a valid model configuration."""
        return DictConfig(
            {
                "num_items": 100,
                "max_seq_len": 20,
                "embedding_dim": 64,
                "learning_rate": 0.01,
            }
        )

    @pytest.fixture
    def simple_model(self, valid_config):
        """Fixture providing a simple mock model instance."""
        return SimpleMockModel(valid_config)

    def test_simple_forward_pass(self, simple_model):
        """Test simple model forward pass."""
        sequences = torch.randint(1, simple_model.num_items + 1, (2, 10))

        logits = simple_model(sequences)

        assert logits.shape == (2, 10, simple_model.num_items + 1)
        assert not torch.isnan(logits).any()

    def test_simple_prediction(self, simple_model):
        """Test simple model prediction."""
        sequences = torch.randint(1, simple_model.num_items + 1, (2, 10))

        predictions = simple_model.predict_next_items(sequences, k=5)

        assert predictions.shape == (2, 5)
        assert (predictions >= 1).all()  # Should not predict padding token

    def test_simple_loss_computation(self, simple_model):
        """Test simple model loss computation."""
        sequences = torch.randint(1, simple_model.num_items + 1, (2, 10))
        targets = torch.randint(1, simple_model.num_items + 1, (2, 10))

        logits = simple_model(sequences)
        loss = simple_model.compute_loss(logits, targets)

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestModelErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        config = DictConfig(
            {
                "num_items": 100,
                "max_seq_len": 10,
                "embedding_dim": 64,
                "learning_rate": 0.01,
            }
        )

        model = SimpleMockModel(config)

        # Test with all padding tokens
        sequences = torch.zeros((2, 10), dtype=torch.long)

        logits = model(sequences)
        assert logits.shape == (2, 10, 101)  # num_items + 1

        predictions = model.predict_next_items(sequences, k=5)
        assert predictions.shape == (2, 5)

    def test_single_item_sequence(self):
        """Test handling of single item sequences."""
        config = DictConfig(
            {
                "num_items": 100,
                "max_seq_len": 10,
                "embedding_dim": 64,
                "learning_rate": 0.01,
            }
        )

        model = SimpleMockModel(config)

        # Test with single item and padding
        sequences = torch.zeros((2, 10), dtype=torch.long)
        sequences[:, 0] = 1  # First position has item 1

        logits = model(sequences)
        predictions = model.predict_next_items(sequences, k=5)

        assert logits.shape == (2, 10, 101)
        assert predictions.shape == (2, 5)
        assert (predictions >= 1).all()

    def test_invalid_k_parameter(self):
        """Test prediction with invalid k parameter."""
        config = DictConfig(
            {
                "num_items": 10,  # Small number of items
                "max_seq_len": 5,
                "embedding_dim": 32,
                "learning_rate": 0.01,
            }
        )

        model = SimpleMockModel(config)
        sequences = torch.randint(1, 11, (1, 5))

        # Request more items than available
        predictions = model.predict_next_items(sequences, k=15)
        # Should still work, just return top available items
        assert predictions.shape[1] <= 15
