"""Unit tests for the model registry system.

This module contains comprehensive tests for the ModelRegistry class,
model registration, discovery, and instantiation functionality.
"""

import pytest
import torch
from omegaconf import DictConfig
from unittest.mock import patch

from pytorch_recsys.models.registry import ModelRegistry, ModelMetadata, create_model
from pytorch_recsys.models.base import BaseSequentialModel
from pytorch_recsys.models.mock import MockSequentialModel, SimpleMockModel


class DummyModel(BaseSequentialModel):
    """Dummy model for testing registry functionality."""
    
    def forward(self, sequences, **kwargs):
        return torch.randn(sequences.shape[0], sequences.shape[1], self.num_items + 1)
    
    def predict_next_items(self, sequences, k=10):
        batch_size = sequences.shape[0]
        return torch.randint(1, self.num_items + 1, (batch_size, k))
    
    def compute_loss(self, logits, targets):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        return self.loss_fn(logits_flat, targets_flat)


class TestModelRegistry:
    """Test cases for ModelRegistry functionality."""
    
    def setup_method(self):
        """Setup method run before each test."""
        # Clear registry before each test
        ModelRegistry.clear()
    
    def teardown_method(self):
        """Teardown method run after each test."""
        # Clear registry after each test
        ModelRegistry.clear()
    
    def test_model_registration(self):
        """Test basic model registration."""
        @ModelRegistry.register(
            name="test_model",
            description="Test model for unit testing",
            version="1.0.0"
        )
        class TestModel(BaseSequentialModel):
            def forward(self, sequences, **kwargs):
                return torch.randn(sequences.shape[0], sequences.shape[1], self.num_items + 1)
            
            def predict_next_items(self, sequences, k=10):
                return torch.randint(1, self.num_items + 1, (sequences.shape[0], k))
            
            def compute_loss(self, logits, targets):
                return torch.tensor(0.5)
        
        # Check that model is registered
        assert ModelRegistry.is_registered("test_model")
        assert "test_model" in ModelRegistry.get_registered_names()
        
        # Check model info
        info = ModelRegistry.get_model_info("test_model")
        assert info["name"] == "test_model"
        assert info["class_name"] == "TestModel"
        assert info["description"] == "Test model for unit testing"
        assert info["version"] == "1.0.0"
    
    def test_registration_with_tags_and_requirements(self):
        """Test model registration with tags and requirements."""
        @ModelRegistry.register(
            name="advanced_model",
            description="Advanced test model",
            version="2.1.0",
            tags=["transformer", "attention", "advanced"],
            requirements={"torch": ">=2.0.0", "transformers": ">=4.0.0"}
        )
        class AdvancedModel(DummyModel):
            pass
        
        info = ModelRegistry.get_model_info("advanced_model")
        assert info["tags"] == ["transformer", "attention", "advanced"]
        assert info["requirements"]["torch"] == ">=2.0.0"
        assert info["requirements"]["transformers"] == ">=4.0.0"
    
    def test_duplicate_registration_warning(self):
        """Test that duplicate registration shows warning."""
        @ModelRegistry.register(name="duplicate_test")
        class FirstModel(DummyModel):
            pass
        
        with patch('pytorch_recsys.models.registry.logger.warning') as mock_warning:
            @ModelRegistry.register(name="duplicate_test", version="2.0.0")
            class SecondModel(DummyModel):
                pass
            
            mock_warning.assert_called_once()
            # Second model should override first
            assert ModelRegistry.get_model_info("duplicate_test")["version"] == "2.0.0"
    
    def test_invalid_model_registration(self):
        """Test registration of invalid models raises errors."""
        # Test non-BaseSequentialModel class
        with pytest.raises(TypeError, match="must inherit from BaseSequentialModel"):
            @ModelRegistry.register(name="invalid_model")
            class InvalidModel:
                pass
        
        # Test empty name
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            @ModelRegistry.register(name="")
            class EmptyNameModel(DummyModel):
                pass
        
        # Test non-string name
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            @ModelRegistry.register(name=123)
            class NumericNameModel(DummyModel):
                pass
    
    def test_model_creation(self):
        """Test model instantiation from registry."""
        @ModelRegistry.register(name="creation_test")
        class CreationTestModel(DummyModel):
            pass
        
        config = DictConfig({
            "num_items": 1000,
            "max_seq_len": 50,
            "embedding_dim": 128,
            "learning_rate": 0.001
        })
        
        model = ModelRegistry.create_model("creation_test", config)
        
        assert isinstance(model, CreationTestModel)
        assert model.num_items == 1000
        assert model.max_seq_len == 50
        assert model.embedding_dim == 128
    
    def test_model_creation_nonexistent(self):
        """Test that creating non-existent model raises error."""
        config = DictConfig({
            "num_items": 1000,
            "max_seq_len": 50,
            "embedding_dim": 128
        })
        
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            ModelRegistry.create_model("nonexistent", config)
    
    def test_model_creation_failure(self):
        """Test handling of model creation failures."""
        @ModelRegistry.register(name="failing_model")
        class FailingModel(BaseSequentialModel):
            def __init__(self, config):
                # This will fail because we don't call super().__init__
                raise RuntimeError("Intentional failure")
            
            def forward(self, sequences, **kwargs):
                pass
            
            def predict_next_items(self, sequences, k=10):
                pass
            
            def compute_loss(self, logits, targets):
                pass
        
        config = DictConfig({
            "num_items": 1000,
            "max_seq_len": 50,
            "embedding_dim": 128
        })
        
        with pytest.raises(TypeError, match="Failed to create model 'failing_model'"):
            ModelRegistry.create_model("failing_model", config)
    
    def test_get_model_class(self):
        """Test retrieving model class from registry."""
        @ModelRegistry.register(name="class_test")
        class ClassTestModel(DummyModel):
            pass
        
        model_class = ModelRegistry.get_model_class("class_test")
        
        assert model_class == ClassTestModel
        assert issubclass(model_class, BaseSequentialModel)
    
    def test_get_model_class_nonexistent(self):
        """Test that getting non-existent model class raises error."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            ModelRegistry.get_model_class("nonexistent")
    
    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        models = ModelRegistry.list_models()
        assert models == []
    
    def test_list_models_populated(self):
        """Test listing models with populated registry."""
        @ModelRegistry.register(name="model1", description="First model")
        class Model1(DummyModel):
            pass
        
        @ModelRegistry.register(name="model2", description="Second model")
        class Model2(DummyModel):
            pass
        
        models = ModelRegistry.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "model1"  # Should be sorted
        assert models[1]["name"] == "model2"
        
        # Check that all required fields are present
        for model in models:
            assert "name" in model
            assert "class_name" in model
            assert "description" in model
            assert "version" in model
            assert "tags" in model
    
    def test_list_models_with_tag_filter(self):
        """Test listing models with tag filtering."""
        @ModelRegistry.register(name="transformer_model", tags=["transformer", "attention"])
        class TransformerModel(DummyModel):
            pass
        
        @ModelRegistry.register(name="simple_model", tags=["simple"])
        class SimpleModel(DummyModel):
            pass
        
        @ModelRegistry.register(name="rnn_model", tags=["rnn", "sequential"])
        class RNNModel(DummyModel):
            pass
        
        # Filter by single tag
        transformer_models = ModelRegistry.list_models(tags=["transformer"])
        assert len(transformer_models) == 1
        assert transformer_models[0]["name"] == "transformer_model"
        
        # Filter by multiple tags (OR logic)
        filtered_models = ModelRegistry.list_models(tags=["simple", "rnn"])
        assert len(filtered_models) == 2
        
        # Filter by non-existent tag
        empty_models = ModelRegistry.list_models(tags=["nonexistent"])
        assert len(empty_models) == 0
    
    def test_unregister_model(self):
        """Test unregistering models."""
        @ModelRegistry.register(name="unregister_test")
        class UnregisterTestModel(DummyModel):
            pass
        
        # Model should be registered
        assert ModelRegistry.is_registered("unregister_test")
        
        # Unregister model
        result = ModelRegistry.unregister("unregister_test")
        assert result is True
        assert not ModelRegistry.is_registered("unregister_test")
        
        # Try to unregister non-existent model
        result = ModelRegistry.unregister("nonexistent")
        assert result is False
    
    def test_clear_registry(self):
        """Test clearing the entire registry."""
        @ModelRegistry.register(name="clear_test1")
        class ClearTest1(DummyModel):
            pass
        
        @ModelRegistry.register(name="clear_test2")
        class ClearTest2(DummyModel):
            pass
        
        assert len(ModelRegistry.get_registered_names()) == 2
        
        ModelRegistry.clear()
        
        assert len(ModelRegistry.get_registered_names()) == 0
        assert not ModelRegistry.is_registered("clear_test1")
        assert not ModelRegistry.is_registered("clear_test2")
    
    def test_get_models_by_tag(self):
        """Test getting models by specific tag."""
        @ModelRegistry.register(name="tag_test1", tags=["tag1", "tag2"])
        class TagTest1(DummyModel):
            pass
        
        @ModelRegistry.register(name="tag_test2", tags=["tag2", "tag3"])
        class TagTest2(DummyModel):
            pass
        
        @ModelRegistry.register(name="tag_test3", tags=["tag3"])
        class TagTest3(DummyModel):
            pass
        
        # Get models with tag1
        tag1_models = ModelRegistry.get_models_by_tag("tag1")
        assert tag1_models == ["tag_test1"]
        
        # Get models with tag2
        tag2_models = ModelRegistry.get_models_by_tag("tag2")
        assert set(tag2_models) == {"tag_test1", "tag_test2"}
        
        # Get models with tag3
        tag3_models = ModelRegistry.get_models_by_tag("tag3")
        assert set(tag3_models) == {"tag_test2", "tag_test3"}
        
        # Get models with non-existent tag
        empty_models = ModelRegistry.get_models_by_tag("nonexistent")
        assert empty_models == []
    
    def test_validate_requirements(self):
        """Test requirements validation."""
        @ModelRegistry.register(
            name="requirements_test",
            requirements={"torch": ">=2.0.0", "pytorch-lightning": ">=2.0.0"}
        )
        class RequirementsTestModel(DummyModel):
            pass
        
        validation_results = ModelRegistry.validate_requirements("requirements_test")
        
        assert "torch" in validation_results
        assert "pytorch-lightning" in validation_results
        
        # Both should be True since they're installed
        assert validation_results["torch"] is True
        assert validation_results["pytorch-lightning"] is True
    
    def test_validate_requirements_nonexistent_model(self):
        """Test requirements validation for non-existent model."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            ModelRegistry.validate_requirements("nonexistent")


class TestCreateModelFunction:
    """Test cases for the standalone create_model function."""
    
    def setup_method(self):
        """Setup method run before each test."""
        ModelRegistry.clear()
    
    def teardown_method(self):
        """Teardown method run after each test."""
        ModelRegistry.clear()
    
    def test_create_model_with_dict_config(self):
        """Test creating model with dictionary configuration."""
        @ModelRegistry.register(name="dict_config_test")
        class DictConfigTestModel(DummyModel):
            pass
        
        config_dict = {
            "num_items": 500,
            "max_seq_len": 30,
            "embedding_dim": 64,
            "learning_rate": 0.01
        }
        
        model = create_model("dict_config_test", config_dict)
        
        assert isinstance(model, DictConfigTestModel)
        assert model.num_items == 500
        assert model.max_seq_len == 30
        assert model.embedding_dim == 64
    
    def test_create_model_with_omegaconf(self):
        """Test creating model with OmegaConf configuration."""
        @ModelRegistry.register(name="omega_config_test")
        class OmegaConfigTestModel(DummyModel):
            pass
        
        config = DictConfig({
            "num_items": 2000,
            "max_seq_len": 100,
            "embedding_dim": 256,
            "learning_rate": 0.0001
        })
        
        model = create_model("omega_config_test", config)
        
        assert isinstance(model, OmegaConfigTestModel)
        assert model.num_items == 2000
        assert model.max_seq_len == 100
        assert model.embedding_dim == 256


class TestModelMetadata:
    """Test cases for ModelMetadata Pydantic model."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = ModelMetadata(
            name="test_model",
            class_name="TestModel",
            description="Test description",
            version="1.0.0",
            module="test.module",
            file_path="/path/to/file.py",
            model_class=DummyModel,
            tags=["test", "mock"],
            requirements={"torch": ">=2.0.0"}
        )
        
        assert metadata.name == "test_model"
        assert metadata.class_name == "TestModel"
        assert metadata.description == "Test description"
        assert metadata.version == "1.0.0"
        assert metadata.tags == ["test", "mock"]
        assert metadata.requirements == {"torch": ">=2.0.0"}
    
    def test_metadata_defaults(self):
        """Test metadata with default values."""
        metadata = ModelMetadata(
            name="minimal_model",
            class_name="MinimalModel",
            module="test.module",
            file_path="/path/to/file.py",
            model_class=DummyModel
        )
        
        assert metadata.description == ""
        assert metadata.version == "1.0.0"
        assert metadata.tags == []
        assert metadata.requirements == {}


class TestMockModelRegistration:
    """Test that mock models are properly registered."""
    
    def test_mock_models_are_registered(self):
        """Test that MockSequentialModel and SimpleMockModel are registered."""
        # Import should trigger registration
        from pytorch_recsys.models.mock import MockSequentialModel, SimpleMockModel
        
        assert ModelRegistry.is_registered("mock_model")
        assert ModelRegistry.is_registered("simple_mock")
        
        # Test mock model info
        mock_info = ModelRegistry.get_model_info("mock_model")
        assert mock_info["description"] == "Simple mock model for testing and development"
        assert "mock" in mock_info["tags"]
        assert "testing" in mock_info["tags"]
        
        # Test simple mock info
        simple_info = ModelRegistry.get_model_info("simple_mock")
        assert simple_info["description"] == "Even simpler mock model for basic testing"
        assert "simple" in simple_info["tags"]
    
    def test_create_mock_models(self):
        """Test creating mock models from registry."""
        config = DictConfig({
            "num_items": 100,
            "max_seq_len": 20,
            "embedding_dim": 64,
            "learning_rate": 0.01
        })
        
        # Create mock model
        mock_model = create_model("mock_model", config)
        assert isinstance(mock_model, MockSequentialModel)
        
        # Create simple mock model
        simple_model = create_model("simple_mock", config)
        assert isinstance(simple_model, SimpleMockModel)
        
        # Test that both models work
        sequences = torch.randint(1, 101, (2, 10))
        
        mock_output = mock_model(sequences)
        simple_output = simple_model(sequences)
        
        assert mock_output.shape == (2, 10, 101)
        assert simple_output.shape == (2, 10, 101)