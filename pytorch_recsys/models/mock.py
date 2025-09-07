"""Mock model implementation for testing the base interface.

This module provides a simple mock model that implements all required abstract
methods from BaseSequentialModel. It's designed for testing purposes and as
a reference implementation for new model developers.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from .base import BaseSequentialModel
from .registry import ModelRegistry


@ModelRegistry.register(
    name="mock_model",
    description="Simple mock model for testing and development",
    version="1.0.0",
    tags=["mock", "testing", "simple"],
    requirements={"torch": ">=2.0.0"}
)
class MockSequentialModel(BaseSequentialModel):
    """Simple mock implementation of BaseSequentialModel for testing.
    
    This model provides a minimal implementation that:
    - Uses item embeddings with a simple linear layer
    - Implements basic forward pass with dropout
    - Provides simple prediction logic
    - Uses standard cross-entropy loss
    
    It's designed to be:
    - Fast to train and test
    - Simple to understand
    - Suitable for unit testing
    - A reference for implementing new models
    
    Args:
        config: Model configuration containing required parameters
        
    Example:
        ```python
        config = DictConfig({
            "num_items": 1000,
            "max_seq_len": 50,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "dropout": 0.1,
            "learning_rate": 0.001
        })
        model = MockSequentialModel(config)
        ```
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the mock sequential model.
        
        Args:
            config: Model configuration with all required parameters
        """
        super().__init__(config)
        
        # Extract mock-specific parameters
        self.hidden_dim = config.get('hidden_dim', self.embedding_dim * 2)
        self.dropout_rate = config.get('dropout', 0.1)
        self.num_layers = config.get('num_layers', 2)
        
        # Build the model architecture
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the mock model architecture."""
        # Positional embeddings
        self.position_embeddings = nn.Embedding(
            self.max_seq_len,
            self.embedding_dim
        )
        
        # Simple transformer-like encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout_rate,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Output projection
        self.output_projection = nn.Linear(self.embedding_dim, self.num_items + 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, sequences: Tensor, **kwargs) -> Tensor:
        """Forward pass of the mock model.
        
        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            **kwargs: Additional arguments (unused in mock model)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, num_items + 1)
        """
        batch_size, seq_len = sequences.shape
        
        # Create attention mask for padding tokens
        attention_mask = (sequences != 0).float()
        
        # Get item embeddings
        item_embeds = self.item_embeddings(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embeddings(positions)
        
        # Combine embeddings
        embeddings = item_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Apply transformer encoder layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            # Create causal attention mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=sequences.device), diagonal=1).bool()
            
            # Apply padding mask
            src_key_padding_mask = (sequences == 0)
            
            hidden_states = layer(
                hidden_states,
                src_mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Project to vocabulary size
        logits = self.output_projection(hidden_states)
        
        # Mask out logits for padding tokens
        logits = logits * attention_mask.unsqueeze(-1)
        
        return logits
    
    def predict_next_items(self, sequences: Tensor, k: int = 10) -> Tensor:
        """Generate top-k item predictions for given sequences.
        
        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            k: Number of items to predict
            
        Returns:
            Tensor of shape (batch_size, k) containing top-k item IDs
        """
        # Get model predictions
        logits = self(sequences)  # (batch_size, seq_len, num_items + 1)
        
        # Use the last non-padding position for prediction
        batch_size = sequences.shape[0]
        last_positions = []
        
        for i in range(batch_size):
            # Find last non-zero (non-padding) position
            non_zero = (sequences[i] != 0).nonzero(as_tuple=True)[0]
            if len(non_zero) > 0:
                last_positions.append(non_zero[-1].item())
            else:
                last_positions.append(0)  # If all padding, use position 0
        
        # Extract logits at last positions
        last_logits = []
        for i, pos in enumerate(last_positions):
            last_logits.append(logits[i, pos])
        
        last_logits = torch.stack(last_logits)  # (batch_size, num_items + 1)
        
        # Exclude padding token (index 0) from predictions
        last_logits[:, 0] = float('-inf')
        
        # Get top-k predictions
        _, top_k_items = torch.topk(last_logits, k, dim=-1)
        
        return top_k_items
    
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the training loss.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, num_items + 1)
            targets: Target sequences of shape (batch_size, seq_len)
            
        Returns:
            Scalar loss tensor
        """
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten logits and targets
        logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, num_items + 1)
        targets_flat = targets.view(-1)  # (batch_size * seq_len,)
        
        # Compute cross-entropy loss (ignores padding tokens with index 0)
        loss = self.loss_fn(logits_flat, targets_flat)
        
        return loss
    
    def get_attention_weights(self, sequences: Tensor) -> Tensor:
        """Get attention weights from the model (mock implementation).
        
        This is a mock implementation that returns random attention weights
        for demonstration purposes. Real models would return actual attention weights.
        
        Args:
            sequences: Input sequences of shape (batch_size, seq_len)
            
        Returns:
            Mock attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = sequences.shape
        
        # Create mock attention weights (random for demonstration)
        attention_weights = torch.rand(batch_size, seq_len, seq_len, device=sequences.device)
        
        # Make it causal (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=sequences.device))
        attention_weights = attention_weights * mask
        
        # Normalize
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        return attention_weights
    
    def get_item_embeddings(self) -> Tensor:
        """Get the item embedding matrix.
        
        Returns:
            Item embeddings tensor of shape (num_items + 1, embedding_dim)
        """
        return self.item_embeddings.weight.detach()
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture.
        
        Returns:
            String summary of the model
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
MockSequentialModel Summary:
- Items: {self.num_items}
- Max sequence length: {self.max_seq_len}
- Embedding dimension: {self.embedding_dim}
- Hidden dimension: {self.hidden_dim}
- Number of layers: {self.num_layers}
- Dropout rate: {self.dropout_rate}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
        """.strip()
        
        return summary


@ModelRegistry.register(
    name="simple_mock",
    description="Even simpler mock model for basic testing",
    version="1.0.0",
    tags=["mock", "simple", "minimal"]
)
class SimpleMockModel(BaseSequentialModel):
    """Ultra-simple mock model for basic testing.
    
    This model implements the absolute minimum required by BaseSequentialModel:
    - Single linear layer from embeddings to logits
    - No attention mechanisms or complex architectures
    - Suitable for fast unit testing
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the simple mock model."""
        super().__init__(config)
        
        # Simple linear transformation
        self.linear = nn.Linear(self.embedding_dim, self.num_items + 1)
        
        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
    
    def forward(self, sequences: Tensor, **kwargs) -> Tensor:
        """Simple forward pass."""
        # Get embeddings
        embeddings = self.item_embeddings(sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Linear transformation to logits
        logits = self.linear(embeddings)  # (batch_size, seq_len, num_items + 1)
        
        return logits
    
    def predict_next_items(self, sequences: Tensor, k: int = 10) -> Tensor:
        """Simple prediction using last position."""
        logits = self(sequences)
        
        # Use last position for each sequence
        last_logits = logits[:, -1, :]  # (batch_size, num_items + 1)
        
        # Exclude padding token
        last_logits[:, 0] = float('-inf')
        
        # Get top-k
        _, top_k = torch.topk(last_logits, k, dim=-1)
        
        return top_k
    
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Simple cross-entropy loss."""
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        return self.loss_fn(logits_flat, targets_flat)