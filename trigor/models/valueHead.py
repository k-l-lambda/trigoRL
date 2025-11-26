"""
Value Head module for game outcome prediction.

Implements AlphaGo Zero-inspired architecture adapted for transformer models.
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


class ValueHead(nn.Module):
	"""
	Standalone value head module inspired by AlphaGo Zero architecture.

	Adapted for transformer-based models:
	- Input: hidden states [batch, seq_len, hidden_dim] or [batch, hidden_dim]
	- Output: value predictions [batch] or [batch, seq_len] in range [-1, 1]
	- Architecture: FC → LN → ReLU → FC → LN → ReLU → FC → tanh

	Follows AlphaGo Zero's bottleneck design with progressive dimensionality reduction.

	Args:
		hidden_dim: Input hidden dimension (must match base model)
		intermediate_dim: First reduction layer dimension (default: 256)
		bottleneck_dim: Second reduction layer dimension (default: 64)
		dropout: Dropout probability for regularization (default: 0.1)
		use_layer_norm: Use LayerNorm after linear layers (default: True)
		activation: Activation function for hidden layers (default: 'relu')
		output_activation: Activation for output layer (default: 'tanh')

	Example:
		>>> value_head = ValueHead(hidden_dim=256, intermediate_dim=256, bottleneck_dim=64)
		>>> hidden_states = torch.randn(4, 10, 256)  # [batch, seq_len, hidden_dim]
		>>> values = value_head(hidden_states)        # [4, 10] in [-1, 1]
	"""

	def __init__(
		self,
		hidden_dim: int,
		intermediate_dim: int = 256,
		bottleneck_dim: int = 64,
		dropout: float = 0.1,
		use_layer_norm: bool = True,
		activation: str = 'relu',
		output_activation: str = 'tanh',
	):
		"""Initialize ValueHead with AlphaGo Zero-inspired architecture."""
		super().__init__()

		# Store configuration
		self.hidden_dim = hidden_dim
		self.intermediate_dim = intermediate_dim
		self.bottleneck_dim = bottleneck_dim
		self.dropout_p = dropout
		self.use_layer_norm = use_layer_norm

		# Layer 1: hidden_dim → intermediate_dim
		self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
		if use_layer_norm:
			self.ln1 = nn.LayerNorm(intermediate_dim)
		self.dropout1 = nn.Dropout(dropout)

		# Layer 2: intermediate_dim → bottleneck_dim
		self.fc2 = nn.Linear(intermediate_dim, bottleneck_dim)
		if use_layer_norm:
			self.ln2 = nn.LayerNorm(bottleneck_dim)
		self.dropout2 = nn.Dropout(dropout)

		# Output layer: bottleneck_dim → 1
		self.fc_out = nn.Linear(bottleneck_dim, 1)

		# Activation functions
		self.activation_fn = self._get_activation(activation)
		self.output_activation_fn = self._get_activation(output_activation)


	def _get_activation(self, activation: str) -> nn.Module:
		"""Get activation function by name."""
		activations = {
			'relu': nn.ReLU(),
			'gelu': nn.GELU(),
			'tanh': nn.Tanh(),
			'sigmoid': nn.Sigmoid(),
		}

		if activation.lower() not in activations:
			raise ValueError(
				f"Unknown activation '{activation}'. "
				f"Available: {', '.join(activations.keys())}"
			)

		return activations[activation.lower()]


	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], DictConfig]) -> 'ValueHead':
		"""
		Create ValueHead from configuration.

		Args:
			config: Configuration dictionary or DictConfig with keys:
				- hidden_dim: Input dimension (required)
				- intermediate_dim: First reduction dimension (default: 256)
				- bottleneck_dim: Second reduction dimension (default: 64)
				- dropout: Dropout probability (default: 0.1)
				- use_layer_norm: Use LayerNorm (default: True)
				- activation: Hidden activation (default: 'relu')
				- output_activation: Output activation (default: 'tanh')

		Returns:
			Instantiated ValueHead module

		Example:
			>>> config = {
			...     'hidden_dim': 512,
			...     'intermediate_dim': 256,
			...     'bottleneck_dim': 64,
			...     'dropout': 0.2,
			... }
			>>> value_head = ValueHead.from_config(config)
		"""
		# Convert to DictConfig for unified API
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Extract parameters with defaults
		hidden_dim = config.hidden_dim  # Required
		intermediate_dim = config.get('intermediate_dim', 256)
		bottleneck_dim = config.get('bottleneck_dim', 64)
		dropout = config.get('dropout', 0.1)
		use_layer_norm = config.get('use_layer_norm', True)
		activation = config.get('activation', 'relu')
		output_activation = config.get('output_activation', 'tanh')

		return cls(
			hidden_dim=hidden_dim,
			intermediate_dim=intermediate_dim,
			bottleneck_dim=bottleneck_dim,
			dropout=dropout,
			use_layer_norm=use_layer_norm,
			activation=activation,
			output_activation=output_activation,
		)


	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass.

		Accepts both 2D and 3D inputs, automatically handling shape transformations.

		Args:
			hidden_states: Input hidden states
				- 2D: [batch_size, hidden_dim]
				- 3D: [batch_size, seq_len, hidden_dim]

		Returns:
			values: Value predictions in range [-1, 1]
				- 2D input: [batch_size]
				- 3D input: [batch_size, seq_len]

		Example:
			>>> value_head = ValueHead(hidden_dim=256)
			>>> # 2D input
			>>> hidden = torch.randn(4, 256)
			>>> values = value_head(hidden)  # [4]
			>>> # 3D input
			>>> hidden = torch.randn(4, 10, 256)
			>>> values = value_head(hidden)  # [4, 10]
		"""
		# Shape detection
		input_shape = hidden_states.shape
		is_3d = len(input_shape) == 3

		if is_3d:
			batch_size, seq_len, hidden_dim = input_shape
			# Flatten for processing: [batch * seq_len, hidden_dim]
			x = hidden_states.view(-1, hidden_dim)
		else:
			x = hidden_states

		# Layer 1: hidden_dim → intermediate_dim
		x = self.fc1(x)
		if self.use_layer_norm:
			x = self.ln1(x)
		x = self.activation_fn(x)
		x = self.dropout1(x)

		# Layer 2: intermediate_dim → bottleneck_dim
		x = self.fc2(x)
		if self.use_layer_norm:
			x = self.ln2(x)
		x = self.activation_fn(x)
		x = self.dropout2(x)

		# Output: bottleneck_dim → 1
		x = self.fc_out(x)  # [batch * seq_len, 1] or [batch, 1]
		x = self.output_activation_fn(x)

		# Squeeze last dimension
		x = x.squeeze(-1)  # [batch * seq_len] or [batch]

		# Restore 3D shape if needed
		if is_3d:
			x = x.view(batch_size, seq_len)

		return x


	def get_model_info(self) -> Dict[str, Any]:
		"""
		Get model architecture information.

		Returns:
			Dictionary containing model configuration details
		"""
		return {
			'model_type': 'ValueHead',
			'hidden_dim': self.hidden_dim,
			'intermediate_dim': self.intermediate_dim,
			'bottleneck_dim': self.bottleneck_dim,
			'dropout': self.dropout_p,
			'use_layer_norm': self.use_layer_norm,
			'total_parameters': self.count_parameters()['total'],
			'trainable_parameters': self.count_parameters()['trainable'],
		}


	def count_parameters(self) -> Dict[str, int]:
		"""
		Count total and trainable parameters.

		Returns:
			Dictionary with 'total' and 'trainable' parameter counts
		"""
		total = sum(p.numel() for p in self.parameters())
		trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

		return {
			'total': total,
			'trainable': trainable,
			'non_trainable': total - trainable,
		}


	def __repr__(self) -> str:
		"""Return readable string representation."""
		info = self.get_model_info()
		return (
			f"ValueHead(\n"
			f"  hidden_dim={info['hidden_dim']},\n"
			f"  intermediate_dim={info['intermediate_dim']},\n"
			f"  bottleneck_dim={info['bottleneck_dim']},\n"
			f"  dropout={info['dropout']},\n"
			f"  use_layer_norm={info['use_layer_norm']},\n"
			f"  parameters={info['total_parameters']:,}\n"
			f")"
		)
