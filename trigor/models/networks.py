"""Neural network architectures for TrigoRL."""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
	"""
	Simple Multi-Layer Perceptron.

	Basic feedforward network for policy/value functions.
	"""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dims: List[int] = [256, 256],
		activation: str = 'relu',
		dropout: float = 0.0,
	):
		"""
		Initialize MLP.

		Args:
		    input_dim: Input dimension
		    output_dim: Output dimension
		    hidden_dims: List of hidden layer dimensions
		    activation: Activation function ('relu', 'tanh', 'gelu')
		    dropout: Dropout probability
		"""
		super().__init__()

		# Build layers
		layers = []
		prev_dim = input_dim

		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(prev_dim, hidden_dim))

			# Activation
			if activation == 'relu':
				layers.append(nn.ReLU())
			elif activation == 'tanh':
				layers.append(nn.Tanh())
			elif activation == 'gelu':
				layers.append(nn.GELU())
			else:
				raise ValueError(f"Unknown activation: {activation}")

			# Dropout
			if dropout > 0:
				layers.append(nn.Dropout(dropout))

			prev_dim = hidden_dim

		# Output layer
		layers.append(nn.Linear(prev_dim, output_dim))

		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass."""
		return self.network(x)


class PolicyValueNetwork(nn.Module):
	"""
	Shared network for policy and value functions.

	Used in actor-critic algorithms like PPO.
	"""

	def __init__(
		self,
		observation_dim: int,
		action_dim: int,
		hidden_dims: List[int] = [256, 256],
		activation: str = 'relu',
		dropout: float = 0.0,
	):
		"""
		Initialize policy-value network.

		Args:
		    observation_dim: Observation space dimension
		    action_dim: Action space dimension
		    hidden_dims: Shared hidden layer dimensions
		    activation: Activation function
		    dropout: Dropout probability
		"""
		super().__init__()

		# Shared feature extractor
		feature_layers = []
		prev_dim = observation_dim

		for hidden_dim in hidden_dims:
			feature_layers.append(nn.Linear(prev_dim, hidden_dim))

			if activation == 'relu':
				feature_layers.append(nn.ReLU())
			elif activation == 'tanh':
				feature_layers.append(nn.Tanh())
			elif activation == 'gelu':
				feature_layers.append(nn.GELU())

			if dropout > 0:
				feature_layers.append(nn.Dropout(dropout))

			prev_dim = hidden_dim

		self.features = nn.Sequential(*feature_layers)

		# Policy head (outputs action logits)
		self.policy_head = nn.Linear(prev_dim, action_dim)

		# Value head (outputs state value)
		self.value_head = nn.Linear(prev_dim, 1)

	def forward(self, x: torch.Tensor) -> tuple:
		"""
		Forward pass.

		Returns:
		    (action_logits, state_value)
		"""
		features = self.features(x)
		action_logits = self.policy_head(features)
		state_value = self.value_head(features)

		return action_logits, state_value

	def get_action_logits(self, x: torch.Tensor) -> torch.Tensor:
		"""Get action logits only."""
		features = self.features(x)
		return self.policy_head(features)

	def get_value(self, x: torch.Tensor) -> torch.Tensor:
		"""Get state value only."""
		features = self.features(x)
		return self.value_head(features)
