"""
Unit tests for ValueHead module.

Tests value head architecture, forward pass, configuration, and integration.
"""

import pytest
import torch
from omegaconf import OmegaConf

from trigor.models import ValueHead, make_model


class TestValueHeadForward:
	"""Test forward pass functionality."""

	def test_value_head_2d_forward(self):
		"""Test forward pass with [batch, hidden_dim] input."""
		value_head = ValueHead(hidden_dim=256, intermediate_dim=128, bottleneck_dim=32)

		hidden_states = torch.randn(4, 256)
		values = value_head(hidden_states)

		assert values.shape == (4,)
		assert values.min() >= -1.0 and values.max() <= 1.0


	def test_value_head_3d_forward(self):
		"""Test forward pass with [batch, seq_len, hidden_dim] input."""
		value_head = ValueHead(hidden_dim=256)

		hidden_states = torch.randn(2, 10, 256)
		values = value_head(hidden_states)

		assert values.shape == (2, 10)
		assert torch.all(values >= -1.0) and torch.all(values <= 1.0)


	def test_value_head_output_range(self):
		"""Test that output is bounded to [-1, 1] by tanh."""
		value_head = ValueHead(hidden_dim=64)

		# Extreme input values
		hidden_states = torch.randn(100, 64) * 100
		values = value_head(hidden_states)

		assert torch.all(values >= -1.0)
		assert torch.all(values <= 1.0)


	def test_value_head_large_batch(self):
		"""Test with large batch size."""
		value_head = ValueHead(hidden_dim=128)

		hidden_states = torch.randn(128, 128)
		values = value_head(hidden_states)

		assert values.shape == (128,)


	def test_value_head_single_sample(self):
		"""Test with single sample."""
		value_head = ValueHead(hidden_dim=256)

		hidden_states = torch.randn(1, 256)
		values = value_head(hidden_states)

		assert values.shape == (1,)


class TestValueHeadConfiguration:
	"""Test configuration and initialization."""

	def test_value_head_from_config_dict(self):
		"""Test creating ValueHead from config dict."""
		config = {
			'hidden_dim': 512,
			'intermediate_dim': 256,
			'bottleneck_dim': 64,
			'dropout': 0.2,
		}

		value_head = ValueHead.from_config(config)

		assert value_head.fc1.in_features == 512
		assert value_head.fc1.out_features == 256
		assert value_head.fc2.out_features == 64
		assert value_head.fc_out.out_features == 1


	def test_value_head_from_config_omegaconf(self):
		"""Test creating ValueHead from OmegaConf."""
		config = OmegaConf.create({
			'hidden_dim': 256,
			'intermediate_dim': 128,
			'bottleneck_dim': 32,
		})

		value_head = ValueHead.from_config(config)

		assert value_head.hidden_dim == 256
		assert value_head.intermediate_dim == 128
		assert value_head.bottleneck_dim == 32


	def test_value_head_defaults(self):
		"""Test default parameter values."""
		value_head = ValueHead(hidden_dim=256)

		assert value_head.intermediate_dim == 256
		assert value_head.bottleneck_dim == 64
		assert value_head.dropout_p == 0.1
		assert value_head.use_layer_norm == True


	def test_value_head_no_layer_norm(self):
		"""Test ValueHead without LayerNorm."""
		value_head = ValueHead(hidden_dim=128, use_layer_norm=False)

		assert not hasattr(value_head, 'ln1') or value_head.ln1 is None
		assert not hasattr(value_head, 'ln2') or value_head.ln2 is None


	def test_value_head_activation_options(self):
		"""Test different activation functions."""
		# ReLU
		vh_relu = ValueHead(hidden_dim=128, activation='relu')
		assert isinstance(vh_relu.activation_fn, torch.nn.ReLU)

		# GELU
		vh_gelu = ValueHead(hidden_dim=128, activation='gelu')
		assert isinstance(vh_gelu.activation_fn, torch.nn.GELU)

		# Invalid activation
		with pytest.raises(ValueError):
			ValueHead(hidden_dim=128, activation='invalid')


class TestValueHeadUtilities:
	"""Test utility methods."""

	def test_value_head_count_parameters(self):
		"""Test parameter counting utility."""
		value_head = ValueHead(hidden_dim=256, intermediate_dim=128, bottleneck_dim=32)

		counts = value_head.count_parameters()

		assert 'total' in counts
		assert 'trainable' in counts
		assert 'non_trainable' in counts
		assert counts['total'] > 0
		assert counts['trainable'] == counts['total']
		assert counts['non_trainable'] == 0


	def test_value_head_get_model_info(self):
		"""Test get_model_info method."""
		value_head = ValueHead(
			hidden_dim=256,
			intermediate_dim=128,
			bottleneck_dim=32,
			dropout=0.15,
		)

		info = value_head.get_model_info()

		assert info['model_type'] == 'ValueHead'
		assert info['hidden_dim'] == 256
		assert info['intermediate_dim'] == 128
		assert info['bottleneck_dim'] == 32
		assert info['dropout'] == 0.15
		assert 'total_parameters' in info
		assert 'trainable_parameters' in info


	def test_value_head_repr(self):
		"""Test __repr__ method."""
		value_head = ValueHead(hidden_dim=128)

		repr_str = repr(value_head)

		assert 'ValueHead' in repr_str
		assert '128' in repr_str
		assert 'parameters' in repr_str


class TestValueHeadGradients:
	"""Test gradient flow and backward pass."""

	def test_value_head_gradient_flow(self):
		"""Test that gradients flow through the network."""
		value_head = ValueHead(hidden_dim=128)

		hidden_states = torch.randn(4, 128, requires_grad=True)
		values = value_head(hidden_states)

		loss = values.sum()
		loss.backward()

		assert hidden_states.grad is not None
		for param in value_head.parameters():
			assert param.grad is not None


	def test_value_head_gradient_magnitudes(self):
		"""Test gradient magnitudes are reasonable."""
		value_head = ValueHead(hidden_dim=64)

		hidden_states = torch.randn(8, 64, requires_grad=True)
		values = value_head(hidden_states)

		loss = values.mean()
		loss.backward()

		# Check gradients are not too large or too small
		for param in value_head.parameters():
			if param.grad is not None:
				grad_norm = param.grad.norm().item()
				assert grad_norm < 100.0  # Not exploding
				assert grad_norm > 1e-8  # Not vanishing


class TestValueHeadRegistry:
	"""Test model registry integration."""

	def test_value_head_registry(self):
		"""Test that ValueHead is properly registered."""
		from trigor.models import make_model

		config = {'hidden_dim': 128}
		value_head = make_model('ValueHead', config)

		assert isinstance(value_head, ValueHead)


	def test_value_head_list_models(self):
		"""Test ValueHead appears in list_models."""
		from trigor.models import list_models

		models = list_models()

		assert 'ValueHead' in models


class TestValueHeadIntegration:
	"""Integration tests with other models."""

	def test_value_head_with_gpt2(self):
		"""Test ValueHead with GPT2 hidden states."""
		from trigor.models import GPT2CausalLM

		# Create base model
		model = GPT2CausalLM.from_config({
			'vocab_size': 128,
			'hidden_size': 256,
			'num_layers': 2,
			'num_heads': 4,
		})

		# Create value head
		value_head = ValueHead(hidden_dim=256)

		# Forward pass
		input_ids = torch.randint(0, 128, (2, 10))
		outputs = model(input_ids, output_hidden_states=True)
		hidden_states = outputs.hidden_states[-1]

		# Value predictions
		values = value_head(hidden_states)

		assert values.shape == (2, 10)
		assert torch.all(values >= -1.0) and torch.all(values <= 1.0)


	def test_value_head_with_different_dtypes(self):
		"""Test ValueHead with different dtypes."""
		value_head = ValueHead(hidden_dim=128)

		# Float32
		hidden_fp32 = torch.randn(4, 128, dtype=torch.float32)
		values_fp32 = value_head(hidden_fp32)
		assert values_fp32.dtype == torch.float32

		# Float16 (if GPU available)
		if torch.cuda.is_available():
			value_head_gpu = value_head.cuda().half()
			hidden_fp16 = torch.randn(4, 128, dtype=torch.float16, device='cuda')
			values_fp16 = value_head_gpu(hidden_fp16)
			assert values_fp16.dtype == torch.float16


class TestValueHeadEdgeCases:
	"""Test edge cases and error handling."""

	def test_value_head_zero_batch(self):
		"""Test with zero batch size (should work)."""
		value_head = ValueHead(hidden_dim=128)

		hidden_states = torch.randn(0, 128)
		values = value_head(hidden_states)

		assert values.shape == (0,)


	def test_value_head_long_sequence(self):
		"""Test with long sequence."""
		value_head = ValueHead(hidden_dim=64)

		hidden_states = torch.randn(2, 1000, 64)
		values = value_head(hidden_states)

		assert values.shape == (2, 1000)


	def test_value_head_minimal_config(self):
		"""Test with minimal configuration."""
		config = {'hidden_dim': 32}
		value_head = ValueHead.from_config(config)

		hidden_states = torch.randn(4, 32)
		values = value_head(hidden_states)

		assert values.shape == (4,)


	def test_value_head_large_hidden_dim(self):
		"""Test with large hidden dimension."""
		value_head = ValueHead(hidden_dim=2048, intermediate_dim=1024, bottleneck_dim=256)

		hidden_states = torch.randn(2, 2048)
		values = value_head(hidden_states)

		assert values.shape == (2,)


class TestValueHeadConsistency:
	"""Test consistency and determinism."""

	def test_value_head_deterministic(self):
		"""Test that forward pass is deterministic."""
		torch.manual_seed(42)
		value_head = ValueHead(hidden_dim=128)
		value_head.eval()

		hidden_states = torch.randn(4, 128)

		# First pass
		with torch.no_grad():
			values1 = value_head(hidden_states)

		# Second pass
		with torch.no_grad():
			values2 = value_head(hidden_states)

		assert torch.allclose(values1, values2)


	def test_value_head_eval_mode(self):
		"""Test evaluation mode (no dropout)."""
		torch.manual_seed(42)
		value_head = ValueHead(hidden_dim=128, dropout=0.5)

		hidden_states = torch.randn(8, 128)

		# Training mode
		value_head.train()
		values_train = value_head(hidden_states)

		# Eval mode
		value_head.eval()
		with torch.no_grad():
			values_eval1 = value_head(hidden_states)
			values_eval2 = value_head(hidden_states)

		# Eval mode should be deterministic
		assert torch.allclose(values_eval1, values_eval2)
