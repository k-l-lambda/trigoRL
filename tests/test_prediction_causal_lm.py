"""
Unit tests for PredictionCausalLM model.

Tests the prediction mode wrapper for ONNX export, including:
- Forward pass with custom attention masking
- Output slicing based on evaluated_ids
- Mask combination logic
- Helper functions for creating prediction masks
"""

import pytest
import torch
import torch.nn as nn

from trigor.models import (
	PredictionCausalLM,
	create_bidirectional_prediction_mask,
	create_diagonal_prediction_mask,
)


class DummyCausalLM(nn.Module):
	"""Dummy CausalLM for testing that returns predictable logits."""

	def __init__(self, vocab_size=100, hidden_size=64):
		super().__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.lm_head = nn.Linear(hidden_size, vocab_size)

	def forward(self, input_ids, attention_mask=None):
		# Simple forward that we can predict
		hidden = self.embedding(input_ids)
		logits = self.lm_head(hidden)

		# Return HuggingFace-style output
		from collections import namedtuple
		Output = namedtuple('Output', ['logits'])
		return Output(logits=logits)


@pytest.fixture
def dummy_model():
	"""Create a dummy causal LM model for testing."""
	return DummyCausalLM(vocab_size=100, hidden_size=64)


@pytest.fixture
def prediction_model(dummy_model):
	"""Create PredictionCausalLM wrapping dummy model."""
	return PredictionCausalLM(dummy_model)


class TestPredictionCausalLM:
	"""Test suite for PredictionCausalLM."""

	def test_initialization(self, dummy_model):
		"""Test model initialization."""
		model = PredictionCausalLM(dummy_model)
		assert model.model is dummy_model

	def test_basic_forward(self, prediction_model):
		"""Test basic forward pass returns correct shape."""
		batch_size = 2
		seq_len = 10
		prefix_len = 5

		# Create inputs
		input_ids = torch.randint(0, 100, (batch_size, seq_len))
		prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

		# Mark prediction region as evaluated
		evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
		evaluated_ids[:, prefix_len:] = 1

		# Forward pass
		logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

		# Check output shape
		num_evaluated = seq_len - prefix_len
		assert logits.shape == (batch_size, num_evaluated, 100)  # vocab_size=100

	def test_evaluated_ids_slicing(self, prediction_model):
		"""Test that evaluated_ids correctly slices output."""
		batch_size = 2
		seq_len = 20
		prefix_len = 10

		input_ids = torch.randint(0, 100, (batch_size, seq_len))
		prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

		# Evaluate only specific positions
		evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
		evaluated_ids[:, 12] = 1  # Position 12
		evaluated_ids[:, 15] = 1  # Position 15
		evaluated_ids[:, 18] = 1  # Position 18

		# Forward pass
		logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

		# Should return logits for only 3 positions
		assert logits.shape == (batch_size, 3, 100)

	def test_single_evaluated_position(self, prediction_model):
		"""Test with only one evaluated position."""
		batch_size = 1
		seq_len = 8
		prefix_len = 5

		input_ids = torch.randint(0, 100, (batch_size, seq_len))
		prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

		# Evaluate only position 6
		evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
		evaluated_ids[:, 6] = 1

		# Forward pass
		logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

		assert logits.shape == (batch_size, 1, 100)

	def test_different_batch_sizes(self, prediction_model):
		"""Test various batch sizes."""
		seq_len = 16
		prefix_len = 8

		for batch_size in [1, 4, 8]:
			input_ids = torch.randint(0, 100, (batch_size, seq_len))
			prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
			prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

			evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
			evaluated_ids[:, prefix_len:] = 1

			logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

			num_evaluated = seq_len - prefix_len
			assert logits.shape == (batch_size, num_evaluated, 100)

	def test_different_sequence_lengths(self, prediction_model):
		"""Test various sequence lengths."""
		batch_size = 2

		for seq_len in [10, 50, 100]:
			prefix_len = seq_len // 2

			input_ids = torch.randint(0, 100, (batch_size, seq_len))
			prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
			prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

			evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
			evaluated_ids[:, prefix_len:] = 1

			logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

			num_evaluated = seq_len - prefix_len
			assert logits.shape == (batch_size, num_evaluated, 100)

	def test_mask_broadcasting(self, prediction_model):
		"""Test that prediction_mask with batch_size=1 broadcasts correctly."""
		batch_size = 4
		seq_len = 12
		prefix_len = 6

		input_ids = torch.randint(0, 100, (batch_size, seq_len))

		# Create mask with batch_size=1 (should broadcast)
		prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		assert prediction_mask.shape == (1, seq_len, seq_len)

		evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
		evaluated_ids[:, prefix_len:] = 1

		# Should not raise error
		logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

		assert logits.shape == (batch_size, seq_len - prefix_len, 100)

	def test_model_info(self, prediction_model):
		"""Test get_model_info method."""
		info = prediction_model.get_model_info()

		assert 'model_class' in info
		assert info['model_class'] == 'PredictionCausalLM'
		assert info['mode'] == 'prediction'
		assert info['onnx_compatible'] is True

	def test_from_base_model(self, dummy_model):
		"""Test from_base_model class method."""
		model = PredictionCausalLM.from_base_model(dummy_model)

		assert isinstance(model, PredictionCausalLM)
		assert model.model is dummy_model

	def test_from_config(self, dummy_model):
		"""Test from_config class method."""
		config = {'test': 'config'}
		model = PredictionCausalLM.from_config(config, dummy_model)

		assert isinstance(model, PredictionCausalLM)
		assert model.model is dummy_model


class TestPredictionMaskHelpers:
	"""Test helper functions for creating prediction masks."""

	def test_bidirectional_mask_shape(self):
		"""Test bidirectional mask has correct shape."""
		seq_len = 20
		prefix_len = 10

		mask = create_bidirectional_prediction_mask(seq_len, prefix_len)

		assert mask.shape == (1, seq_len, seq_len)
		assert mask.dtype == torch.float32

	def test_bidirectional_mask_pattern(self):
		"""Test bidirectional mask has correct attention pattern."""
		seq_len = 10
		prefix_len = 5

		mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		mask = mask.squeeze(0)  # Remove batch dim

		# Check prefix region (causal)
		for i in range(prefix_len):
			for j in range(seq_len):
				if j <= i:
					assert mask[i, j] == 1.0, f"Position [{i}, {j}] should be 1 (causal)"
				else:
					assert mask[i, j] == 0.0, f"Position [{i}, {j}] should be 0 (causal)"

		# Check prediction region (bidirectional within prediction)
		for i in range(prefix_len, seq_len):
			# Can attend to all prefix positions
			for j in range(prefix_len):
				assert mask[i, j] == 1.0, f"Position [{i}, {j}] should attend to prefix"

			# Can attend to all prediction positions
			for j in range(prefix_len, seq_len):
				assert mask[i, j] == 1.0, f"Position [{i}, {j}] should be bidirectional"

	def test_diagonal_mask_shape(self):
		"""Test diagonal mask has correct shape."""
		seq_len = 20
		prefix_len = 10

		mask = create_diagonal_prediction_mask(seq_len, prefix_len)

		assert mask.shape == (1, seq_len, seq_len)
		assert mask.dtype == torch.float32

	def test_diagonal_mask_pattern(self):
		"""Test diagonal mask has correct attention pattern."""
		seq_len = 10
		prefix_len = 5

		mask = create_diagonal_prediction_mask(seq_len, prefix_len)
		mask = mask.squeeze(0)

		# Check prefix region (causal)
		for i in range(prefix_len):
			for j in range(seq_len):
				if j <= i:
					assert mask[i, j] == 1.0
				else:
					assert mask[i, j] == 0.0

		# Check prediction region (diagonal - only self + prefix)
		for i in range(prefix_len, seq_len):
			# Can attend to all prefix positions
			for j in range(prefix_len):
				assert mask[i, j] == 1.0

			# Can only attend to self in prediction region
			for j in range(prefix_len, seq_len):
				if i == j:
					assert mask[i, j] == 1.0, f"Position [{i}, {j}] should attend to self"
				else:
					assert mask[i, j] == 0.0, f"Position [{i}, {j}] should not attend"

	def test_mask_device_placement(self):
		"""Test mask is created on correct device."""
		seq_len = 10
		prefix_len = 5

		# CPU
		mask = create_bidirectional_prediction_mask(seq_len, prefix_len, device=torch.device('cpu'))
		assert mask.device.type == 'cpu'

		# CUDA (if available)
		if torch.cuda.is_available():
			mask = create_bidirectional_prediction_mask(seq_len, prefix_len, device=torch.device('cuda:0'))
			assert mask.device.type == 'cuda'


class TestWithRealModel:
	"""Test PredictionCausalLM with real GPT-2 model (integration tests)."""

	@pytest.fixture
	def gpt2_config(self):
		"""Create minimal GPT-2 config for testing."""
		return {
			'type': 'GPT2CausalLM',
			'config': {
				'vocab_size': 259,
				'n_positions': 256,
				'n_embd': 64,
				'n_layer': 2,
				'n_head': 4,
				'activation_function': 'gelu',
				'resid_pdrop': 0.0,
				'embd_pdrop': 0.0,
				'attn_pdrop': 0.0,
			}
		}

	def test_with_gpt2_model(self, gpt2_config):
		"""Test PredictionCausalLM wrapping real GPT-2 model."""
		from trigor.models import make_model

		# Create GPT-2 model using make_model
		base_model = make_model('GPT2CausalLM', gpt2_config['config'])

		# Wrap in PredictionCausalLM
		prediction_model = PredictionCausalLM(base_model)
		prediction_model.eval()

		# Test forward pass
		batch_size = 2
		seq_len = 32
		prefix_len = 16

		input_ids = torch.randint(0, 259, (batch_size, seq_len))
		prediction_mask = create_bidirectional_prediction_mask(seq_len, prefix_len)
		prediction_mask = prediction_mask.expand(batch_size, seq_len, seq_len)

		evaluated_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
		evaluated_ids[:, prefix_len:] = 1

		# Forward pass
		with torch.no_grad():
			logits = prediction_model(input_ids, prediction_mask, evaluated_ids)

		# Check output
		num_evaluated = seq_len - prefix_len
		assert logits.shape == (batch_size, num_evaluated, 259)

		# Check logits are reasonable (not NaN, not all zeros)
		assert not torch.isnan(logits).any()
		assert torch.abs(logits).sum() > 0


if __name__ == '__main__':
	pytest.main([__file__, '-v'])
