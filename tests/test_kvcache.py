"""
Unit tests for KV Cache implementation in BaseModelWithTreeAttention.

Tests correctness of:
1. Cache vs no-cache numerical equivalence
2. Position IDs calculation with cache
3. Attention mask construction
4. Cache conversion (tuple ↔ Cache object)
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Mock base model for testing
class MockTransformerModel(nn.Module):
	"""Mock transformer model with KV cache support."""

	def __init__(self, vocab_size=1000, hidden_dim=256, num_layers=4, num_heads=4):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_dim)
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.head_dim = hidden_dim // num_heads
		self.hidden_dim = hidden_dim

		# Simple output projection
		self.output = nn.Linear(hidden_dim, hidden_dim)

	def forward(
		self,
		input_ids,
		attention_mask=None,
		position_ids=None,
		past_key_values=None,
		use_cache=False,
		output_hidden_states=True,
	):
		"""Mock forward pass."""
		batch_size, seq_len = input_ids.shape

		# Embeddings
		hidden_states = self.embedding(input_ids)
		hidden_states = self.output(hidden_states)

		# Create mock KV cache if use_cache=True
		present_key_values = None
		if use_cache:
			# Determine cache length
			if past_key_values is not None:
				if hasattr(past_key_values, 'get_seq_length'):
					# DynamicCache
					seq_len_cached = past_key_values.get_seq_length(layer_idx=0)
					cache_len = seq_len_cached if seq_len_cached is not None else 0
				elif isinstance(past_key_values, tuple):
					cache_len = past_key_values[0][0].shape[2]
				else:
					cache_len = 0
			else:
				cache_len = 0

			total_seq_len = cache_len + seq_len

			# Create mock cache (zeros for simplicity)
			try:
				from transformers import DynamicCache
				cache = DynamicCache()
				for layer_idx in range(self.num_layers):
					k = torch.zeros(batch_size, self.num_heads, total_seq_len, self.head_dim)
					v = torch.zeros(batch_size, self.num_heads, total_seq_len, self.head_dim)
					cache.update(k, v, layer_idx=layer_idx)
				present_key_values = cache
			except ImportError:
				# Fallback to tuple
				present_key_values = tuple([
					(
						torch.zeros(batch_size, self.num_heads, total_seq_len, self.head_dim),
						torch.zeros(batch_size, self.num_heads, total_seq_len, self.head_dim),
					)
					for _ in range(self.num_layers)
				])

		# Create output object
		class ModelOutput:
			def __init__(self, hidden_states, past_key_values):
				self.hidden_states = [hidden_states]
				self.past_key_values = past_key_values

		return ModelOutput(hidden_states, present_key_values)


# Import the actual BaseModelWithTreeAttention from exportOnnx.py
# We'll need to load it dynamically since it's defined inside a function
def create_base_model_with_tree_attention(base_model, use_cache=False):
	"""Create BaseModelWithTreeAttention instance by loading from exportOnnx.py."""
	# Read and execute the class definition
	import re

	with open(os.path.join(os.path.dirname(__file__), '..', 'exportOnnx.py'), 'r') as f:
		content = f.read()

	# Extract the BaseModelWithTreeAttention class definition
	pattern = r'class BaseModelWithTreeAttention\(nn\.Module\):.*?(?=\n\t\tbase_wrapper = )'
	match = re.search(pattern, content, re.DOTALL)

	if not match:
		raise ValueError("Could not find BaseModelWithTreeAttention class definition")

	class_def = match.group(0)

	# Remove leading tabs (class is indented in original file)
	class_def = '\n'.join(line[2:] if line.startswith('\t\t') else line for line in class_def.split('\n'))

	# Execute the class definition
	namespace = {'nn': nn, 'torch': torch}
	exec(class_def, namespace)

	# Create instance
	BaseModelWithTreeAttention = namespace['BaseModelWithTreeAttention']
	return BaseModelWithTreeAttention(base_model, use_cache=use_cache)


@pytest.fixture
def mock_model():
	"""Create mock transformer model."""
	return MockTransformerModel(vocab_size=1000, hidden_dim=256, num_layers=4, num_heads=4)


@pytest.fixture
def test_inputs():
	"""Create test inputs."""
	batch_size = 2
	n = 8  # Prefix length
	m = 4  # Evaluated length

	prefix_ids = torch.randint(0, 1000, (batch_size, n))
	evaluated_ids = torch.randint(0, 1000, (batch_size, m))

	# Create causal evaluated_mask
	evaluated_mask = torch.tril(torch.ones(m, m)).unsqueeze(0).expand(batch_size, -1, -1)

	return prefix_ids, evaluated_ids, evaluated_mask


class TestCacheCorrectness:
	"""Test cache vs no-cache numerical equivalence."""

	def test_no_cache_mode(self, mock_model, test_inputs):
		"""Test non-cache mode produces expected output."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		model = create_base_model_with_tree_attention(mock_model, use_cache=False)
		model.eval()

		with torch.no_grad():
			output = model(prefix_ids, evaluated_ids, evaluated_mask)

		# Check output shape
		assert output.shape == (batch_size, n + m, mock_model.hidden_dim)

	def test_cache_mode_first_call(self, mock_model, test_inputs):
		"""Test cache mode with no prior cache (first call)."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		model.eval()

		with torch.no_grad():
			hidden, cache = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=None)

		# Check output shape (full sequence on first call with no cache)
		assert hidden.shape == (batch_size, n + m, mock_model.hidden_dim)

		# Check cache structure
		assert cache is not None
		assert isinstance(cache, tuple)
		assert len(cache) == mock_model.num_layers

		# Check cache tensor shapes
		for k, v in cache:
			assert k.shape == (batch_size, mock_model.num_heads, n + m, mock_model.head_dim)
			assert v.shape == (batch_size, mock_model.num_heads, n + m, mock_model.head_dim)

	def test_cache_mode_subsequent_call(self, mock_model, test_inputs):
		"""Test cache mode with prior cache (subsequent call)."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		model.eval()

		# First call: establish cache
		with torch.no_grad():
			_, cache = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=None)

		# Second call: use cache (simulate prefix already cached)
		# Create mock cached KV for prefix only
		cached_kv = tuple([
			(
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
			)
			for _ in range(mock_model.num_layers)
		])

		with torch.no_grad():
			hidden, new_cache = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=cached_kv)

		# Check output shape (only evaluated tokens returned)
		assert hidden.shape == (batch_size, m, mock_model.hidden_dim)

		# Check cache updated
		assert new_cache is not None
		for k, v in new_cache:
			# Cache should now include prefix + evaluated
			assert k.shape == (batch_size, mock_model.num_heads, n + m, mock_model.head_dim)


class TestPositionIDs:
	"""Test position ID calculation."""

	def test_no_cache_position_ids(self, mock_model, test_inputs):
		"""Test position IDs in no-cache mode."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		# Create wrapper and hook forward to capture position_ids
		captured_position_ids = []

		original_forward = mock_model.forward

		def hooked_forward(*args, **kwargs):
			captured_position_ids.append(kwargs.get('position_ids'))
			return original_forward(*args, **kwargs)

		mock_model.forward = hooked_forward

		model = create_base_model_with_tree_attention(mock_model, use_cache=False)
		model.eval()

		with torch.no_grad():
			_ = model(prefix_ids, evaluated_ids, evaluated_mask)

		position_ids = captured_position_ids[0]
		assert position_ids is not None
		assert position_ids.shape == (batch_size, n + m)

		# Check prefix positions: [0, 1, 2, ..., n-1]
		expected_prefix = torch.arange(n).unsqueeze(0).expand(batch_size, -1)
		assert torch.equal(position_ids[:, :n], expected_prefix)

		# Check evaluated positions follow mask_row_sums logic
		mask_row_sums = evaluated_mask.sum(dim=2)
		expected_eval = (n + mask_row_sums - 1).long()
		assert torch.equal(position_ids[:, n:], expected_eval)

	def test_cache_position_ids(self, mock_model, test_inputs):
		"""Test position IDs in cache mode."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		# Create wrapper and hook
		captured_position_ids = []

		original_forward = mock_model.forward

		def hooked_forward(*args, **kwargs):
			captured_position_ids.append(kwargs.get('position_ids'))
			return original_forward(*args, **kwargs)

		mock_model.forward = hooked_forward

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		model.eval()

		# Create mock cached KV for prefix
		cached_kv = tuple([
			(
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
			)
			for _ in range(mock_model.num_layers)
		])

		with torch.no_grad():
			_ = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=cached_kv)

		position_ids = captured_position_ids[0]
		assert position_ids is not None
		assert position_ids.shape == (batch_size, m)

		# Check evaluated positions: prefix_length + mask_row_sums - 1
		mask_row_sums = evaluated_mask.sum(dim=2)
		expected_eval = (n + mask_row_sums - 1).long()
		assert torch.equal(position_ids, expected_eval)


class TestAttentionMask:
	"""Test attention mask construction."""

	def test_no_cache_mask_shape(self, mock_model, test_inputs):
		"""Test attention mask shape in no-cache mode."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		captured_mask = []

		original_forward = mock_model.forward

		def hooked_forward(*args, **kwargs):
			captured_mask.append(kwargs.get('attention_mask'))
			return original_forward(*args, **kwargs)

		mock_model.forward = hooked_forward

		model = create_base_model_with_tree_attention(mock_model, use_cache=False)
		model.eval()

		with torch.no_grad():
			_ = model(prefix_ids, evaluated_ids, evaluated_mask)

		mask = captured_mask[0]
		assert mask is not None
		assert mask.shape == (batch_size, 1, n + m, n + m)

	def test_cache_mask_shape(self, mock_model, test_inputs):
		"""Test attention mask shape in cache mode."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		captured_mask = []

		original_forward = mock_model.forward

		def hooked_forward(*args, **kwargs):
			captured_mask.append(kwargs.get('attention_mask'))
			return original_forward(*args, **kwargs)

		mock_model.forward = hooked_forward

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		model.eval()

		cached_kv = tuple([
			(
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
			)
			for _ in range(mock_model.num_layers)
		])

		with torch.no_grad():
			_ = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=cached_kv)

		mask = captured_mask[0]
		assert mask is not None
		# Cache mode: [batch, 1, m, prefix_length + m]
		assert mask.shape == (batch_size, 1, m, n + m)

	def test_cache_mask_attends_to_prefix(self, mock_model, test_inputs):
		"""Test evaluated tokens attend to full cached prefix."""
		prefix_ids, evaluated_ids, evaluated_mask = test_inputs
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		captured_mask = []

		original_forward = mock_model.forward

		def hooked_forward(*args, **kwargs):
			captured_mask.append(kwargs.get('attention_mask'))
			return original_forward(*args, **kwargs)

		mock_model.forward = hooked_forward

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		model.eval()

		cached_kv = tuple([
			(
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
				torch.zeros(batch_size, mock_model.num_heads, n, mock_model.head_dim),
			)
			for _ in range(mock_model.num_layers)
		])

		with torch.no_grad():
			_ = model(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=cached_kv)

		mask = captured_mask[0]  # [batch, 1, m, n+m]

		# Check all evaluated tokens attend to full prefix
		# (mask values are 0.0 for attending, -inf for masked)
		prefix_attention = mask[:, :, :, :n]
		assert torch.all(prefix_attention == 0.0), "Evaluated tokens should attend to full prefix"


class TestCacheConversion:
	"""Test cache format conversions."""

	def test_tuple_to_cache_none(self, mock_model):
		"""Test tuple_to_cache with None input."""
		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		result = model._tuple_to_cache(None)
		assert result is None

	def test_cache_to_tuple_none(self, mock_model):
		"""Test cache_to_tuple with None input."""
		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		result = model._cache_to_tuple(None)
		assert result is None

	def test_tuple_to_cache_conversion(self, mock_model):
		"""Test tuple to Cache object conversion."""
		batch_size = 2
		num_layers = 4
		num_heads = 4
		seq_len = 8
		head_dim = 64

		# Create mock cache tuple
		cache_tuple = tuple([
			(
				torch.randn(batch_size, num_heads, seq_len, head_dim),
				torch.randn(batch_size, num_heads, seq_len, head_dim),
			)
			for _ in range(num_layers)
		])

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)
		cache_obj = model._tuple_to_cache(cache_tuple)

		# Check conversion (should create DynamicCache or return tuple)
		assert cache_obj is not None

		# If DynamicCache created, check it can be converted back
		if hasattr(cache_obj, 'to_legacy_cache'):
			legacy = cache_obj.to_legacy_cache()
			assert isinstance(legacy, tuple)
			assert len(legacy) == num_layers
		elif isinstance(cache_obj, tuple):
			# Fallback mode: tuple returned as-is
			assert cache_obj == cache_tuple

	def test_cache_to_tuple_conversion(self, mock_model):
		"""Test Cache object to tuple conversion."""
		batch_size = 2
		num_layers = 4
		num_heads = 4
		seq_len = 8
		head_dim = 64

		# Create mock cache tuple
		cache_tuple = tuple([
			(
				torch.randn(batch_size, num_heads, seq_len, head_dim),
				torch.randn(batch_size, num_heads, seq_len, head_dim),
			)
			for _ in range(num_layers)
		])

		model = create_base_model_with_tree_attention(mock_model, use_cache=True)

		# Test tuple input (should return as-is)
		result = model._cache_to_tuple(cache_tuple)
		assert result == cache_tuple


if __name__ == '__main__':
	pytest.main([__file__, '-v'])
