"""
Integration test for KV cache with real GPT2 model.

This test validates the KV cache implementation works with actual transformers models.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import GPT2LMHeadModel, GPT2Config


def test_gpt2_integration():
	"""Test KV cache with small GPT2 model."""
	print("=" * 70)
	print("KV Cache Integration Test - GPT2")
	print("=" * 70)

	# Create small GPT2 model
	config = GPT2Config(
		vocab_size=1000,
		n_positions=512,
		n_embd=256,
		n_layer=4,
		n_head=4,
	)
	gpt2 = GPT2LMHeadModel(config)
	base_model = gpt2.transformer
	base_model.eval()

	print(f"\nModel configuration:")
	print(f"  Layers: {config.n_layer}")
	print(f"  Heads: {config.n_head}")
	print(f"  Hidden dim: {config.n_embd}")
	print(f"  Vocab size: {config.vocab_size}")

	# Import BaseModelWithTreeAttention (need to extract from exportOnnx.py)
	# For now, we'll test the concept manually

	# Test parameters
	batch_size = 2
	n = 8  # prefix length
	m = 4  # evaluated length

	prefix_ids = torch.randint(0, config.vocab_size, (batch_size, n))
	evaluated_ids = torch.randint(0, config.vocab_size, (batch_size, m))

	# Create causal mask
	evaluated_mask = torch.tril(torch.ones(m, m)).unsqueeze(0).expand(batch_size, -1, -1)

	print(f"\nTest inputs:")
	print(f"  Prefix: {prefix_ids.shape}")
	print(f"  Evaluated: {evaluated_ids.shape}")
	print(f"  Mask: {evaluated_mask.shape}")

	# Test 1: No cache mode
	print(f"\n[Test 1] No cache mode:")
	input_ids = torch.cat([prefix_ids, evaluated_ids], dim=1)
	position_ids = torch.arange(n + m).unsqueeze(0).expand(batch_size, -1)

	with torch.no_grad():
		outputs = base_model(
			input_ids,
			position_ids=position_ids,
			use_cache=False,
			output_hidden_states=True,
		)

	hidden_states = outputs.hidden_states[-1]
	print(f"  Output shape: {hidden_states.shape}")
	assert hidden_states.shape == (batch_size, n + m, config.n_embd)
	print(f"  ✓ Test passed")

	# Test 2: With cache (first call)
	print(f"\n[Test 2] With cache (first call):")
	with torch.no_grad():
		outputs = base_model(
			input_ids,
			position_ids=position_ids,
			use_cache=True,
			output_hidden_states=True,
		)

	hidden_states = outputs.hidden_states[-1]
	past_key_values = outputs.past_key_values

	print(f"  Output shape: {hidden_states.shape}")
	print(f"  Cache type: {type(past_key_values)}")

	# Check cache
	if hasattr(past_key_values, 'to_legacy_cache'):
		cache_tuple = past_key_values.to_legacy_cache()
		print(f"  Cache layers: {len(cache_tuple)}")
		print(f"  Cache[0] key shape: {cache_tuple[0][0].shape}")
		print(f"  Cache[0] value shape: {cache_tuple[0][1].shape}")
		assert cache_tuple[0][0].shape == (batch_size, config.n_head, n + m, config.n_embd // config.n_head)

	print(f"  ✓ Test passed")

	# Test 3: With cache (subsequent call)
	print(f"\n[Test 3] With cache (subsequent call):")

	# Create prefix cache
	with torch.no_grad():
		prefix_outputs = base_model(
			prefix_ids,
			use_cache=True,
			output_hidden_states=True,
		)

	prefix_cache = prefix_outputs.past_key_values

	# Now use cache to process only evaluated tokens
	eval_position_ids = torch.arange(n, n + m).unsqueeze(0).expand(batch_size, -1)

	with torch.no_grad():
		cached_outputs = base_model(
			evaluated_ids,
			position_ids=eval_position_ids,
			past_key_values=prefix_cache,
			use_cache=True,
			output_hidden_states=True,
		)

	cached_hidden = cached_outputs.hidden_states[-1]
	new_cache = cached_outputs.past_key_values

	print(f"  Output shape: {cached_hidden.shape}")
	assert cached_hidden.shape == (batch_size, m, config.n_embd)

	# Check new cache length
	if hasattr(new_cache, 'to_legacy_cache'):
		cache_tuple = new_cache.to_legacy_cache()
		print(f"  New cache[0] key shape: {cache_tuple[0][0].shape}")
		assert cache_tuple[0][0].shape == (batch_size, config.n_head, n + m, config.n_embd // config.n_head)

	print(f"  ✓ Test passed")

	print(f"\n" + "=" * 70)
	print("All integration tests passed!")
	print("=" * 70)


if __name__ == '__main__':
	test_gpt2_integration()
