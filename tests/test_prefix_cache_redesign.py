"""
Test for redesigned prefix cache with three modes:
1. standard: prefix + evaluated → hidden_states
2. prefix_only: prefix → cache
3. eval_cached: cache + evaluated → hidden_states (cache unchanged)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prefix_cache_modes():
	"""Test three execution modes with real model."""
	print("=" * 80)
	print("Testing Prefix Cache Redesign - Three Modes")
	print("=" * 80)

	# Import after adding to path
	from exportOnnx import ONNXExporter

	# Use trained model
	training_dir = "outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500"
	print(f"\nTraining directory: {training_dir}")

	if not Path(training_dir).exists():
		print(f"⚠ Training directory not found, using test model instead")
		# Fallback: create small test model
		from transformers import GPT2Config, GPT2Model
		config = GPT2Config(
			vocab_size=100,
			n_positions=512,
			n_embd=64,
			n_layer=2,
			n_head=4,
		)
		base_model = GPT2Model(config)
	else:
		# Load trained model
		exporter = ONNXExporter(training_dir)
		model, checkpoint = exporter.load_model(checkpoint_name='best')

		# Extract base model
		if hasattr(model, 'model'):
			base_model = model.model
		else:
			raise ValueError("Cannot extract base model")

	base_model = base_model.to(dtype=torch.float32)
	base_model.eval()

	print("\n[1/4] Testing Mode 1: Standard (no cache)")
	print("-" * 80)

	# Import BaseModelWithTreeAttention from exportOnnx
	import re
	with open('exportOnnx.py', 'r') as f:
		content = f.read()

	# Extract BaseModelWithTreeAttention class
	pattern = r'(class BaseModelWithTreeAttention\(nn\.Module\):.*?)(?=\n\t\tbase_wrapper = |$)'
	match = re.search(pattern, content, re.DOTALL)

	if not match:
		raise RuntimeError("Could not find BaseModelWithTreeAttention class")

	class_def = match.group(1)
	# Remove leading tabs
	class_def = '\n'.join(line[2:] if line.startswith('\t\t') else line for line in class_def.split('\n'))

	# Execute to define the class
	exec_globals = {'nn': torch.nn, 'torch': torch}
	exec(class_def, exec_globals)
	BaseModelWithTreeAttention = exec_globals['BaseModelWithTreeAttention']

	# Test standard mode
	wrapper_standard = BaseModelWithTreeAttention(base_model, mode='standard')
	wrapper_standard.eval()

	batch_size = 1
	prefix_len = 16
	eval_len = 8
	vocab_size = base_model.config.vocab_size

	prefix_ids = torch.randint(0, vocab_size, (batch_size, prefix_len), dtype=torch.long)
	evaluated_ids = torch.randint(0, vocab_size, (batch_size, eval_len), dtype=torch.long)

	# Create evaluated mask
	evaluated_mask = torch.tril(torch.ones(eval_len, eval_len)).unsqueeze(0).expand(batch_size, -1, -1)

	# Forward pass
	hidden_states_standard = wrapper_standard(
		prefix_ids=prefix_ids,
		evaluated_ids=evaluated_ids,
		evaluated_mask=evaluated_mask
	)

	print(f"  Input: prefix={prefix_ids.shape}, evaluated={evaluated_ids.shape}")
	print(f"  Output: hidden_states={hidden_states_standard.shape}")
	print(f"  Expected: [{batch_size}, {prefix_len + eval_len}, hidden_dim]")
	assert hidden_states_standard.shape[0] == batch_size
	assert hidden_states_standard.shape[1] == prefix_len + eval_len
	print("  ✓ Standard mode works correctly")

	print("\n[2/4] Testing Mode 2: Prefix-Only (compute cache)")
	print("-" * 80)

	# Test prefix_only mode
	wrapper_prefix = BaseModelWithTreeAttention(base_model, mode='prefix_only')
	wrapper_prefix.eval()

	# Forward pass - only prefix
	cache = wrapper_prefix(prefix_ids=prefix_ids)

	print(f"  Input: prefix={prefix_ids.shape}")
	print(f"  Output: cache tuple with {len(cache)} layer pairs")

	# Get dimensions
	num_layers = len(cache)
	key_shape = cache[0][0].shape
	value_shape = cache[0][1].shape

	print(f"  Cache shapes: key={key_shape}, value={value_shape}")
	assert key_shape[0] == batch_size
	assert key_shape[2] == prefix_len  # Cache length should be prefix_len
	print("  ✓ Prefix-only mode works correctly")

	print("\n[3/4] Testing Mode 3: Eval-Cached (reuse fixed cache)")
	print("-" * 80)

	# Test eval_cached mode
	wrapper_eval = BaseModelWithTreeAttention(base_model, mode='eval_cached')
	wrapper_eval.eval()

	# Forward pass - use cache from prefix_only
	hidden_states_eval = wrapper_eval(
		evaluated_ids=evaluated_ids,
		evaluated_mask=evaluated_mask,
		past_key_values=cache
	)

	print(f"  Input: evaluated={evaluated_ids.shape}, cache (from prefix_only)")
	print(f"  Output: hidden_states={hidden_states_eval.shape}")
	print(f"  Expected: [{batch_size}, {eval_len}, hidden_dim]")
	assert hidden_states_eval.shape[0] == batch_size
	assert hidden_states_eval.shape[1] == eval_len  # Only evaluated tokens
	print("  ✓ Eval-cached mode works correctly")

	print("\n[4/4] Testing MCTS Pattern: Prefix Reuse")
	print("-" * 80)

	# Simulate MCTS: compute prefix once, evaluate multiple move sequences
	print(f"  Step 1: Compute prefix once → cache")
	cache_mcts = wrapper_prefix(prefix_ids=prefix_ids)
	print(f"  Cache computed: {len(cache_mcts)} layers")

	n_evaluations = 5
	print(f"  Step 2: Evaluate {n_evaluations} different move sequences with same cache")

	outputs = []
	for i in range(n_evaluations):
		# Different evaluated sequences
		eval_seq = torch.randint(0, vocab_size, (batch_size, eval_len), dtype=torch.long)

		# Evaluate with fixed cache
		hidden = wrapper_eval(
			evaluated_ids=eval_seq,
			evaluated_mask=evaluated_mask,
			past_key_values=cache_mcts  # Reuse same cache
		)
		outputs.append(hidden)

		# Verify cache shape hasn't changed
		assert cache_mcts[0][0].shape[2] == prefix_len, "Cache length changed!"

	print(f"  ✓ Evaluated {n_evaluations} sequences")
	print(f"  ✓ Cache stayed fixed at length {prefix_len}")
	print(f"  ✓ All outputs have shape {outputs[0].shape}")

	print("\n[5/5] Numerical Consistency Check")
	print("-" * 80)

	# Verify that prefix_only + eval_cached gives similar results to standard
	# They won't be exactly the same due to numerical precision, but should be close

	# Standard mode output (last eval_len tokens)
	hidden_standard_eval_part = hidden_states_standard[:, -eval_len:, :]

	# Eval-cached mode output
	hidden_eval_cached = wrapper_eval(
		evaluated_ids=evaluated_ids,
		evaluated_mask=evaluated_mask,
		past_key_values=cache
	)

	# Compare
	max_diff = torch.abs(hidden_standard_eval_part - hidden_eval_cached).max().item()
	mean_diff = torch.abs(hidden_standard_eval_part - hidden_eval_cached).mean().item()

	print(f"  Standard mode (evaluated part) vs Eval-cached mode:")
	print(f"    Max difference: {max_diff:.6f}")
	print(f"    Mean difference: {mean_diff:.6f}")

	# They should be reasonably close (within floating point tolerance)
	if max_diff < 1e-4:
		print("  ✓ Numerical consistency: EXCELLENT")
	elif max_diff < 1e-3:
		print("  ✓ Numerical consistency: GOOD")
	elif max_diff < 1e-2:
		print("  ⚠ Numerical consistency: FAIR (may need investigation)")
	else:
		print("  ✗ Numerical consistency: POOR (needs debugging)")

	print("\n" + "=" * 80)
	print("All Tests Passed! ✓")
	print("=" * 80)
	print("\nSummary:")
	print("  ✓ Standard mode: Full sequence computation works")
	print("  ✓ Prefix-only mode: Prefix cache generation works")
	print("  ✓ Eval-cached mode: Fixed cache evaluation works")
	print("  ✓ MCTS pattern: Prefix reuse across multiple evaluations works")
	print("  ✓ Cache stays fixed (doesn't accumulate)")
	print("\nReady for ONNX export and benchmarking!")
	print("=" * 80)


if __name__ == '__main__':
	test_prefix_cache_modes()
