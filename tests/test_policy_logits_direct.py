"""
Direct comparison of C++ policy logits using same evaluated_ids.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_policy_logits_direct():
	"""Compare using exact same evaluated_ids as C++."""
	print("=" * 80)
	print("Direct Policy Logits Comparison (Same evaluated_ids as C++)")
	print("=" * 80)

	model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached"

	# From C++ test output
	prefix_tokens = [1, 91, 66, 111, 97, 114, 100, 32, 53, 120, 53, 93, 10, 10]

	# C++ builds tree with full sequences, but ONNX eval_cached expects only evaluated part
	# evaluated_ids from C++: [1, 91, ..., 10, 10, 97, 48, 97, 98, 121, 122]
	# We need only the part after prefix (last 6 tokens): [97, 48, 97, 98, 121, 122]
	full_evaluated_ids = [1, 91, 66, 111, 97, 114, 100, 32, 53, 120, 53, 93, 10, 10, 97, 48, 97, 98, 121, 122]
	evaluated_ids = full_evaluated_ids[len(prefix_tokens):]  # Only diverging part

	# Move to leaf positions need to be adjusted (subtract prefix length)
	move_to_leaf_full = {
		'aa': 16,
		'ab': 17,
		'a0': 15,
		'ay': 18,
		'az': 19,
	}

	prefix_len = len(prefix_tokens)
	move_to_leaf = {k: v - prefix_len for k, v in move_to_leaf_full.items()}

	move_last_tokens = {
		'aa': 97,  # 'a'
		'ab': 98,  # 'b'
		'a0': 48,  # '0'
		'ay': 121, # 'y'
		'az': 122, # 'z'
	}

	num_nodes = len(evaluated_ids)

	print(f"\nPrefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
	print(f"Evaluated IDs ({num_nodes}): {evaluated_ids}")
	print(f"Move to leaf: {move_to_leaf}")
	print()

	# Load ONNX models
	print("[STEP 1] Loading ONNX models...")
	sess_options = ort.SessionOptions()
	sess_options.intra_op_num_threads = 4
	sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

	prefix_session = ort.InferenceSession(
		f"{model_dir}/base_model_prefix.onnx",
		sess_options,
		providers=['CPUExecutionProvider']
	)

	eval_cached_session = ort.InferenceSession(
		f"{model_dir}/base_model_eval_cached.onnx",
		sess_options,
		providers=['CPUExecutionProvider']
	)

	policy_session = ort.InferenceSession(
		f"{model_dir}/policy_head.onnx",
		sess_options,
		providers=['CPUExecutionProvider']
	)
	print("  ✓ Models loaded")
	print()

	# STEP 2: Compute prefix cache
	print("[STEP 2] Computing prefix cache...")
	prefix_ids_np = np.array([prefix_tokens], dtype=np.int64)

	prefix_outputs = prefix_session.run(None, {'prefix_ids': prefix_ids_np})

	# Extract cache tensors
	cached_keys = []
	cached_values = []
	for i in range(0, len(prefix_outputs), 2):
		cached_keys.append(prefix_outputs[i])
		cached_values.append(prefix_outputs[i + 1])

	print(f"  ✓ Cache computed ({len(cached_keys)} layers)")
	print(f"  Cache key[0] shape: {cached_keys[0].shape}")
	print(f"  Cache key[0] sample (first 5 elements): {cached_keys[0].flatten()[:5]}")
	print()

	# STEP 3: Evaluate with cache using tree structure
	print("[STEP 3] Evaluating with tree structure...")

	# Use evaluated_ids (which includes prefix + divergent paths)
	evaluated_ids_np = np.array([evaluated_ids], dtype=np.int64)

	# Create tree attention mask (matches C++ PrefixTreeBuilder)
	# From C++ output, the mask structure is:
	# Row 0: 1 0 0 0 0 0   (root 'a')
	# Row 1: 1 1 0 0 0 0   ('0' branch from root)
	# Row 2: 1 0 1 0 0 0   ('a' branch from root - "aa")
	# Row 3: 1 0 0 1 0 0   ('b' branch from root - "ab")
	# Row 4: 1 0 0 0 1 0   ('y' branch from root - "ay")
	# Row 5: 1 0 0 0 0 1   ('z' branch from root - "az")

	# All moves start with 'a' (position 0 = root)
	# Then they diverge to different second characters
	mask = np.array([
		[1, 0, 0, 0, 0, 0],  # pos 0: root 'a'
		[1, 1, 0, 0, 0, 0],  # pos 1: 'a' -> '0'
		[1, 0, 1, 0, 0, 0],  # pos 2: 'a' -> 'a'
		[1, 0, 0, 1, 0, 0],  # pos 3: 'a' -> 'b'
		[1, 0, 0, 0, 1, 0],  # pos 4: 'a' -> 'y'
		[1, 0, 0, 0, 0, 1],  # pos 5: 'a' -> 'z'
	], dtype=np.float32)
	evaluated_mask = mask[np.newaxis, :, :]

	print(f"  Evaluated mask shape: {evaluated_mask.shape}")
	print(f"  Tree attention mask (1=attend, 0=mask):")
	for i in range(num_nodes):
		print(f"    Row {i}: {evaluated_mask[0, i, :]}")
	print()

	# Prepare inputs
	eval_inputs = {
		'evaluated_ids': evaluated_ids_np,
		'evaluated_mask': evaluated_mask,
	}

	# Add cache tensors
	for i in range(len(cached_keys)):
		eval_inputs[f'past_key_{i}'] = cached_keys[i]
		eval_inputs[f'past_value_{i}'] = cached_values[i]

	# Run eval_cached model
	hidden_states = eval_cached_session.run(None, eval_inputs)[0]

	hidden_dim = hidden_states.shape[2]
	print(f"  ✓ Hidden states computed")
	print(f"  Shape: {hidden_states.shape}")
	print(f"  Hidden dim: {hidden_dim}")

	# Debug: Print a few hidden state values
	print(f"  Hidden states at position {move_to_leaf['ay']} ('ay', first 10 dims): {hidden_states[0, move_to_leaf['ay'], :10]}")
	print()

	# STEP 4: Run policy head
	print("[STEP 4] Running policy head...")

	logits = policy_session.run(None, {'hidden_states': hidden_states})[0]

	vocab_size = logits.shape[2]
	print(f"  ✓ Policy logits computed")
	print(f"  Shape: {logits.shape}")
	print(f"  Vocab size: {vocab_size}")
	print()

	# Extract move logits
	print("[PYTHON RESULT] Policy head logits:")
	python_logits = {}
	moves = ['aa', 'ab', 'a0', 'ay', 'az']

	# Debug: Print logits at all leaf positions
	print("\n  DEBUG: All tokens at leaf positions:")
	for move in moves:
		pos = move_to_leaf[move]
		token_id = evaluated_ids[pos]
		print(f"    {move} at position {pos}: token={token_id}, top logits:")
		# Get top 5 logits at this position
		pos_logits = logits[0, pos, :]
		top_indices = np.argsort(pos_logits)[-5:][::-1]
		for idx in top_indices:
			print(f"      token {idx}: {pos_logits[idx]:.4f}")

	print("\n  RESULT:")
	for move in moves:
		leaf_pos = move_to_leaf[move]
		last_token = move_last_tokens[move]

		# Get logit for last token at leaf position
		logit = logits[0, leaf_pos, last_token]
		python_logits[move] = logit

		print(f"  {move}: leaf_pos={leaf_pos}, last_token={last_token}, logit={logit:.4f}")
	print()

	# Compare with C++
	print("[C++ RESULT] (from test output):")
	cpp_logits = {
		'aa': 4.5719,
		'ab': 3.9626,
		'a0': 2.7860,
		'ay': 4.0158,
		'az': 4.6816,
	}

	for move in moves:
		print(f"  {move}: {cpp_logits[move]:.4f}")
	print()

	# Calculate differences
	print("[COMPARISON]")
	max_diff = 0.0
	for move in moves:
		py_logit = python_logits[move]
		cpp_logit = cpp_logits[move]
		diff = abs(py_logit - cpp_logit)
		max_diff = max(max_diff, diff)

		status = "✓" if diff < 0.001 else "✗"
		print(f"  {move}: Python={py_logit:.4f}, C++={cpp_logit:.4f}, diff={diff:.6f} {status}")

	print()
	print(f"Maximum difference: {max_diff:.6f}")

	if max_diff < 0.001:
		print("✓ SUCCESS: C++ and Python produce identical logits!")
		return True
	else:
		print("✗ FAILURE: C++ and Python logits differ!")
		print("\nPossible reasons:")
		print("  1. evaluated_mask construction differs")
		print("  2. Tree structure interpretation differs")
		print("  3. C++ logit extraction uses wrong index")
		return False


if __name__ == "__main__":
	success = test_policy_logits_direct()
	sys.exit(0 if success else 1)
