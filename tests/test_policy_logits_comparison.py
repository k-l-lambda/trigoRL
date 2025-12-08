"""
Compare C++ and Python policy logits for the same input.

This test validates that the C++ PrefixCacheInferencer produces
identical policy logits to the Python/ONNX implementation.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_policy_logits_comparison():
	"""Compare C++ vs Python policy logits."""
	print("=" * 80)
	print("Policy Logits Comparison: C++ vs Python")
	print("=" * 80)

	# Test configuration (same as C++ test)
	model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached"

	# Test input: Empty 5x5 board
	tgn_text = "[Board 5x5]\n\n"
	moves = ["aa", "ab", "a0", "ay", "az"]

	print(f"\nModel: {model_dir}")
	print(f"TGN: {repr(tgn_text)}")
	print(f"Moves: {moves}")
	print()

	# Initialize tokenizer
	tokenizer = TGNTokenizer()

	# Tokenize prefix (game state)
	prefix_tensor = tokenizer.encode(tgn_text, max_length=8192,
	                                  add_special_tokens=True,
	                                  add_value_token=False,
	                                  padding=False,
	                                  truncation=True)
	prefix_tokens = prefix_tensor.tolist()
	prefix_len = len(prefix_tokens)

	print(f"Prefix tokens: {prefix_tokens}")
	print(f"Prefix length: {prefix_len}")
	print()

	# Build candidate sequences
	candidate_sequences = []
	for move in moves:
		seq = prefix_tokens.copy()
		move_tensor = tokenizer.encode(move, max_length=2048,
		                                add_special_tokens=False,
		                                add_value_token=False,
		                                padding=False,
		                                truncation=True)
		seq.extend(move_tensor.tolist())
		candidate_sequences.append(seq)

	print("Candidate sequences:")
	for i, (move, seq) in enumerate(zip(moves, candidate_sequences)):
		print(f"  {move}: {seq}")
	print()

	# Build prefix tree
	from trigor.inference.prefixTreeBuilder import build_prefix_tree
	tree = build_prefix_tree(candidate_sequences)

	print(f"Tree structure:")
	print(f"  Num nodes: {tree['num_nodes']}")
	print(f"  Evaluated IDs shape: {tree['evaluated_ids'].shape}")
	print(f"  Evaluated mask shape: {tree['evaluated_mask'].shape}")
	print(f"  Move to leaf: {tree['move_to_leaf']}")
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

	# STEP 1: Compute prefix cache
	print("[STEP 2] Computing prefix cache...")
	prefix_ids_np = np.array([prefix_tokens], dtype=np.int64)  # [1, prefix_len]

	prefix_outputs = prefix_session.run(None, {'input_ids': prefix_ids_np})

	# Extract cache tensors (cache_key_0, cache_value_0, ...)
	cached_keys = []
	cached_values = []
	for i in range(0, len(prefix_outputs), 2):
		cached_keys.append(prefix_outputs[i])
		cached_values.append(prefix_outputs[i + 1])

	print(f"  ✓ Cache computed")
	print(f"  Num layers: {len(cached_keys)}")
	print(f"  Cache shape: {cached_keys[0].shape}")
	print()

	# STEP 2: Evaluate with cache
	print("[STEP 3] Evaluating with cache...")

	# Prepare inputs for eval_cached model
	eval_inputs = {
		'evaluated_ids': tree['evaluated_ids'].numpy(),
		'evaluated_mask': tree['evaluated_mask'].numpy(),
	}

	# Add cache tensors
	for i in range(len(cached_keys)):
		eval_inputs[f'past_key_{i}'] = cached_keys[i]
		eval_inputs[f'past_value_{i}'] = cached_values[i]

	# Run eval_cached model
	hidden_states = eval_cached_session.run(None, eval_inputs)[0]

	hidden_dim = hidden_states.shape[2]
	print(f"  ✓ Hidden states computed")
	print(f"  Hidden states shape: {hidden_states.shape}")
	print(f"  Hidden dim: {hidden_dim}")
	print()

	# STEP 3: Run policy head
	print("[STEP 4] Running policy head...")

	logits = policy_session.run(None, {'hidden_states': hidden_states})[0]

	vocab_size = logits.shape[2]
	print(f"  ✓ Policy logits computed")
	print(f"  Logits shape: {logits.shape}")
	print(f"  Vocab size: {vocab_size}")
	print()

	# Extract move logits
	print("[PYTHON RESULT] Policy head logits:")
	python_logits = []
	for i, move in enumerate(moves):
		leaf_pos = tree['move_to_leaf'][i]
		last_token = candidate_sequences[i][-1]

		# Get logit for last token at leaf position
		logit = logits[0, leaf_pos, last_token]
		python_logits.append(logit)

		print(f"  {move}: {logit:.4f}")
	print()

	# Compare with C++ output
	print("[C++ RESULT] (from previous test):")
	cpp_logits = {
		'aa': 5.5733,
		'ab': 5.0475,
		'a0': 5.0486,
		'ay': 5.1323,
		'az': 5.6256,
	}

	for move, expected in cpp_logits.items():
		print(f"  {move}: {expected:.4f}")
	print()

	# Calculate differences
	print("[COMPARISON]")
	max_diff = 0.0
	for i, move in enumerate(moves):
		py_logit = python_logits[i]
		cpp_logit = cpp_logits[move]
		diff = abs(py_logit - cpp_logit)
		max_diff = max(max_diff, diff)

		status = "✓" if diff < 0.001 else "✗"
		print(f"  {move}: Python={py_logit:.4f}, C++={cpp_logit:.4f}, diff={diff:.6f} {status}")

	print()
	print(f"Maximum difference: {max_diff:.6f}")

	if max_diff < 0.001:
		print("✓ SUCCESS: C++ and Python produce identical logits!")
	else:
		print("✗ FAILURE: C++ and Python logits differ!")

	return max_diff < 0.001


if __name__ == "__main__":
	success = test_policy_logits_comparison()
	sys.exit(0 if success else 1)
