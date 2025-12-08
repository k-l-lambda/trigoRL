"""
Compare C++ and Python policy logits for the same input (simplified version).

This test validates that the C++ PrefixCacheInferencer produces
identical policy logits to the Python/ONNX implementation.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_policy_logits_simple():
	"""Compare C++ vs Python policy logits using simple sequential evaluation."""
	print("=" * 80)
	print("Policy Logits Comparison: C++ vs Python (Simplified)")
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

	# Tokenize prefix (game state) - WITHOUT end token to match C++
	prefix_tensor = tokenizer.encode(tgn_text, max_length=8192,
	                                  add_special_tokens=False,  # Don't add START/END
	                                  add_value_token=False,
	                                  padding=False,
	                                  truncation=True)
	# Manually add START token to match C++
	prefix_tokens = [1] + prefix_tensor.tolist()  # START token = 1
	prefix_len = len(prefix_tokens)

	print(f"Prefix tokens: {prefix_tokens}")
	print(f"Prefix length: {prefix_len}")
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
	prefix_ids_np = np.array([prefix_tokens], dtype=np.int64)  # [1, prefix_len]

	prefix_outputs = prefix_session.run(None, {'prefix_ids': prefix_ids_np})

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

	# STEP 3: Evaluate each move individually
	print("[STEP 3] Evaluating moves with cache...")
	python_logits = {}

	for move in moves:
		# Tokenize move
		move_tensor = tokenizer.encode(move, max_length=2048,
		                                add_special_tokens=False,
		                                add_value_token=False,
		                                padding=False,
		                                truncation=True)
		move_tokens = move_tensor.tolist()

		# Create evaluated_ids: just the move tokens [1, num_move_tokens]
		evaluated_ids = np.array([move_tokens], dtype=np.int64)
		num_tokens = len(move_tokens)

		# Create trivial evaluated_mask: lower triangular [1, num_tokens, num_tokens]
		mask = np.tril(np.ones((num_tokens, num_tokens), dtype=np.float32))
		evaluated_mask = mask[np.newaxis, :, :]  # [1, num_tokens, num_tokens]

		# Prepare inputs for eval_cached model
		eval_inputs = {
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}

		# Add cache tensors
		for i in range(len(cached_keys)):
			eval_inputs[f'past_key_{i}'] = cached_keys[i]
			eval_inputs[f'past_value_{i}'] = cached_values[i]

		# Run eval_cached model
		hidden_states = eval_cached_session.run(None, eval_inputs)[0]

		# Run policy head
		logits = policy_session.run(None, {'hidden_states': hidden_states})[0]

		# Extract logit for last token
		last_token = move_tokens[-1]
		last_position = num_tokens - 1
		logit = logits[0, last_position, last_token]

		python_logits[move] = logit

		print(f"  {move}: tokens={move_tokens}, last_token={last_token}, logit={logit:.4f}")

	print()

	# Compare with C++ output
	print("[PYTHON RESULT] Policy head logits:")
	for move in moves:
		print(f"  {move}: {python_logits[move]:.4f}")
	print()

	print("[C++ RESULT] (from previous test):")
	cpp_logits = {
		'aa': 5.5733,
		'ab': 5.0475,
		'a0': 5.0486,
		'ay': 5.1323,
		'az': 5.6256,
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
		return False


if __name__ == "__main__":
	success = test_policy_logits_simple()
	sys.exit(0 if success else 1)
