"""
Benchmark for MCTS prefix cache with redesigned architecture.

Tests the actual MCTS pattern:
1. Compute prefix once with prefix_only model
2. Reuse fixed cache for multiple evaluated sequences with eval_cached model
3. Measure speedup vs standard model

Usage:
    python tests/benchmark_prefix_cache_final.py <onnx_model_dir>

Example:
    python tests/benchmark_prefix_cache_final.py outputs/trigor/.../GPT2CausalLM_ep0019_shared_cached
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def benchmark_prefix_cache(model_dir: str, n_iterations: int = 20, n_evaluations: int = 10):
	"""
	Benchmark MCTS prefix cache pattern.

	Args:
	    model_dir: Directory containing the 5 ONNX models
	    n_iterations: Number of benchmark iterations
	    n_evaluations: Number of move evaluations per MCTS iteration

	Returns:
	    Dict with benchmark results
	"""
	print("=" * 80)
	print("MCTS Prefix Cache Benchmark - Redesigned Architecture")
	print("=" * 80)

	model_dir = Path(model_dir)
	if not model_dir.exists():
		raise FileNotFoundError(f"Model directory not found: {model_dir}")

	# Check that all required models exist
	required_models = [
		'base_model.onnx',
		'base_model_prefix.onnx',
		'base_model_eval_cached.onnx',
	]

	for model_name in required_models:
		model_path = model_dir / model_name
		if not model_path.exists():
			raise FileNotFoundError(f"Required model not found: {model_path}")

	print(f"\nModel directory: {model_dir}")
	print(f"Benchmark parameters:")
	print(f"  Iterations: {n_iterations}")
	print(f"  Evaluations per iteration: {n_evaluations}")

	# Load models
	print("\n[1/3] Loading ONNX models...")
	sess_standard = ort.InferenceSession(str(model_dir / 'base_model.onnx'))
	sess_prefix = ort.InferenceSession(str(model_dir / 'base_model_prefix.onnx'))
	sess_eval_cached = ort.InferenceSession(str(model_dir / 'base_model_eval_cached.onnx'))

	print(f"  ✓ Standard model loaded")
	print(f"  ✓ Prefix-only model loaded")
	print(f"  ✓ Eval-cached model loaded")

	# Get model info
	prefix_input = sess_prefix.get_inputs()[0]
	eval_inputs = {inp.name: inp for inp in sess_eval_cached.get_inputs()}

	prefix_len = prefix_input.shape[1] if isinstance(prefix_input.shape[1], int) else 128
	eval_len = eval_inputs['evaluated_ids'].shape[1] if isinstance(eval_inputs['evaluated_ids'].shape[1], int) else 64

	print(f"\nModel configuration:")
	print(f"  Prefix length: {prefix_len}")
	print(f"  Evaluated length: {eval_len}")

	# Create test inputs
	batch_size = 1
	vocab_size = 128  # TrigoRL model vocab size

	prefix_ids = np.random.randint(0, vocab_size, (batch_size, prefix_len), dtype=np.int64)
	evaluated_sequences = [
		np.random.randint(0, vocab_size, (batch_size, eval_len), dtype=np.int64)
		for _ in range(n_evaluations)
	]

	# Create causal mask for evaluated tokens
	evaluated_mask = np.tril(np.ones((eval_len, eval_len), dtype=np.float32))
	evaluated_mask = np.expand_dims(evaluated_mask, 0)  # [1, eval_len, eval_len]

	print(f"\nTest inputs:")
	print(f"  Prefix: {prefix_ids.shape}")
	print(f"  Evaluated sequences: {n_evaluations} × {evaluated_sequences[0].shape}")
	print(f"  Evaluated mask: {evaluated_mask.shape}")

	# ========================================================================
	# Benchmark 1: Standard model (no cache)
	# ========================================================================
	print("\n[2/3] Benchmarking Standard Model (no cache)...")
	print(f"  Pattern: Compute full sequence for each evaluation")

	# Warm up
	for _ in range(3):
		sess_standard.run(None, {
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_sequences[0],
			'evaluated_mask': evaluated_mask,
		})

	# Benchmark
	times_standard = []
	for i in range(n_iterations):
		start = time.time()

		# Process each evaluation independently (full sequence each time)
		for eval_seq in evaluated_sequences:
			sess_standard.run(None, {
				'prefix_ids': prefix_ids,
				'evaluated_ids': eval_seq,
				'evaluated_mask': evaluated_mask,
			})

		elapsed = (time.time() - start) * 1000  # ms
		times_standard.append(elapsed)

	avg_time_standard = np.mean(times_standard)
	std_time_standard = np.std(times_standard)

	print(f"  Total time: {avg_time_standard:.2f} ± {std_time_standard:.2f} ms ({n_evaluations} evaluations)")
	print(f"  Per evaluation: {avg_time_standard / n_evaluations:.2f} ms")

	# ========================================================================
	# Benchmark 2: Prefix cache (prefix_only + eval_cached)
	# ========================================================================
	print("\n[3/3] Benchmarking Prefix Cache (prefix + eval_cached)...")
	print(f"  Pattern: Compute prefix once, reuse cache for all evaluations")

	# Warm up
	for _ in range(3):
		# Compute prefix
		cache_outputs = sess_prefix.run(None, {'prefix_ids': prefix_ids})

		# Build cache dict
		cache_dict = {}
		for i in range(len(cache_outputs) // 2):
			cache_dict[f'past_key_{i}'] = cache_outputs[i * 2]
			cache_dict[f'past_value_{i}'] = cache_outputs[i * 2 + 1]

		# Evaluate with cache
		sess_eval_cached.run(None, {
			'evaluated_ids': evaluated_sequences[0],
			'evaluated_mask': evaluated_mask,
			**cache_dict
		})

	# Benchmark
	times_cached = []
	for i in range(n_iterations):
		start = time.time()

		# Step 1: Compute prefix once
		cache_outputs = sess_prefix.run(None, {'prefix_ids': prefix_ids})

		# Build cache dict
		cache_dict = {}
		for j in range(len(cache_outputs) // 2):
			cache_dict[f'past_key_{j}'] = cache_outputs[j * 2]
			cache_dict[f'past_value_{j}'] = cache_outputs[j * 2 + 1]

		# Step 2: Evaluate multiple sequences with same cache
		for eval_seq in evaluated_sequences:
			sess_eval_cached.run(None, {
				'evaluated_ids': eval_seq,
				'evaluated_mask': evaluated_mask,
				**cache_dict  # Reuse same cache
			})

		elapsed = (time.time() - start) * 1000  # ms
		times_cached.append(elapsed)

	avg_time_cached = np.mean(times_cached)
	std_time_cached = np.std(times_cached)

	print(f"  Total time: {avg_time_cached:.2f} ± {std_time_cached:.2f} ms")
	print(f"    - Prefix computation: ~{avg_time_cached / (n_evaluations + 1):.2f} ms (once)")
	print(f"    - Per evaluation (avg): ~{avg_time_cached / n_evaluations:.2f} ms")

	# ========================================================================
	# Results Analysis
	# ========================================================================
	speedup = avg_time_standard / avg_time_cached

	print("\n" + "=" * 80)
	print("Results Summary")
	print("=" * 80)
	print(f"  MCTS Pattern: {n_evaluations} evaluations with shared prefix")
	print(f"  Prefix length: {prefix_len}, Evaluated length: {eval_len}")
	print(f"")
	print(f"  Standard (no cache):  {avg_time_standard:.2f} ± {std_time_standard:.2f} ms")
	print(f"  With prefix cache:    {avg_time_cached:.2f} ± {std_time_cached:.2f} ms")
	print(f"")
	print(f"  Speedup:              {speedup:.2f}×")
	print(f"  Time saved:           {avg_time_standard - avg_time_cached:.2f} ms ({100 * (1 - 1/speedup):.1f}%)")

	if speedup >= 2.0:
		print(f"")
		print(f"  ✓✓✓ EXCELLENT: Speedup exceeds target (>2×)")
	elif speedup >= 1.5:
		print(f"")
		print(f"  ✓✓ GOOD: Significant speedup achieved")
	elif speedup >= 1.2:
		print(f"")
		print(f"  ✓ FAIR: Moderate speedup achieved")
	else:
		print(f"")
		print(f"  ⚠ LIMITED: Speedup below expectations")

	print("=" * 80)

	return {
		'avg_time_standard': avg_time_standard,
		'std_time_standard': std_time_standard,
		'avg_time_cached': avg_time_cached,
		'std_time_cached': std_time_cached,
		'speedup': speedup,
		'n_evaluations': n_evaluations,
		'prefix_len': prefix_len,
		'eval_len': eval_len,
	}


def main():
	parser = argparse.ArgumentParser(description='Benchmark MCTS prefix cache')
	parser.add_argument(
		'model_dir',
		type=str,
		help='Path to directory containing ONNX models'
	)
	parser.add_argument(
		'--iterations',
		type=int,
		default=20,
		help='Number of benchmark iterations (default: 20)'
	)
	parser.add_argument(
		'--evaluations',
		type=int,
		default=10,
		help='Number of evaluations per MCTS iteration (default: 10)'
	)

	args = parser.parse_args()

	try:
		results = benchmark_prefix_cache(args.model_dir, args.iterations, args.evaluations)
		return 0
	except Exception as e:
		print(f"\nBenchmark failed: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == '__main__':
	sys.exit(main())
