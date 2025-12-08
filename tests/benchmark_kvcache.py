"""
Performance benchmark for KV cache with trained trigoRL models.

This benchmark measures the speedup from using KV cache in the MCTS use case
where we reuse a prefix (game state context) while evaluating multiple
different evaluated sequences (possible move combinations).

Usage:
    python tests/benchmark_kvcache.py <training_dir>

Example:
    python tests/benchmark_kvcache.py outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exportOnnx import ONNXExporter


def benchmark_kvcache(training_dir: str, n_iterations: int = 20):
	"""
	Benchmark KV cache performance with a trained model.

	Args:
	    training_dir: Path to training output directory
	    n_iterations: Number of iterations for benchmarking

	Returns:
	    Dict with benchmark results
	"""
	print("=" * 80)
	print("KV Cache Performance Benchmark")
	print("=" * 80)

	training_dir = Path(training_dir)
	if not training_dir.exists():
		raise FileNotFoundError(f"Training directory not found: {training_dir}")

	print(f"\nTraining directory: {training_dir}")

	# Create exporter
	exporter = ONNXExporter(str(training_dir))

	# Load model
	print("\n[1/5] Loading model from checkpoint...")
	model, checkpoint = exporter.load_model(checkpoint_name='best')

	print(f"  Model type: {exporter.config.model.type}")
	print(f"  Epoch: {checkpoint['epoch']}")
	print(f"  Global step: {checkpoint['global_step']}")

	# Export without cache (if not already exported)
	print("\n[2/5] Exporting models (no cache)...")
	output_dir_no_cache = training_dir / "benchmark_no_cache"
	output_dir_no_cache.mkdir(exist_ok=True)

	base_path, policy_path, value_path = exporter.export_shared_architecture(
		model=model,
		output_dir=str(output_dir_no_cache),
		batch_size=1,
		prefix_len=128,
		eval_len=64,
		seq_len=256,
		dynamic_batch=False,
		dynamic_n=False,
		dynamic_m=False,
		dynamic_seq=False,
		opset_version=18,
		with_cache=False,
	)

	print(f"  ✓ Base model: {Path(base_path).stat().st_size / (1024*1024):.2f} MB")

	# Export with cache
	print("\n[3/5] Exporting models (with cache)...")
	output_dir_with_cache = training_dir / "benchmark_with_cache"
	output_dir_with_cache.mkdir(exist_ok=True)

	base_cached_path, base_cached_cached_path, policy_cached_path, value_cached_path = exporter.export_shared_architecture(
		model=model,
		output_dir=str(output_dir_with_cache),
		batch_size=1,
		prefix_len=128,
		eval_len=64,
		seq_len=256,
		dynamic_batch=False,
		dynamic_n=False,
		dynamic_m=False,
		dynamic_seq=False,
		opset_version=18,
		with_cache=True,
	)

	print(f"  ✓ Base model (no cache): {Path(base_cached_path).stat().st_size / (1024*1024):.2f} MB")
	print(f"  ✓ Base model (cached): {Path(base_cached_cached_path).stat().st_size / (1024*1024):.2f} MB")

	# Benchmark no-cache model
	print("\n[4/5] Benchmarking no-cache model...")
	import onnxruntime as ort
	import torch

	from trigor.models import create_causal_evaluated_mask

	# Load no-cache model
	sess_no_cache = ort.InferenceSession(base_path)

	# Create test inputs (simulate MCTS use case)
	batch_size = 1
	prefix_len = 128
	eval_len = 64
	vocab_size = exporter.config.model.config.model_config.config.vocab_size

	# Fixed prefix (game state context)
	prefix_ids = torch.randint(0, vocab_size, (batch_size, prefix_len), dtype=torch.long)

	# Multiple evaluated sequences (different move combinations)
	n_eval_sequences = 10
	evaluated_sequences = [
		torch.randint(0, vocab_size, (batch_size, eval_len), dtype=torch.long)
		for _ in range(n_eval_sequences)
	]

	evaluated_mask = create_causal_evaluated_mask(eval_len).expand(batch_size, -1, -1)

	# Warm up
	for _ in range(3):
		sess_no_cache.run(None, {
			'prefix_ids': prefix_ids.numpy(),
			'evaluated_ids': evaluated_sequences[0].numpy(),
			'evaluated_mask': evaluated_mask.numpy(),
		})

	# Benchmark: Process each evaluated sequence with full prefix
	times_no_cache = []
	for i in range(n_iterations):
		start = time.time()
		for eval_seq in evaluated_sequences:
			sess_no_cache.run(None, {
				'prefix_ids': prefix_ids.numpy(),
				'evaluated_ids': eval_seq.numpy(),
				'evaluated_mask': evaluated_mask.numpy(),
			})
		elapsed = (time.time() - start) * 1000  # ms
		times_no_cache.append(elapsed)

	avg_time_no_cache = np.mean(times_no_cache)
	std_time_no_cache = np.std(times_no_cache)

	print(f"  No cache: {avg_time_no_cache:.2f} ± {std_time_no_cache:.2f} ms ({n_eval_sequences} sequences)")
	print(f"  Per sequence: {avg_time_no_cache / n_eval_sequences:.2f} ms")

	# Benchmark cache model
	print("\n[5/5] Benchmarking cached model...")
	print("  Note: Cached model requires prefix computation via standard model first")
	print("  This is a known limitation - cache mode doesn't separate prefix computation")
	print("  Skipping cache benchmark - see KVCACHE_EXPORT_STATUS.md for details")

	# The cached model has no prefix_ids input, only evaluated_ids + cache
	# This means we need a two-step process:
	# 1. Use standard base_model to compute prefix → get hidden states (but no cache output!)
	# 2. Use cached model with cache... but we can't get cache from standard model
	#
	# The current implementation doesn't support the MCTS use case properly:
	# - MCTS needs: compute prefix once, reuse for multiple evaluations
	# - Current cache: accumulates sequence (autoregressive generation pattern)
	#
	# Workaround: Use empty cache each time (no speedup)
	# OR: Wait for C++ implementation with proper cache management

	# For now, just document that cache model exists and can be loaded
	sess_cached = ort.InferenceSession(base_cached_cached_path)
	print(f"  ✓ Cached model loaded successfully")
	print(f"  ✓ Inputs: {[inp.name for inp in sess_cached.get_inputs()]}")
	print(f"  ✓ Outputs: {[out.name for out in sess_cached.get_outputs()]}")

	# Set avg_time_cached to None to indicate not measured
	avg_time_cached = None
	std_time_cached = None

	# Calculate speedup (N/A for now)
	speedup = None

	print("\n" + "=" * 80)
	print("Results Summary")
	print("=" * 80)
	print(f"  MCTS use case: {n_eval_sequences} evaluated sequences with shared prefix")
	print(f"  Prefix length: {prefix_len}, Evaluated length: {eval_len}")
	print(f"  No cache:   {avg_time_no_cache:.2f} ± {std_time_no_cache:.2f} ms")
	print(f"  With cache: N/A (requires architecture changes)")
	print(f"  Speedup:    N/A")
	print()
	print("  Note: Cache export successful, but current implementation doesn't support")
	print("  the MCTS prefix-reuse pattern. Cache accumulates for autoregressive generation,")
	print("  but MCTS needs fixed prefix with multiple independent evaluations.")
	print()
	print("  Recommendation: Implement in C++ with proper cache lifecycle management")
	print("=" * 80)

	# Cleanup
	import shutil
	shutil.rmtree(output_dir_no_cache, ignore_errors=True)
	shutil.rmtree(output_dir_with_cache, ignore_errors=True)

	return {
		'avg_time_no_cache': avg_time_no_cache,
		'std_time_no_cache': std_time_no_cache,
		'avg_time_cached': avg_time_cached,
		'std_time_cached': std_time_cached,
		'speedup': speedup,
		'n_eval_sequences': n_eval_sequences,
		'prefix_len': prefix_len,
		'eval_len': eval_len,
	}


def main():
	parser = argparse.ArgumentParser(description='Benchmark KV cache performance')
	parser.add_argument(
		'training_dir',
		type=str,
		help='Path to training output directory'
	)
	parser.add_argument(
		'--iterations',
		type=int,
		default=20,
		help='Number of benchmark iterations (default: 20)'
	)

	args = parser.parse_args()

	try:
		results = benchmark_kvcache(args.training_dir, args.iterations)
		return 0
	except Exception as e:
		print(f"\nBenchmark failed: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == '__main__':
	sys.exit(main())
