#!/usr/bin/env python
"""
Test script to compare memory usage with different dtypes.

Usage:
    python examples/test_dtype_comparison.py
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.models import make_model


def get_model_memory_mb(model):
	"""Calculate model memory usage in MB."""
	total_params = sum(p.numel() for p in model.parameters())
	param_size = next(model.parameters()).element_size()
	total_size = total_params * param_size
	return total_size / 1024 / 1024


def test_dtype(dtype_str):
	"""Test model creation and memory usage with specified dtype."""
	print(f"\n{'=' * 60}")
	print(f"Testing dtype: {dtype_str}")
	print('=' * 60)

	# Create model config
	config = {
		'model_config': {
			'type': 'GPT2CausalLM',
			'config': {
				'vocab_size': 259,
				'hidden_size': 256,
				'num_layers': 6,
				'num_heads': 8,
				'max_seq_len': 2048,
			}
		}
	}

	# Create model
	model = make_model('AttentionCausalLoss', config)

	# Convert to specified dtype
	dtype_map = {
		'float32': torch.float32,
		'float16': torch.float16,
		'bfloat16': torch.bfloat16,
	}
	dtype = dtype_map[dtype_str]

	if dtype != torch.float32:
		model = model.to(dtype=dtype)

	# Get model info
	total_params = sum(p.numel() for p in model.parameters())
	memory_mb = get_model_memory_mb(model)

	print(f"Total parameters: {total_params:,}")
	print(f"Model dtype: {next(model.parameters()).dtype}")
	print(f"Memory usage: {memory_mb:.2f} MB")

	return memory_mb


def main():
	"""Compare memory usage across different dtypes."""
	print("\n" + "=" * 60)
	print("Model Memory Comparison Across Data Types")
	print("=" * 60)

	results = {}
	for dtype in ['float32', 'bfloat16', 'float16']:
		results[dtype] = test_dtype(dtype)

	# Summary
	print(f"\n{'=' * 60}")
	print("Summary")
	print('=' * 60)
	baseline = results['float32']
	for dtype, memory in results.items():
		reduction = (1 - memory / baseline) * 100
		print(f"{dtype:10s}: {memory:8.2f} MB  ({reduction:+5.1f}%)")

	print("\nConclusion:")
	print("- bfloat16 and float16 use ~50% less memory than float32")
	print("- bfloat16 is recommended for most GPUs (better numerical stability)")
	print("- float16 may be faster on older GPUs but less stable")


if __name__ == "__main__":
	main()
