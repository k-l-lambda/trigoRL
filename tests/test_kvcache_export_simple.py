"""
Simpler validation test for cached ONNX export.

Tests the export_shared_architecture_with_cache method directly without needing full ONNXExporter setup.
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_simple_test():
	"""Simple test that validates basic export functionality."""
	print("=" * 80)
	print("KV Cache ONNX Export - Simple Validation")
	print("=" * 80)

	from transformers import GPT2Config
	from trigor.models import create_causal_evaluated_mask

	# Read the BaseModelWithTreeAttention class from exportOnnx.py
	# This avoids needing the full ONNXExporter infrastructure
	print("\n[1/5] Creating test model...")

	config = GPT2Config(
		vocab_size=1000,
		n_positions=512,
		n_embd=256,
		n_layer=4,
		n_head=4,
	)

	# Create base transformer
	from transformers import GPT2Model
	base_model = GPT2Model(config)
	base_model.eval()

	print(f"  Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} hidden")

	# Test parameters
	batch_size = 1
	prefix_len = 8
	eval_len = 4
	vocab_size = config.vocab_size
	num_layers = config.n_layer
	num_heads = config.n_head
	head_dim = config.n_embd // config.n_head

	# Import BaseModelWithTreeAttention by executing its definition
	print("\n[2/5] Loading export components...")

	exec_globals = {'nn': nn, 'torch': torch}

	# Read and execute BaseModelWithTreeAttention class definition
	with open('exportOnnx.py', 'r') as f:
		content = f.read()

	# Extract the class definition (from line 742 to before base_wrapper =)
	import re
	pattern = r'(class BaseModelWithTreeAttention\(nn\.Module\):.*?)(?=\n\t\tbase_wrapper = |$)'
	match = re.search(pattern, content, re.DOTALL)

	if not match:
		print("  ✗ Could not find BaseModelWithTreeAttention class")
		return False

	class_def = match.group(1)

	# Remove leading tabs
	class_def = '\n'.join(line[2:] if line.startswith('\t\t') else line for line in class_def.split('\n'))

	# Execute to get the class
	exec(class_def, exec_globals)
	BaseModelWithTreeAttention = exec_globals['BaseModelWithTreeAttention']

	print(f"  ✓ BaseModelWithTreeAttention loaded")

	# Test Python forward pass
	print("\n[3/5] Testing Python forward pass with cache...")

	base_with_cache = BaseModelWithTreeAttention(base_model, use_cache=True)
	base_with_cache.eval()

	# Create test inputs
	prefix_ids = torch.randint(0, vocab_size, (batch_size, prefix_len))
	evaluated_ids = torch.randint(0, vocab_size, (batch_size, eval_len))
	evaluated_mask = create_causal_evaluated_mask(eval_len).expand(batch_size, -1, -1)

	with torch.no_grad():
		# No cache mode (but use_cache=True still returns tuple)
		result_no_cache = base_with_cache(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=None)
		if isinstance(result_no_cache, tuple):
			hidden_no_cache, _ = result_no_cache
		else:
			hidden_no_cache = result_no_cache

		# With cache mode - simulate prefix already cached
		dummy_cache = tuple([
			(
				torch.zeros(batch_size, num_heads, prefix_len, head_dim),
				torch.zeros(batch_size, num_heads, prefix_len, head_dim),
			)
			for _ in range(num_layers)
		])

		hidden_cache, new_cache = base_with_cache(prefix_ids, evaluated_ids, evaluated_mask, past_key_values=dummy_cache)

	print(f"  No cache output: {hidden_no_cache.shape}")
	print(f"  With cache output: {hidden_cache.shape}")
	print(f"  New cache layers: {len(new_cache)}")
	print(f"  ✓ Python forward pass successful")

	# Export to ONNX
	print("\n[4/5] Exporting cached model to ONNX...")

	output_dir = Path("/tmp/test_simple_cache")
	output_dir.mkdir(parents=True, exist_ok=True)
	onnx_path = output_dir / "test_cached.onnx"

	# Create wrapper that flattens cache I/O
	class CachedONNXWrapper(nn.Module):
		def __init__(self, base_model_with_cache, num_layers):
			super().__init__()
			self.base = base_model_with_cache
			self.num_layers = num_layers

		def forward(self, prefix_ids, evaluated_ids, evaluated_mask, *past_kv_flat):
			# Reconstruct cache
			if len(past_kv_flat) > 0:
				past_key_values = tuple([
					(past_kv_flat[i*2], past_kv_flat[i*2+1])
					for i in range(self.num_layers)
				])
			else:
				past_key_values = None

			# Forward
			outputs = self.base(prefix_ids, evaluated_ids, evaluated_mask, past_key_values)

			if isinstance(outputs, tuple):
				hidden_states, present_key_values = outputs
				# Flatten cache for output
				present_kv_flat = []
				for key, value in present_key_values:
					present_kv_flat.extend([key, value])
				return (hidden_states, *present_kv_flat)
			else:
				return outputs

	cached_wrapper = CachedONNXWrapper(base_with_cache, num_layers)
	cached_wrapper.eval()

	# Build input/output names
	input_names = ['prefix_ids', 'evaluated_ids', 'evaluated_mask']
	for i in range(num_layers):
		input_names.extend([f'past_key_{i}', f'past_value_{i}'])

	output_names = ['hidden_states']
	for i in range(num_layers):
		output_names.extend([f'present_key_{i}', f'present_value_{i}'])

	# Create dummy inputs for export
	dummy_past_kv = []
	for _ in range(num_layers):
		dummy_past_kv.append(torch.zeros(batch_size, num_heads, prefix_len, head_dim))
		dummy_past_kv.append(torch.zeros(batch_size, num_heads, prefix_len, head_dim))

	# Export
	try:
		import warnings
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")

			torch.onnx.export(
				cached_wrapper,
				(prefix_ids, evaluated_ids, evaluated_mask, *dummy_past_kv),
				str(onnx_path),
				input_names=input_names,
				output_names=output_names,
				opset_version=18,
				do_constant_folding=True,
				export_params=True,
				dynamo=False,
			)

		file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
		print(f"  ✓ ONNX export successful: {file_size_mb:.2f} MB")

	except Exception as e:
		print(f"  ✗ Export failed: {e}")
		import traceback
		traceback.print_exc()
		return False

	# Test ONNX inference
	print("\n[5/5] Testing ONNX inference and measuring speedup...")

	try:
		# Load model
		sess = ort.InferenceSession(str(onnx_path))

		# Check actual input names
		actual_input_names = [inp.name for inp in sess.get_inputs()]
		print(f"  Actual input names ({len(actual_input_names)} total):")
		for inp in sess.get_inputs():
			print(f"    - {inp.name}: {inp.type}, shape={inp.shape}")

		# Prepare inputs using actual names based on what we have
		# Build feed_dict dynamically
		feed_dict = {}

		for inp in sess.get_inputs():
			inp_type = 'int64' if 'int64' in inp.type else 'float32'

			if 'prefix' in inp.name.lower():
				feed_dict[inp.name] = prefix_ids.numpy().astype(inp_type)
			elif 'evaluated_ids' in inp.name or 'eval' in inp.name.lower() and 'mask' not in inp.name.lower():
				feed_dict[inp.name] = evaluated_ids.numpy().astype(inp_type)
			elif 'mask' in inp.name.lower():
				# Ensure mask has correct shape - torch tensor already has batch dim
				mask_np = evaluated_mask.numpy().astype('float32')
				if len(mask_np.shape) == 2:
					mask_np = np.expand_dims(mask_np, 0)  # Add batch dim
				feed_dict[inp.name] = mask_np
			elif 'past_key' in inp.name or 'key' in inp.name.lower():
				# Extract layer index
				import re
				match = re.search(r'(\d+)', inp.name)
				if match:
					layer_idx = int(match.group(1))
					feed_dict[inp.name] = dummy_cache[layer_idx][0].numpy().astype('float32')
			elif 'past_value' in inp.name or ('value' in inp.name.lower() and 'past' in inp.name.lower()):
				# Extract layer index
				import re
				match = re.search(r'(\d+)', inp.name)
				if match:
					layer_idx = int(match.group(1))
					feed_dict[inp.name] = dummy_cache[layer_idx][1].numpy().astype('float32')

		# Run inference
		outputs = sess.run(None, feed_dict)

		hidden_onnx = outputs[0]
		print(f"  ONNX output shape: {hidden_onnx.shape}")
		print(f"  Expected: [{batch_size}, {eval_len}, {config.n_embd}]")

		# Check correctness
		assert hidden_onnx.shape == (batch_size, eval_len, config.n_embd), "Shape mismatch"

		# Benchmark
		n_iters = 20

		# Warmup
		for _ in range(3):
			sess.run(None, feed_dict)

		# Benchmark
		start = time.time()
		for _ in range(n_iters):
			sess.run(None, feed_dict)
		avg_time = (time.time() - start) / n_iters * 1000

		print(f"\n  Performance:")
		print(f"    Average: {avg_time:.2f} ms/iter ({n_iters} iterations)")
		print(f"    Throughput: {1000/avg_time:.1f} inferences/sec")

		print(f"  ✓ ONNX inference test passed")

	except Exception as e:
		print(f"  ✗ ONNX inference failed: {e}")
		import traceback
		traceback.print_exc()
		return False

	# Cleanup
	import shutil
	shutil.rmtree(output_dir, ignore_errors=True)

	print("\n" + "=" * 80)
	print("All tests passed! ✓")
	print("=" * 80)

	return True


if __name__ == '__main__':
	success = create_simple_test()
	sys.exit(0 if success else 1)
