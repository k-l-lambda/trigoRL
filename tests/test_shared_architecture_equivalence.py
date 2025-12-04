"""
Test to verify equivalence of shared architecture export vs original exports.

This test ensures that:
1. Shared architecture (base + policy_head + value_head) produces same policy logits as TreeLM
2. Shared architecture produces same values as EvaluationLM
3. Manual composition of shared models is equivalent to monolithic models

"""

import sys
import argparse
import tempfile
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort


def export_models(training_dir: str, output_dir: Path):
	"""Export all model variants for testing."""
	print("\n" + "=" * 80)
	print("Exporting Models for Testing")
	print("=" * 80)

	# Import here to avoid module-level import issues
	sys.path.insert(0, str(Path(__file__).parent.parent))
	from exportOnnx import ONNXExporter

	exporter = ONNXExporter(training_dir)

	# Export TreeLM (policy)
	print("\n[1/3] Exporting TreeLM...")
	tree_path = str(output_dir / "tree_model.onnx")
	exporter.run(
		checkpoint_name='best',
		output_path=tree_path,
		tree_mode=True,
		prefix_len=128,
		dynamic_batch=True,
		dynamic_seq=True,
	)
	print(f"✓ TreeLM exported: {tree_path}")

	# Export EvaluationLM (value)
	print("\n[2/3] Exporting EvaluationLM...")
	eval_path = str(output_dir / "eval_model.onnx")
	exporter.run(
		checkpoint_name='best',
		output_path=eval_path,
		evaluation_mode=True,
		dynamic_batch=True,
		dynamic_seq=True,
	)
	print(f"✓ EvaluationLM exported: {eval_path}")

	# Export Shared Architecture
	print("\n[3/3] Exporting Shared Architecture...")
	shared_dir = str(output_dir / "shared")
	exporter.run(
		checkpoint_name='best',
		output_path=shared_dir,
		shared_architecture=True,
		prefix_len=128,
		eval_len=64,
		dynamic_batch=True,
		dynamic_seq=True,
	)
	print(f"✓ Shared architecture exported: {shared_dir}")

	return {
		'tree': tree_path,
		'eval': eval_path,
		'shared_base': str(output_dir / "shared_shared" / "base_model.onnx"),
		'shared_policy': str(output_dir / "shared_shared" / "policy_head.onnx"),
		'shared_value': str(output_dir / "shared_shared" / "value_head.onnx"),
	}


def create_test_inputs(batch_size=2, prefix_len=128, eval_len=64, seq_len=256):
	"""Create test inputs for inference."""
	print("\nCreating test inputs...")

	# TreeLM inputs
	prefix_ids = np.random.randint(0, 128, (batch_size, prefix_len), dtype=np.int64)
	evaluated_ids = np.random.randint(0, 128, (batch_size, eval_len), dtype=np.int64)

	# Create causal evaluated_mask (lower triangular)
	evaluated_mask = np.tril(np.ones((eval_len, eval_len), dtype=np.float32))
	evaluated_mask = np.expand_dims(evaluated_mask, 0).repeat(batch_size, axis=0)

	# EvaluationLM inputs
	input_ids = np.random.randint(0, 128, (batch_size, seq_len), dtype=np.int64)

	print(f"  prefix_ids: {prefix_ids.shape}")
	print(f"  evaluated_ids: {evaluated_ids.shape}")
	print(f"  evaluated_mask: {evaluated_mask.shape}")
	print(f"  input_ids (value): {input_ids.shape}")

	return {
		'prefix_ids': prefix_ids,
		'evaluated_ids': evaluated_ids,
		'evaluated_mask': evaluated_mask,
		'input_ids': input_ids,
	}


def test_policy_equivalence(model_paths: dict, test_inputs: dict, rtol=1e-4, atol=1e-5):
	"""Test that shared architecture policy matches TreeLM."""
	print("\n" + "=" * 80)
	print("Testing Policy Equivalence (TreeLM vs Shared Architecture)")
	print("=" * 80)

	# Load models
	print("\nLoading models...")
	tree_session = ort.InferenceSession(model_paths['tree'])
	base_session = ort.InferenceSession(model_paths['shared_base'])
	policy_session = ort.InferenceSession(model_paths['shared_policy'])

	prefix_ids = test_inputs['prefix_ids']
	evaluated_ids = test_inputs['evaluated_ids']
	evaluated_mask = test_inputs['evaluated_mask']
	batch_size, n = prefix_ids.shape
	_, m = evaluated_ids.shape

	# Run TreeLM (monolithic)
	print("\n[1/2] Running TreeLM (monolithic)...")
	tree_outputs = tree_session.run(
		None,
		{
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}
	)
	tree_logits = tree_outputs[0]  # [batch, m+1, vocab_size]
	print(f"  TreeLM output shape: {tree_logits.shape}")
	print(f"  TreeLM logits range: [{tree_logits.min():.4f}, {tree_logits.max():.4f}]")

	# Run Shared Architecture (manual composition)
	print("\n[2/2] Running Shared Architecture (base + policy_head)...")

	# Step 1: Run base model
	base_outputs = base_session.run(
		None,
		{
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}
	)
	hidden_states = base_outputs[0]  # [batch, n+m, hidden_dim]
	print(f"  Base model output shape: {hidden_states.shape}")
	print(f"  Hidden states range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

	# Step 2: Extract relevant positions (last prefix + all evaluated)
	# TreeLM returns logits for positions [n-1, n, n+1, ..., n+m-1]
	hidden_for_policy = hidden_states[:, n-1:, :]  # [batch, m+1, hidden_dim]
	print(f"  Hidden states for policy: {hidden_for_policy.shape}")

	# Step 3: Run policy head
	policy_outputs = policy_session.run(
		None,
		{'hidden_states': hidden_for_policy}
	)
	shared_logits = policy_outputs[0]  # [batch, m+1, vocab_size]
	print(f"  Shared architecture output shape: {shared_logits.shape}")
	print(f"  Shared logits range: [{shared_logits.min():.4f}, {shared_logits.max():.4f}]")

	# Compare outputs
	print("\n" + "-" * 80)
	print("Comparing outputs...")
	print("-" * 80)

	# Check shapes
	assert tree_logits.shape == shared_logits.shape, \
		f"Shape mismatch: TreeLM {tree_logits.shape} vs Shared {shared_logits.shape}"
	print(f"✓ Shapes match: {tree_logits.shape}")

	# Compute differences
	abs_diff = np.abs(tree_logits - shared_logits)
	rel_diff = abs_diff / (np.abs(tree_logits) + 1e-8)

	max_abs_diff = abs_diff.max()
	max_rel_diff = rel_diff.max()
	mean_abs_diff = abs_diff.mean()
	mean_rel_diff = rel_diff.mean()

	print(f"\nDifference statistics:")
	print(f"  Max absolute difference: {max_abs_diff:.6e}")
	print(f"  Max relative difference: {max_rel_diff:.6e}")
	print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
	print(f"  Mean relative difference: {mean_rel_diff:.6e}")

	# Check equivalence
	if np.allclose(tree_logits, shared_logits, rtol=rtol, atol=atol):
		print(f"\n✅ POLICY EQUIVALENCE PASSED!")
		print(f"   TreeLM and Shared Architecture produce identical policy logits")
		print(f"   (within rtol={rtol}, atol={atol})")
		return True
	else:
		print(f"\n❌ POLICY EQUIVALENCE FAILED!")
		print(f"   Outputs differ beyond tolerance (rtol={rtol}, atol={atol})")
		print(f"   Max absolute difference: {max_abs_diff:.6e}")

		# Show some example differences
		print("\nSample differences (first batch, first 5 positions, first 10 vocab):")
		print("TreeLM logits:")
		print(tree_logits[0, :5, :10])
		print("\nShared logits:")
		print(shared_logits[0, :5, :10])
		print("\nDifference:")
		print(abs_diff[0, :5, :10])

		return False


def test_value_equivalence(model_paths: dict, test_inputs: dict, rtol=1e-4, atol=1e-5):
	"""Test that shared architecture value matches EvaluationLM."""
	print("\n" + "=" * 80)
	print("Testing Value Equivalence (EvaluationLM vs Shared Architecture)")
	print("=" * 80)

	# Load models
	print("\nLoading models...")
	eval_session = ort.InferenceSession(model_paths['eval'])
	base_session = ort.InferenceSession(model_paths['shared_base'])
	value_session = ort.InferenceSession(model_paths['shared_value'])

	input_ids = test_inputs['input_ids']
	batch_size, seq_len = input_ids.shape

	# Run EvaluationLM (monolithic)
	print("\n[1/2] Running EvaluationLM (monolithic)...")
	eval_outputs = eval_session.run(
		None,
		{'input_ids': input_ids}
	)
	eval_values = eval_outputs[0]  # [batch]
	print(f"  EvaluationLM output shape: {eval_values.shape}")
	print(f"  EvaluationLM values: {eval_values}")

	# Run Shared Architecture (manual composition)
	# NOTE: EvaluationLM internally APPENDS a VALUE token (id=3), so we need to do the same
	print("\n[2/2] Running Shared Architecture (base + value_head)...")
	print("  Note: Appending VALUE token (id=3) to match EvaluationLM behavior...")

	# Append VALUE token to input_ids
	value_token_id = 3
	value_token = np.full((batch_size, 1), value_token_id, dtype=np.int64)
	input_ids_with_value = np.concatenate([input_ids, value_token], axis=1)  # [batch, seq_len+1]
	print(f"  Input with VALUE token shape: {input_ids_with_value.shape}")

	# For shared architecture, we need to split into prefix and evaluated
	# Use same split as export (prefix_len=128)
	prefix_len = 128
	eval_len = input_ids_with_value.shape[1] - prefix_len  # seq_len + 1 - 128

	prefix_ids = input_ids_with_value[:, :prefix_len]
	evaluated_ids = input_ids_with_value[:, prefix_len:]

	# Create causal mask for evaluated region
	evaluated_mask = np.tril(np.ones((eval_len, eval_len), dtype=np.float32))
	evaluated_mask = np.expand_dims(evaluated_mask, 0).repeat(batch_size, axis=0)

	# Step 1: Run base model
	base_outputs = base_session.run(
		None,
		{
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}
	)
	hidden_states = base_outputs[0]  # [batch, n+m, hidden_dim]
	print(f"  Base model output shape: {hidden_states.shape}")

	# Step 2: Extract last position hidden state (where VALUE token is)
	value_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
	print(f"  Hidden state for value (at VALUE token position): {value_hidden.shape}")

	# Step 3: Run value head
	value_outputs = value_session.run(
		None,
		{'hidden_states': value_hidden}
	)
	shared_values = value_outputs[0]  # [batch]
	print(f"  Shared architecture output shape: {shared_values.shape}")
	print(f"  Shared values: {shared_values}")

	# Compare outputs
	print("\n" + "-" * 80)
	print("Comparing outputs...")
	print("-" * 80)

	# Check shapes
	assert eval_values.shape == shared_values.shape, \
		f"Shape mismatch: EvaluationLM {eval_values.shape} vs Shared {shared_values.shape}"
	print(f"✓ Shapes match: {eval_values.shape}")

	# Compute differences
	abs_diff = np.abs(eval_values - shared_values)
	rel_diff = abs_diff / (np.abs(eval_values) + 1e-8)

	max_abs_diff = abs_diff.max()
	max_rel_diff = rel_diff.max()
	mean_abs_diff = abs_diff.mean()
	mean_rel_diff = rel_diff.mean()

	print(f"\nDifference statistics:")
	print(f"  Max absolute difference: {max_abs_diff:.6e}")
	print(f"  Max relative difference: {max_rel_diff:.6e}")
	print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
	print(f"  Mean relative difference: {mean_rel_diff:.6e}")

	# Check equivalence
	if np.allclose(eval_values, shared_values, rtol=rtol, atol=atol):
		print(f"\n✅ VALUE EQUIVALENCE PASSED!")
		print(f"   EvaluationLM and Shared Architecture produce identical values")
		print(f"   (within rtol={rtol}, atol={atol})")
		return True
	else:
		print(f"\n❌ VALUE EQUIVALENCE FAILED!")
		print(f"   Outputs differ beyond tolerance (rtol={rtol}, atol={atol})")
		print(f"   Max absolute difference: {max_abs_diff:.6e}")

		# Show differences
		print("\nPer-batch comparison:")
		for i in range(batch_size):
			print(f"  Batch {i}: Eval={eval_values[i]:.6f}, Shared={shared_values[i]:.6f}, "
			      f"Diff={abs_diff[i]:.6e}")

		return False


def main():
	parser = argparse.ArgumentParser(
		description='Test equivalence of shared architecture vs original models'
	)
	parser.add_argument(
		'training_dir',
		type=str,
		help='Path to training directory with checkpoints'
	)
	parser.add_argument(
		'--output-dir',
		type=str,
		default=None,
		help='Directory to save exported models (default: temp directory)'
	)
	parser.add_argument(
		'--rtol',
		type=float,
		default=1e-4,
		help='Relative tolerance for comparison (default: 1e-4)'
	)
	parser.add_argument(
		'--atol',
		type=float,
		default=1e-5,
		help='Absolute tolerance for comparison (default: 1e-5)'
	)
	parser.add_argument(
		'--batch-size',
		type=int,
		default=2,
		help='Batch size for test inputs (default: 2)'
	)

	args = parser.parse_args()

	print("=" * 80)
	print("Shared Architecture Equivalence Test")
	print("=" * 80)
	print(f"Training directory: {args.training_dir}")
	print(f"Batch size: {args.batch_size}")
	print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")

	# Create output directory
	if args.output_dir:
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		temp_cleanup = False
	else:
		output_dir = Path(tempfile.mkdtemp(prefix='trigo_test_'))
		temp_cleanup = True
		print(f"Using temporary directory: {output_dir}")

	try:
		# Step 1: Export all models
		model_paths = export_models(args.training_dir, output_dir)

		# Step 2: Create test inputs
		test_inputs = create_test_inputs(batch_size=args.batch_size)

		# Step 3: Test policy equivalence
		policy_passed = test_policy_equivalence(
			model_paths, test_inputs, rtol=args.rtol, atol=args.atol
		)

		# Step 4: Test value equivalence
		value_passed = test_value_equivalence(
			model_paths, test_inputs, rtol=args.rtol, atol=args.atol
		)

		# Final summary
		print("\n" + "=" * 80)
		print("FINAL RESULTS")
		print("=" * 80)
		print(f"Policy equivalence: {'✅ PASSED' if policy_passed else '❌ FAILED'}")
		print(f"Value equivalence:  {'✅ PASSED' if value_passed else '❌ FAILED'}")

		if policy_passed and value_passed:
			print("\n🎉 ALL TESTS PASSED!")
			print("   Shared architecture is mathematically equivalent to original models.")
			print("   Safe to use for C++ inference!")
			return 0
		else:
			print("\n⚠️  SOME TESTS FAILED")
			print("   Please review the differences above.")
			return 1

	finally:
		# Cleanup temp directory if used
		if temp_cleanup:
			import shutil
			print(f"\nCleaning up temporary directory: {output_dir}")
			shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == '__main__':
	sys.exit(main())
