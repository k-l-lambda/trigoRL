"""
Verify exported ONNX models with correct data types and vocabulary constraints.
"""

import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_tree_mode(model_path: Path, vocab_size: int = 128):
	"""Verify tree mode ONNX model."""
	logger.info(f"\n{'=' * 80}")
	logger.info(f"Verifying Tree Mode: {model_path.name}")
	logger.info(f"{'=' * 80}")

	if not model_path.exists():
		logger.error(f"✗ Model not found: {model_path}")
		return False

	try:
		session = ort.InferenceSession(str(model_path))

		# Display input/output info
		logger.info("Inputs:")
		for inp in session.get_inputs():
			logger.info(f"  {inp.name}: {inp.type} {inp.shape}")
		logger.info("Outputs:")
		for out in session.get_outputs():
			logger.info(f"  {out.name}: {out.type} {out.shape}")

		# Create test inputs with correct types
		prefix_len = 10
		evaluated_len = 5

		# Token IDs must be in range [0, vocab_size-1]
		prefix_ids = np.random.randint(0, vocab_size, (1, prefix_len), dtype=np.int64)
		evaluated_ids = np.random.randint(0, vocab_size, (1, evaluated_len), dtype=np.int64)

		# CRITICAL FIX: Mask must be 3D attention mask [batch, m, m], not 2D [batch, m]
		# Lower triangular mask for causal attention
		evaluated_mask = np.tril(np.ones((1, evaluated_len, evaluated_len), dtype=np.float32))

		inputs = {
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}

		# Run inference
		outputs = session.run(None, inputs)

		logger.info(f"✓ Inference successful")
		logger.info(f"  prefix_ids shape: {prefix_ids.shape}")
		logger.info(f"  evaluated_ids shape: {evaluated_ids.shape}")
		logger.info(f"  evaluated_mask shape: {evaluated_mask.shape} (dtype: {evaluated_mask.dtype})")
		logger.info(f"  logits shape: {outputs[0].shape}")

		return True

	except Exception as e:
		logger.error(f"✗ Verification failed: {e}")
		return False


def verify_evaluation_mode(model_path: Path, vocab_size: int = 128):
	"""Verify evaluation mode ONNX model."""
	logger.info(f"\n{'=' * 80}")
	logger.info(f"Verifying Evaluation Mode: {model_path.name}")
	logger.info(f"{'=' * 80}")

	if not model_path.exists():
		logger.error(f"✗ Model not found: {model_path}")
		return False

	try:
		session = ort.InferenceSession(str(model_path))

		# Display input/output info
		logger.info("Inputs:")
		for inp in session.get_inputs():
			logger.info(f"  {inp.name}: {inp.type} {inp.shape}")
		logger.info("Outputs:")
		for out in session.get_outputs():
			logger.info(f"  {out.name}: {out.type} {out.shape}")

		# Create test input with correct vocabulary range
		# NOTE: EvaluationLM adds +1 for VALUE token, so final length will be seq_len+1
		# Use a common sequence length that won't cause RoPE issues
		seq_len = 256
		# CRITICAL FIX: Token IDs must be in range [0, vocab_size-1]
		input_ids = np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int64)

		inputs = {'input_ids': input_ids}

		# Run inference
		outputs = session.run(None, inputs)

		logger.info(f"✓ Inference successful")
		logger.info(f"  input_ids shape: {input_ids.shape} (range: [{input_ids.min()}, {input_ids.max()}])")
		logger.info(f"  values shape: {outputs[0].shape}")
		logger.info(f"  value: {outputs[0][0]:.4f}")

		return True

	except Exception as e:
		logger.error(f"✗ Verification failed: {e}")
		return False


def verify_shared_architecture(model_dir: Path, vocab_size: int = 128):
	"""Verify shared architecture ONNX models."""
	logger.info(f"\n{'=' * 80}")
	logger.info(f"Verifying Shared Architecture: {model_dir.name}")
	logger.info(f"{'=' * 80}")

	base_model_path = model_dir / "base_model.onnx"
	policy_head_path = model_dir / "policy_head.onnx"
	value_head_path = model_dir / "value_head.onnx"

	if not all([base_model_path.exists(), policy_head_path.exists(), value_head_path.exists()]):
		logger.error(f"✗ Missing model files in: {model_dir}")
		return False

	try:
		# Load all three models
		base_session = ort.InferenceSession(str(base_model_path))
		policy_session = ort.InferenceSession(str(policy_head_path))
		value_session = ort.InferenceSession(str(value_head_path))

		# Display base model info
		logger.info("\nBase Model:")
		for inp in base_session.get_inputs():
			logger.info(f"  Input: {inp.name}: {inp.type} {inp.shape}")
		for out in base_session.get_outputs():
			logger.info(f"  Output: {out.name}: {out.type} {out.shape}")

		# Create test inputs
		prefix_len = 10
		evaluated_len = 5

		# Token IDs must be in range [0, vocab_size-1]
		prefix_ids = np.random.randint(0, vocab_size, (1, prefix_len), dtype=np.int64)
		evaluated_ids = np.random.randint(0, vocab_size, (1, evaluated_len), dtype=np.int64)

		# CRITICAL FIX: Mask must be 3D attention mask [batch, m, m], not 2D [batch, m]
		# Lower triangular mask for causal attention
		evaluated_mask = np.tril(np.ones((1, evaluated_len, evaluated_len), dtype=np.float32))

		base_inputs = {
			'prefix_ids': prefix_ids,
			'evaluated_ids': evaluated_ids,
			'evaluated_mask': evaluated_mask,
		}

		# Run base model
		base_outputs = base_session.run(None, base_inputs)
		hidden_states = base_outputs[0]

		logger.info(f"\n✓ Base model inference successful")
		logger.info(f"  hidden_states shape: {hidden_states.shape}")

		# Run policy head
		policy_outputs = policy_session.run(None, {'hidden_states': hidden_states})
		logger.info(f"✓ Policy head inference successful")
		logger.info(f"  logits shape: {policy_outputs[0].shape}")

		# Run value head
		# Value head expects 2D input: [batch, hidden_dim]
		# Extract last token's hidden state from 3D output: [batch, seq_len, hidden_dim]
		value_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
		value_outputs = value_session.run(None, {'hidden_states': value_hidden})
		logger.info(f"✓ Value head inference successful")
		logger.info(f"  values shape: {value_outputs[0].shape}")
		logger.info(f"  value: {value_outputs[0][0]:.4f}")

		return True

	except Exception as e:
		logger.error(f"✗ Verification failed: {e}")
		return False


def main():
	"""Main verification script."""
	logger.info("=" * 80)
	logger.info("ONNX Model Verification Script")
	logger.info("=" * 80)

	# Model directory
	model_dir = Path("/home/camus/work/trigoRL/outputs/trigor/20251224-trigo-value-llama-l6-h64-it2_251221-value0.02")

	# Vocabulary size from config
	vocab_size = 128
	logger.info(f"\nModel directory: {model_dir}")
	logger.info(f"Vocabulary size: {vocab_size}")

	# Find exported models
	tree_model = list(model_dir.glob("*_tree.onnx"))
	eval_model = list(model_dir.glob("*_evaluation.onnx"))
	shared_dir = list(model_dir.glob("*_shared"))

	results = []

	# Verify tree mode
	if tree_model:
		success = verify_tree_mode(tree_model[0], vocab_size)
		results.append(("Tree Mode", success))
	else:
		logger.warning("⚠ Tree mode model not found")

	# Verify evaluation mode
	if eval_model:
		success = verify_evaluation_mode(eval_model[0], vocab_size)
		results.append(("Evaluation Mode", success))
	else:
		logger.warning("⚠ Evaluation mode model not found")

	# Verify shared architecture
	if shared_dir:
		success = verify_shared_architecture(shared_dir[0], vocab_size)
		results.append(("Shared Architecture", success))
	else:
		logger.warning("⚠ Shared architecture not found")

	# Summary
	logger.info(f"\n{'=' * 80}")
	logger.info("Verification Summary")
	logger.info(f"{'=' * 80}")

	for name, success in results:
		status = "✓ PASS" if success else "✗ FAIL"
		logger.info(f"{name:25s} {status}")

	all_passed = all(success for _, success in results)
	if all_passed:
		logger.info(f"\n{'=' * 80}")
		logger.info("All verifications passed! ✓")
		logger.info(f"{'=' * 80}")
	else:
		logger.error(f"\n{'=' * 80}")
		logger.error("Some verifications failed! ✗")
		logger.error(f"{'=' * 80}")

	return all_passed


if __name__ == '__main__':
	import sys
	success = main()
	sys.exit(0 if success else 1)
