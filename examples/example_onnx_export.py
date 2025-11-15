"""
Example usage of the ONNX export script.

This script demonstrates how to export trained models to ONNX format
and use them for inference.
"""

import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model_example(training_dir: str):
	"""
	Example: Export a model from a training directory.

	Args:
	    training_dir: Path to training output directory
	"""
	from exportOnnx import ONNXExporter

	logger.info("=" * 80)
	logger.info("Example: Export Model to ONNX")
	logger.info("=" * 80)

	# Create exporter
	exporter = ONNXExporter(training_dir)

	# Export latest checkpoint
	output_path = exporter.run(
		checkpoint_name='latest',  # or 'best' or specific filename
		output_path=None,  # auto-generate name
		batch_size=1,
		seq_len=256,
		dynamic_batch=True,  # allow variable batch size
		dynamic_seq=True,  # allow variable sequence length
		opset_version=14,
	)

	logger.info(f"Exported to: {output_path}")

	return output_path


def inference_example(onnx_path: str, vocab_size: int = 259):
	"""
	Example: Run inference with exported ONNX model.

	Args:
	    onnx_path: Path to ONNX model
	    vocab_size: Vocabulary size for token generation
	"""
	logger.info("=" * 80)
	logger.info("Example: ONNX Model Inference")
	logger.info("=" * 80)

	# Create ONNX Runtime session
	session = ort.InferenceSession(onnx_path)

	# Print model info
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name

	logger.info(f"Input name: {input_name}")
	logger.info(f"Output name: {output_name}")

	# Create sample input (random token IDs)
	batch_size = 2
	seq_len = 128
	input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int64)

	logger.info(f"Input shape: {input_ids.shape}")

	# Run inference
	outputs = session.run([output_name], {input_name: input_ids})
	logits = outputs[0]

	logger.info(f"Output shape: {logits.shape}")
	logger.info(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

	# Get predicted tokens (greedy decoding)
	predicted_tokens = np.argmax(logits, axis=-1)
	logger.info(f"Predicted tokens shape: {predicted_tokens.shape}")

	logger.info("✓ Inference successful!")

	return logits


def batch_inference_example(onnx_path: str, num_sequences: int = 10, vocab_size: int = 259):
	"""
	Example: Batch inference with different sequence lengths.

	Args:
	    onnx_path: Path to ONNX model
	    num_sequences: Number of sequences to process
	    vocab_size: Vocabulary size
	"""
	logger.info("=" * 80)
	logger.info("Example: Batch Inference with Variable Lengths")
	logger.info("=" * 80)

	session = ort.InferenceSession(onnx_path)
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name

	# Process sequences with different batch sizes
	for batch_size in [1, 2, 4, 8]:
		for seq_len in [64, 128, 256, 512]:
			# Create input
			input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int64)

			# Run inference
			outputs = session.run([output_name], {input_name: input_ids})
			logits = outputs[0]

			logger.info(f"  Batch={batch_size}, SeqLen={seq_len}: Output shape {logits.shape}")

	logger.info("✓ Batch inference successful!")


def main():
	"""Main example execution."""
	import sys

	if len(sys.argv) < 2:
		print("Usage: python examples/example_onnx_export.py <training_dir>")
		print("\nExample:")
		print("  python examples/example_onnx_export.py training_output/trigo-gpt2-20250115_120000")
		sys.exit(1)

	training_dir = sys.argv[1]

	if not Path(training_dir).exists():
		logger.error(f"Training directory not found: {training_dir}")
		sys.exit(1)

	# Export model
	onnx_path = export_model_example(training_dir)

	# Run inference
	inference_example(onnx_path)

	# Batch inference (if model has dynamic axes)
	try:
		batch_inference_example(onnx_path)
	except Exception as e:
		logger.warning(f"Batch inference skipped: {e}")
		logger.warning("Note: Export with --dynamic-batch and --dynamic-seq for variable-size inputs")

	logger.info("=" * 80)
	logger.info("All examples completed! ✓")
	logger.info("=" * 80)


if __name__ == '__main__':
	main()
