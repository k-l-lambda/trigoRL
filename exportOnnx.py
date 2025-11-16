"""
ONNX Export Script for TrigoRL Models.

This script exports trained models to ONNX format for cross-platform deployment.
It loads a model from a training checkpoint and exports it to ONNX.

Usage:
    python exportOnnx.py <training_dir> [options]

Examples:
    # Export latest checkpoint with default settings
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000

    # Export best checkpoint
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 --checkpoint best

    # Export specific checkpoint with custom output name
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --checkpoint ep0050_loss_0.1234.chkpt \\
        --output my_model.onnx

    # Export with dynamic batch size
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --dynamic-batch

    # Export with specific sequence length
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --seq-len 512
"""

import argparse
import glob
import re
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from trigor.models import make_model
from trigor.utils.checkpoint import CheckpointManager


logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXExporter:
	"""
	ONNX exporter for TrigoRL models.

	Supports exporting GPT-2, LLAMA, RWKV, and xLSTM models to ONNX format.
	"""

	def __init__(self, training_dir: str):
		"""
		Initialize ONNX exporter.

		Args:
		    training_dir: Path to training output directory containing checkpoints
		"""
		self.training_dir = Path(training_dir)

		if not self.training_dir.exists():
			raise FileNotFoundError(f"Training directory not found: {training_dir}")

		# Load config from training directory
		config_path = self.training_dir / 'config.yaml'
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")

		self.config = OmegaConf.load(config_path)
		logger.info(f"Loaded config from: {config_path}")

		# Initialize checkpoint manager
		self.checkpoint_mgr = CheckpointManager(
			checkpoint_dir=str(self.training_dir / "checkpoints"),
			save_mode=self.config.training.save_mode,
			monitor_field=self.config.training.monitor.field,
			monitor_mode=self.config.training.monitor.mode,
		)



	def load_model(self, checkpoint_name: Optional[str] = None) -> Tuple[nn.Module, Dict]:
		"""
		Load model from checkpoint.

		Args:
		    checkpoint_name: Checkpoint filename ('latest', 'best', or specific filename)
		                     If None, loads latest checkpoint

		Returns:
		    Tuple of (model, checkpoint_dict)
		"""
		# Determine checkpoint path
		if checkpoint_name is None or checkpoint_name == 'latest':
			checkpoint_path = self.checkpoint_mgr.get_latest_checkpoint()
			if checkpoint_path is None:
				raise FileNotFoundError("No latest checkpoint found")
			logger.info("Using latest checkpoint")
		elif checkpoint_name == 'best':
			checkpoint_path = self.checkpoint_mgr.get_best_checkpoint()
			if checkpoint_path is None:
				raise FileNotFoundError("No best checkpoint found")
			logger.info("Using best checkpoint")
		else:
			# Specific checkpoint filename
			checkpoint_path = str(self.training_dir / "checkpoints" / checkpoint_name)
			if not Path(checkpoint_path).exists():
				raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

		logger.info(f"Loading checkpoint: {checkpoint_path}")

		# Load checkpoint
		checkpoint = torch.load(checkpoint_path, map_location='cpu')

		# Create model from config
		model = make_model(self.config.model.type, self.config.model.config)

		# Load model weights
		model.load_state_dict(checkpoint['model_state_dict'])

		# Set to evaluation mode
		model.eval()

		# Get dtype from config or checkpoint config
		dtype_str = self.config.training.get('dtype', 'float32')
		dtype_map = {
			'float32': torch.float32,
			'fp32': torch.float32,
			'float16': torch.float16,
			'fp16': torch.float16,
			'bfloat16': torch.bfloat16,
			'bf16': torch.bfloat16,
		}
		dtype = dtype_map.get(dtype_str.lower(), torch.float32)

		# Convert model dtype
		if dtype != torch.float32:
			logger.info(f"Converting model to dtype: {dtype}")
			model = model.to(dtype=dtype)

		logger.info("Model loaded successfully")
		logger.info(f"  Model type: {self.config.model.type}")
		logger.info(f"  Epoch: {checkpoint['epoch']}")
		logger.info(f"  Global step: {checkpoint['global_step']}")
		logger.info(f"  Dtype: {dtype}")

		return model, checkpoint


	def export_to_onnx(
		self,
		model: nn.Module,
		output_path: str,
		batch_size: int = 1,
		seq_len: int = 256,
		dynamic_batch: bool = False,
		dynamic_seq: bool = False,
		opset_version: int = 14,
	) -> None:
		"""
		Export model to ONNX format.

		Args:
		    model: PyTorch model to export
		    output_path: Path to save ONNX model
		    batch_size: Batch size for dummy input (default: 1)
		    seq_len: Sequence length for dummy input (default: 256)
		    dynamic_batch: Enable dynamic batch size axis
		    dynamic_seq: Enable dynamic sequence length axis
		    opset_version: ONNX opset version (default: 14)
		"""
		logger.info("=" * 80)
		logger.info("Exporting to ONNX")
		logger.info("=" * 80)

		# Wrap model to return only logits (HuggingFace models may return cache)
		class ModelWrapper(nn.Module):
			def __init__(self, model):
				super().__init__()
				self.model = model

			def forward(self, input_ids):
				# Unwrap AttentionCausalLoss to get base model if needed
				if hasattr(self.model, "model"):
					outputs = self.model.model(input_ids)
				else:
					outputs = self.model(input_ids)
				# Handle both direct tensor output and dict/named tuple output
				if isinstance(outputs, torch.Tensor):
					return outputs
				elif hasattr(outputs, 'logits'):
					return outputs.logits
				elif isinstance(outputs, (tuple, list)):
					return outputs[0]
				elif isinstance(outputs, dict):
					return outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
				else:
					raise ValueError(f"Unexpected output type: {type(outputs)}")

		wrapped_model = ModelWrapper(model)
		wrapped_model.eval()

		# Create dummy input
		dummy_input = torch.randint(
			0,
			self.config.model.config.model_config.config.vocab_size,
			(batch_size, seq_len),
			dtype=torch.long
		)

		logger.info(f"Dummy input shape: {dummy_input.shape}")

		# Define input names
		input_names = ['input_ids']
		output_names = ['logits']

		# Define dynamic axes
		dynamic_axes = {}
		if dynamic_batch or dynamic_seq:
			axes = {}
			if dynamic_batch:
				axes[0] = 'batch_size'
			if dynamic_seq:
				axes[1] = 'sequence_length'

			dynamic_axes['input_ids'] = axes
			dynamic_axes['logits'] = axes

			logger.info(f"Dynamic axes: {dynamic_axes}")

		# Export to ONNX
		logger.info(f"Exporting to: {output_path}")

		try:
			# Use export API with explicit settings to avoid torch.export issues
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					wrapped_model,
					dummy_input,
					output_path,
					input_names=input_names,
					output_names=output_names,
					dynamic_axes=dynamic_axes if dynamic_axes else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					# Use JIT trace instead of dynamo (legacy API)
					dynamo=False,
				)

			logger.info("✓ ONNX export successful!")

			# Get file size
			file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
			logger.info(f"  File size: {file_size_mb:.2f} MB")
			logger.info(f"  Opset version: {opset_version}")
			logger.info(f"  Input: {input_names[0]} - shape {list(dummy_input.shape)}")
			logger.info(f"  Output: {output_names[0]}")

		except Exception as e:
			logger.error(f"✗ ONNX export failed: {e}")
			raise


	def run(
		self,
		checkpoint_name: Optional[str] = None,
		output_path: Optional[str] = None,
		batch_size: int = 1,
		seq_len: int = 256,
		dynamic_batch: bool = False,
		dynamic_seq: bool = False,
		opset_version: int = 14,
	) -> str:
		"""
		Run complete export pipeline.

		Args:
		    checkpoint_name: Checkpoint to load ('latest', 'best', or filename)
		    output_path: Path to save ONNX model (auto-generated if None)
		    batch_size: Batch size for dummy input
		    seq_len: Sequence length for dummy input
		    dynamic_batch: Enable dynamic batch size
		    dynamic_seq: Enable dynamic sequence length
		    opset_version: ONNX opset version

		Returns:
		    Path to exported ONNX model
		"""
		logger.info("=" * 80)
		logger.info("TrigoRL ONNX Export")
		logger.info("=" * 80)
		logger.info(f"Training directory: {self.training_dir}")

		# Load model
		model, checkpoint = self.load_model(checkpoint_name)

		# Generate output path if not specified
		if output_path is None:
			model_name = self.config.model.config.model_config.type
			epoch = checkpoint['epoch']
			output_path = str(self.training_dir / f"{model_name}_ep{epoch:04d}.onnx")

		output_path = str(Path(output_path).resolve())

		# Export to ONNX
		self.export_to_onnx(
			model=model,
			output_path=output_path,
			batch_size=batch_size,
			seq_len=seq_len,
			dynamic_batch=dynamic_batch,
			dynamic_seq=dynamic_seq,
			opset_version=opset_version,
		)

		logger.info("=" * 80)
		logger.info("Export complete!")
		logger.info(f"Saved to: {output_path}")
		logger.info("=" * 80)

		return output_path


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Export TrigoRL models to ONNX format',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__
	)

	parser.add_argument(
		'training_dir',
		type=str,
		help='Path to training output directory containing checkpoints'
	)

	parser.add_argument(
		'--checkpoint',
		type=str,
		default='best',
		help='Checkpoint to export: "latest", "best", or specific filename (default: best)'
	)

	parser.add_argument(
		'--output',
		type=str,
		default=None,
		help='Output ONNX file path (default: auto-generated in training_dir)'
	)

	parser.add_argument(
		'--batch-size',
		type=int,
		default=1,
		help='Batch size for dummy input (default: 1)'
	)

	parser.add_argument(
		'--seq-len',
		type=int,
		default=256,
		help='Sequence length for dummy input (default: 256)'
	)

	parser.add_argument(
		'--dynamic-batch',
		action='store_true',
		help='Enable dynamic batch size axis'
	)

	parser.add_argument(
		'--dynamic-seq',
		action='store_true',
		help='Enable dynamic sequence length axis'
	)

	parser.add_argument(
		'--opset-version',
		type=int,
		default=14,
		help='ONNX opset version (default: 14)'
	)

	return parser.parse_args()


def main():
	"""Main entry point."""
	args = parse_args()

	try:
		# Create exporter
		exporter = ONNXExporter(args.training_dir)

		# Run export
		output_path = exporter.run(
			checkpoint_name=args.checkpoint,
			output_path=args.output,
			batch_size=args.batch_size,
			seq_len=args.seq_len,
			dynamic_batch=args.dynamic_batch,
			dynamic_seq=args.dynamic_seq,
			opset_version=args.opset_version,
		)

		return 0

	except Exception as e:
		logger.error(f"Export failed: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == '__main__':
	sys.exit(main())
