"""
ONNX Export Script for TrigoRL Models.

This script exports trained models to ONNX format for cross-platform deployment.
It loads a model from a training checkpoint and exports it to ONNX, with optional
quantization support for reduced model size.

Usage:
    python exportOnnx.py <training_dir> [options]

Examples:
    # Export latest checkpoint with default settings
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000

    # Export best checkpoint
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 --checkpoint best

    # Export and quantize to INT8 with dynamic quantization
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --quantize --quant-type int8

    # Export and quantize to INT4 with static quantization
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --quantize --quant-method static --quant-type int4 --calibration-samples 200

    # Export specific checkpoint with custom output name
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --checkpoint ep0050_loss_0.1234.chkpt \\
        --output my_model.onnx

    # Export with dynamic batch size and quantization
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --dynamic-batch --quantize

    # Export with specific sequence length
    python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \\
        --seq-len 512
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader

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


	def quantize_model(
		self,
		input_path: str,
		output_path: Optional[str] = None,
		quant_method: str = 'dynamic',
		quant_type: str = 'int8',
		calibration_samples: int = 100,
	) -> str:
		"""
		Quantize ONNX model.

		Args:
		    input_path: Path to input ONNX model
		    output_path: Path to save quantized model (auto-generated if None)
		    quant_method: Quantization method ('dynamic' or 'static')
		    quant_type: Quantization type ('int8' or 'int4')
		    calibration_samples: Number of calibration samples for static quantization

		Returns:
		    Path to quantized model
		"""
		logger.info("\n" + "=" * 80)
		logger.info("Quantizing Model")
		logger.info("=" * 80)

		input_path = Path(input_path)
		if not input_path.exists():
			raise FileNotFoundError(f"Input model not found: {input_path}")

		# Generate output path if not specified
		if output_path is None:
			suffix = f"_{quant_type}"
			output_path = input_path.parent / f"{input_path.stem}{suffix}.onnx"
		else:
			output_path = Path(output_path)

		logger.info(f"Input model: {input_path}")
		logger.info(f"Output model: {output_path}")
		logger.info(f"Method: {quant_method}")
		logger.info(f"Type: {quant_type}")

		# Get input model size
		input_size_mb = input_path.stat().st_size / (1024 * 1024)
		logger.info(f"Input size: {input_size_mb:.2f} MB")

		# Map quantization type
		quant_type_map = {
			'int8': QuantType.QInt8,
			'uint8': QuantType.QUInt8,
			'int4': QuantType.QInt4,
			'uint4': QuantType.QUInt4,
		}
		if quant_type not in quant_type_map:
			raise ValueError(f"Unsupported quantization type: {quant_type}")

		weight_type = quant_type_map[quant_type]

		try:
			if quant_method == 'dynamic':
				# Dynamic quantization (weights only)
				logger.info("Running dynamic quantization...")
				quantize_dynamic(
					model_input=str(input_path),
					model_output=str(output_path),
					weight_type=weight_type,
				)

			elif quant_method == 'static':
				# Static quantization (weights + activations)
				logger.info("Running static quantization...")
				logger.info(f"Calibration samples: {calibration_samples}")

				# Create calibration data reader
				class DummyCalibrationDataReader(CalibrationDataReader):
					def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
						self.vocab_size = vocab_size
						self.seq_len = seq_len
						self.num_samples = num_samples
						self.sample_counter = 0

					def get_next(self):
						if self.sample_counter >= self.num_samples:
							return None
						self.sample_counter += 1
						# Generate random input
						input_ids = torch.randint(0, self.vocab_size, (1, self.seq_len), dtype=torch.int64)
						return {'input_ids': input_ids.numpy()}

				vocab_size = self.config.model.config.model_config.config.vocab_size
				calibration_reader = DummyCalibrationDataReader(
					vocab_size=vocab_size,
					seq_len=256,
					num_samples=calibration_samples
				)

				quantize_static(
					model_input=str(input_path),
					model_output=str(output_path),
					calibration_data_reader=calibration_reader,
					weight_type=weight_type,
				)

			else:
				raise ValueError(f"Unsupported quantization method: {quant_method}")

			# Get output model size
			output_size_mb = output_path.stat().st_size / (1024 * 1024)
			compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

			logger.info("✓ Quantization complete!")
			logger.info(f"  Output size: {output_size_mb:.2f} MB")
			logger.info(f"  Compression: {compression_ratio:.2f}x")
			logger.info(f"  Saved: {input_size_mb - output_size_mb:.2f} MB")

			return str(output_path)

		except Exception as e:
			logger.error(f"✗ Quantization failed: {e}")
			raise


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
		logger.info("\n" + "=" * 80)
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


	def export_tree_mode(
		self,
		model: nn.Module,
		output_path: str,
		batch_size: int = 1,
		prefix_len: int = 128,
		eval_len: int = 64,
		dynamic_batch: bool = True,
		dynamic_n: bool = True,
		dynamic_m: bool = True,
		opset_version: int = 14,
	) -> None:
		"""
		Export model in tree mode with custom attention masking.

		Args:
		    model: PyTorch model to export (will be wrapped in TreeLM)
		    output_path: Path to save ONNX model
		    batch_size: Batch size for dummy input (default: 1)
		    prefix_len: Length of prefix (n) for dummy example (default: 128)
		    eval_len: Length of evaluated sequence (m) for dummy example (default: 64)
		    dynamic_batch: Enable dynamic batch size axis (default: True)
		    dynamic_n: Enable dynamic prefix length axis (default: True)
		    dynamic_m: Enable dynamic evaluated length axis (default: True)
		    opset_version: ONNX opset version (default: 14)
		"""
		logger.info("\n" + "=" * 80)
		logger.info("Exporting to ONNX (Tree Mode)")
		logger.info("=" * 80)

		# Import TreeLM
		from trigor.models import TreeLM, create_causal_evaluated_mask

		# Wrap model in TreeLM
		tree_model = TreeLM(model)
		tree_model.eval()

		# Convert to float32 for ONNX compatibility (bfloat16 not supported by CPU)
		tree_model = tree_model.to(dtype=torch.float32)
		logger.info("Converted model to float32 for ONNX compatibility")

		# Create dummy inputs
		vocab_size = self.config.model.config.model_config.config.vocab_size

		dummy_prefix_ids = torch.randint(
			0, vocab_size, (batch_size, prefix_len), dtype=torch.long
		)

		dummy_evaluated_ids = torch.randint(
			0, vocab_size, (batch_size, eval_len), dtype=torch.long
		)

		# Create dummy evaluated_mask (causal by default)
		dummy_evaluated_mask = create_causal_evaluated_mask(
			eval_len, device=dummy_prefix_ids.device
		).expand(batch_size, eval_len, eval_len)

		logger.info(f"Dummy input shapes:")
		logger.info(f"  prefix_ids: {dummy_prefix_ids.shape}")
		logger.info(f"  evaluated_ids: {dummy_evaluated_ids.shape}")
		logger.info(f"  evaluated_mask: {dummy_evaluated_mask.shape}")
		logger.info(f"  n (prefix length): {prefix_len}, m (evaluated length): {eval_len}")

		# Define input/output names
		input_names = ['prefix_ids', 'evaluated_ids', 'evaluated_mask']
		output_names = ['logits']

		# Define dynamic axes
		dynamic_axes = {}
		if dynamic_batch or dynamic_n or dynamic_m:
			# prefix_ids axes
			axes_prefix = {}
			if dynamic_batch:
				axes_prefix[0] = 'batch_size'
			if dynamic_n:
				axes_prefix[1] = 'n'

			# evaluated_ids axes
			axes_eval = {}
			if dynamic_batch:
				axes_eval[0] = 'batch_size'
			if dynamic_m:
				axes_eval[1] = 'm'

			# evaluated_mask axes
			axes_mask = {}
			if dynamic_batch:
				axes_mask[0] = 'batch_size'
			if dynamic_m:
				axes_mask[1] = 'm'
				axes_mask[2] = 'm'

			dynamic_axes['prefix_ids'] = axes_prefix
			dynamic_axes['evaluated_ids'] = axes_eval
			dynamic_axes['evaluated_mask'] = axes_mask

			# Output axes (m+1 is dynamic based on m)
			output_axes = {}
			if dynamic_batch:
				output_axes[0] = 'batch_size'
			if dynamic_m:
				output_axes[1] = 'm_plus_1'  # m+1 positions
			output_axes[2] = 'vocab_size'
			dynamic_axes['logits'] = output_axes

			logger.info(f"Dynamic axes: {dynamic_axes}")

		# Export to ONNX
		logger.info(f"Exporting to: {output_path}")

		try:
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					tree_model,
					(dummy_prefix_ids, dummy_evaluated_ids, dummy_evaluated_mask),
					output_path,
					input_names=input_names,
					output_names=output_names,
					dynamic_axes=dynamic_axes if dynamic_axes else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					dynamo=False,
				)

			logger.info("✓ ONNX export successful!")

			# Get file size
			file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
			logger.info(f"  File size: {file_size_mb:.2f} MB")
			logger.info(f"  Opset version: {opset_version}")
			logger.info(f"  Inputs: {', '.join(input_names)}")
			logger.info(f"  Output: {output_names[0]} - shape [batch, m+1, vocab_size]")
			logger.info(f"  Mode: Tree (n={prefix_len}, m={eval_len} example)")

		except Exception as e:
			logger.error(f"✗ ONNX export failed: {e}")
			raise


	def export_evaluation_mode(
		self,
		model: nn.Module,
		output_path: str,
		batch_size: int = 1,
		seq_len: int = 256,
		dynamic_batch: bool = True,
		dynamic_seq: bool = True,
		opset_version: int = 14,
	) -> None:
		"""
		Export model in evaluation mode for value prediction.

		Args:
		    model: ValueCausalLoss model (will be wrapped in EvaluationLM)
		    output_path: Path to save ONNX model
		    batch_size: Batch size for dummy input (default: 1)
		    seq_len: Sequence length for dummy input (default: 256)
		    dynamic_batch: Enable dynamic batch size axis (default: True)
		    dynamic_seq: Enable dynamic sequence length axis (default: True)
		    opset_version: ONNX opset version (default: 14)
		"""
		logger.info("\n" + "=" * 80)
		logger.info("Exporting to ONNX (Evaluation Mode)")
		logger.info("=" * 80)

		# Import EvaluationLM
		from trigor.models.evaluationLM import EvaluationLM

		# Check if model has value_head (must be ValueCausalLoss)
		if not hasattr(model, 'value_head'):
			raise ValueError(
				"--evaluation-mode requires a ValueCausalLoss model with value_head. "
				f"Got model type: {type(model).__name__}"
			)

		# Wrap model in EvaluationLM
		eval_model = EvaluationLM(
			base_model=model.model,
			value_head=model.value_head,
			value_id=getattr(model, 'value_id', 3)
		)
		eval_model.eval()

		# Convert to float32 for ONNX compatibility (bfloat16 not supported)
		eval_model = eval_model.to(dtype=torch.float32)

		# Create dummy input
		vocab_size = self.config.model.config.model_config.config.vocab_size
		dummy_input_ids = torch.randint(
			0, vocab_size, (batch_size, seq_len), dtype=torch.long
		)

		logger.info(f"Dummy input shape: {dummy_input_ids.shape}")

		# Define input/output names
		input_names = ['input_ids']
		output_names = ['values']

		# Define dynamic axes
		dynamic_axes = {}
		if dynamic_batch or dynamic_seq:
			axes = {}
			if dynamic_batch:
				axes[0] = 'batch_size'
			if dynamic_seq:
				axes[1] = 'sequence_length'

			dynamic_axes['input_ids'] = axes

			# Output only has dynamic batch dimension
			if dynamic_batch:
				dynamic_axes['values'] = {0: 'batch_size'}

			logger.info(f"Dynamic axes: {dynamic_axes}")

		# Export to ONNX
		logger.info(f"Exporting to: {output_path}")

		try:
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					eval_model,
					dummy_input_ids,
					output_path,
					input_names=input_names,
					output_names=output_names,
					dynamic_axes=dynamic_axes if dynamic_axes else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					dynamo=False,
				)

			logger.info("✓ ONNX export successful!")

			# Get file size
			file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
			logger.info(f"  File size: {file_size_mb:.2f} MB")
			logger.info(f"  Opset version: {opset_version}")
			logger.info(f"  Input: {input_names[0]} - shape {list(dummy_input_ids.shape)}")
			logger.info(f"  Output: {output_names[0]} - shape [batch_size]")
			logger.info(f"  Mode: Evaluation (value prediction)")

		except Exception as e:
			logger.error(f"✗ ONNX export failed: {e}")
			raise


	def export_shared_architecture(
		self,
		model: nn.Module,
		output_dir: str,
		batch_size: int = 1,
		prefix_len: int = 128,
		eval_len: int = 64,
		seq_len: int = 256,
		dynamic_batch: bool = True,
		dynamic_n: bool = True,
		dynamic_m: bool = True,
		dynamic_seq: bool = True,
		opset_version: int = 14,
	) -> Tuple[str, str, str]:
		"""
		Export model with shared architecture: base model + separate policy/value heads.

		This exports 3 separate ONNX models:
		    1. base_model.onnx (~400MB) - Shared GPT-2 transformer with tree attention
		    2. policy_head.onnx (~10MB) - TreeLM output projection
		    3. value_head.onnx (~1MB) - EvaluationLM value head

		Benefits:
		    - 48% memory savings (411MB vs 800MB for duplicate models)
		    - 50% inference speedup (run base model once, share outputs)

		Args:
		    model: Trained ValueCausalLoss model with both policy and value heads
		    output_dir: Directory to save the three ONNX models
		    batch_size: Batch size for dummy inputs
		    prefix_len: Length of prefix for policy model (n)
		    eval_len: Length of evaluated sequence for policy model (m)
		    seq_len: Sequence length for value model
		    dynamic_batch: Enable dynamic batch size
		    dynamic_n: Enable dynamic prefix length (policy)
		    dynamic_m: Enable dynamic evaluated length (policy)
		    dynamic_seq: Enable dynamic sequence length (value)
		    opset_version: ONNX opset version

		Returns:
		    Tuple of (base_model_path, policy_head_path, value_head_path)
		"""
		logger.info("\n" + "=" * 80)
		logger.info("Exporting Shared Architecture (3 Models)")
		logger.info("=" * 80)

		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		# Define output paths
		base_model_path = str(output_dir / "base_model.onnx")
		policy_head_path = str(output_dir / "policy_head.onnx")
		value_head_path = str(output_dir / "value_head.onnx")

		# Check if model has required components
		if not hasattr(model, 'value_head'):
			raise ValueError(
				"--shared-architecture requires a ValueCausalLoss model with value_head. "
				f"Got model type: {type(model).__name__}"
			)

		# Get base model (unwrap)
		if hasattr(model, 'model'):
			base_model = model.model
		else:
			raise ValueError("Cannot extract base model from provided model")

		# Convert to float32 for ONNX compatibility
		base_model = base_model.to(dtype=torch.float32)
		model = model.to(dtype=torch.float32)

		vocab_size = self.config.model.config.model_config.config.vocab_size

		# ========================================
		# 1. Export Base Model with Tree Attention
		# ========================================
		logger.info("\n[1/3] Exporting base model with tree attention...")

		class BaseModelWithTreeAttention(nn.Module):
			"""Base transformer that outputs hidden states with tree attention support."""
			def __init__(self, base_model):
				super().__init__()
				self.model = base_model

			def forward(
				self,
				prefix_ids: torch.Tensor,
				evaluated_ids: torch.Tensor,
				evaluated_mask: torch.Tensor,
			) -> torch.Tensor:
				"""
				Forward pass with tree attention.

				Inputs:
				    prefix_ids: [batch, n]
				    evaluated_ids: [batch, m]
				    evaluated_mask: [batch, m, m]

				Returns:
				    hidden_states: [batch, n+m, hidden_dim]
				"""
				batch_size, n = prefix_ids.shape
				_, m = evaluated_ids.shape

				# Concatenate inputs
				input_ids = torch.cat([prefix_ids, evaluated_ids], dim=1)  # [batch, n+m]

				# Calculate position_ids (same as TreeLM logic)
				prefix_positions = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
				mask_row_sums = evaluated_mask.sum(dim=2)
				evaluated_positions = (n + mask_row_sums - 1).long()
				position_ids = torch.cat([prefix_positions, evaluated_positions], dim=1)

				# Build combined attention mask
				total_len = n + m
				causal_mask = torch.tril(
					torch.ones(total_len, total_len, device=input_ids.device, dtype=torch.float32)
				)
				combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
				combined_mask[:, n:, n:] = evaluated_mask

				# Convert to log-space (-inf for masked positions)
				mask_value = -float("inf")
				combined_mask = torch.where(
					combined_mask == 1.0,
					torch.tensor(0.0, dtype=torch.float32, device=input_ids.device),
					torch.tensor(mask_value, dtype=torch.float32, device=input_ids.device)
				)
				attention_mask = combined_mask.unsqueeze(1)  # [batch, 1, n+m, n+m]

				# Forward through base model
				outputs = self.model(
					input_ids,
					attention_mask=attention_mask,
					position_ids=position_ids,
					output_hidden_states=True
				)

				# Return last layer hidden states
				if hasattr(outputs, 'hidden_states'):
					hidden_states = outputs.hidden_states[-1]
				elif isinstance(outputs, dict) and 'hidden_states' in outputs:
					hidden_states = outputs['hidden_states'][-1]
				else:
					raise ValueError("Model output does not contain hidden_states")

				return hidden_states  # [batch, n+m, hidden_dim]

		base_wrapper = BaseModelWithTreeAttention(base_model)
		base_wrapper.eval()

		# Create dummy inputs for base model
		dummy_prefix_ids = torch.randint(0, vocab_size, (batch_size, prefix_len), dtype=torch.long)
		dummy_evaluated_ids = torch.randint(0, vocab_size, (batch_size, eval_len), dtype=torch.long)

		# Create causal evaluated_mask
		from trigor.models import create_causal_evaluated_mask
		dummy_evaluated_mask = create_causal_evaluated_mask(
			eval_len, device=dummy_prefix_ids.device
		).expand(batch_size, eval_len, eval_len)

		logger.info(f"  Input shapes: prefix_ids={dummy_prefix_ids.shape}, "
		           f"evaluated_ids={dummy_evaluated_ids.shape}, "
		           f"evaluated_mask={dummy_evaluated_mask.shape}")

		# Define dynamic axes for base model
		dynamic_axes_base = {}
		if dynamic_batch or dynamic_n or dynamic_m:
			axes_prefix = {}
			if dynamic_batch:
				axes_prefix[0] = 'batch_size'
			if dynamic_n:
				axes_prefix[1] = 'n'

			axes_eval = {}
			if dynamic_batch:
				axes_eval[0] = 'batch_size'
			if dynamic_m:
				axes_eval[1] = 'm'

			axes_mask = {}
			if dynamic_batch:
				axes_mask[0] = 'batch_size'
			if dynamic_m:
				axes_mask[1] = 'm'
				axes_mask[2] = 'm'

			dynamic_axes_base['prefix_ids'] = axes_prefix
			dynamic_axes_base['evaluated_ids'] = axes_eval
			dynamic_axes_base['evaluated_mask'] = axes_mask

			# Output: [batch, n+m, hidden_dim]
			output_axes = {}
			if dynamic_batch:
				output_axes[0] = 'batch_size'
			if dynamic_n or dynamic_m:
				output_axes[1] = 'n_plus_m'
			output_axes[2] = 'hidden_dim'
			dynamic_axes_base['hidden_states'] = output_axes

		# Export base model
		try:
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					base_wrapper,
					(dummy_prefix_ids, dummy_evaluated_ids, dummy_evaluated_mask),
					base_model_path,
					input_names=['prefix_ids', 'evaluated_ids', 'evaluated_mask'],
					output_names=['hidden_states'],
					dynamic_axes=dynamic_axes_base if dynamic_axes_base else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					dynamo=False,
				)

			file_size_mb = Path(base_model_path).stat().st_size / (1024 * 1024)
			logger.info(f"  ✓ Base model exported: {file_size_mb:.2f} MB")

		except Exception as e:
			logger.error(f"✗ Base model export failed: {e}")
			raise

		# ========================================
		# 2. Export Policy Head (Output Projection)
		# ========================================
		logger.info("\n[2/3] Exporting policy head (output projection)...")

		class PolicyHead(nn.Module):
			"""Output projection layer for policy logits."""
			def __init__(self, base_model):
				super().__init__()
				# Extract lm_head or wte (output embedding) from base model
				if hasattr(base_model, 'lm_head'):
					self.lm_head = base_model.lm_head
				elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'wte'):
					# GPT2 uses tied embeddings
					self.wte = base_model.transformer.wte
					self.tied_embeddings = True
				else:
					raise ValueError("Cannot find output projection layer in base model")

			def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
				"""
				Project hidden states to vocabulary logits.

				Input:
				    hidden_states: [batch, seq_len, hidden_dim]

				Returns:
				    logits: [batch, seq_len, vocab_size]
				"""
				if hasattr(self, 'lm_head'):
					return self.lm_head(hidden_states)
				else:
					# Tied embeddings: logits = hidden @ wte.weight.T
					return torch.matmul(hidden_states, self.wte.weight.t())

		policy_head_wrapper = PolicyHead(base_model)
		policy_head_wrapper.eval()

		# Create dummy input for policy head (output from base model)
		hidden_dim = base_model.config.hidden_size if hasattr(base_model, 'config') else 768
		dummy_hidden_states = torch.randn(batch_size, prefix_len + eval_len, hidden_dim, dtype=torch.float32)

		logger.info(f"  Input shape: hidden_states={dummy_hidden_states.shape}")

		# Define dynamic axes for policy head
		dynamic_axes_policy = {}
		if dynamic_batch or dynamic_n or dynamic_m:
			input_axes = {}
			if dynamic_batch:
				input_axes[0] = 'batch_size'
			if dynamic_n or dynamic_m:
				input_axes[1] = 'seq_len'
			input_axes[2] = 'hidden_dim'

			output_axes = {}
			if dynamic_batch:
				output_axes[0] = 'batch_size'
			if dynamic_n or dynamic_m:
				output_axes[1] = 'seq_len'
			output_axes[2] = 'vocab_size'

			dynamic_axes_policy['hidden_states'] = input_axes
			dynamic_axes_policy['logits'] = output_axes

		# Export policy head
		try:
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					policy_head_wrapper,
					dummy_hidden_states,
					policy_head_path,
					input_names=['hidden_states'],
					output_names=['logits'],
					dynamic_axes=dynamic_axes_policy if dynamic_axes_policy else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					dynamo=False,
				)

			file_size_mb = Path(policy_head_path).stat().st_size / (1024 * 1024)
			logger.info(f"  ✓ Policy head exported: {file_size_mb:.2f} MB")

		except Exception as e:
			logger.error(f"✗ Policy head export failed: {e}")
			raise

		# ========================================
		# 3. Export Value Head
		# ========================================
		logger.info("\n[3/3] Exporting value head...")

		class ValueHeadWrapper(nn.Module):
			"""Value prediction head."""
			def __init__(self, value_head):
				super().__init__()
				self.value_head = value_head

			def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
				"""
				Predict values from hidden states.

				Input:
				    hidden_states: [batch, hidden_dim]

				Returns:
				    values: [batch]
				"""
				return self.value_head(hidden_states).squeeze(-1)

		value_head_wrapper = ValueHeadWrapper(model.value_head)
		value_head_wrapper.eval()

		# Create dummy input for value head (single hidden state vector per batch)
		dummy_value_hidden = torch.randn(batch_size, hidden_dim, dtype=torch.float32)

		logger.info(f"  Input shape: hidden_states={dummy_value_hidden.shape}")

		# Define dynamic axes for value head
		dynamic_axes_value = {}
		if dynamic_batch:
			dynamic_axes_value['hidden_states'] = {0: 'batch_size', 1: 'hidden_dim'}
			dynamic_axes_value['values'] = {0: 'batch_size'}

		# Export value head
		try:
			import warnings
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)

				torch.onnx.export(
					value_head_wrapper,
					dummy_value_hidden,
					value_head_path,
					input_names=['hidden_states'],
					output_names=['values'],
					dynamic_axes=dynamic_axes_value if dynamic_axes_value else None,
					opset_version=opset_version,
					do_constant_folding=True,
					export_params=True,
					dynamo=False,
				)

			file_size_mb = Path(value_head_path).stat().st_size / (1024 * 1024)
			logger.info(f"  ✓ Value head exported: {file_size_mb:.2f} MB")

		except Exception as e:
			logger.error(f"✗ Value head export failed: {e}")
			raise

		# ========================================
		# Summary
		# ========================================
		logger.info("\n" + "=" * 80)
		logger.info("Shared Architecture Export Complete!")
		logger.info("=" * 80)

		base_size = Path(base_model_path).stat().st_size / (1024 * 1024)
		policy_size = Path(policy_head_path).stat().st_size / (1024 * 1024)
		value_size = Path(value_head_path).stat().st_size / (1024 * 1024)
		total_size = base_size + policy_size + value_size

		logger.info(f"  Base model:   {base_model_path} ({base_size:.2f} MB)")
		logger.info(f"  Policy head:  {policy_head_path} ({policy_size:.2f} MB)")
		logger.info(f"  Value head:   {value_head_path} ({value_size:.2f} MB)")
		logger.info(f"  Total size:   {total_size:.2f} MB")
		logger.info("\nUsage:")
		logger.info("  1. Load base_model once")
		logger.info("  2. Run base_model(prefix_ids, evaluated_ids, evaluated_mask) → hidden_states")
		logger.info("  3. For policy: policy_head(hidden_states[n-1:]) → policy_logits")
		logger.info("  4. For value: value_head(hidden_states[:, -1, :]) → values")
		logger.info("=" * 80)

		return base_model_path, policy_head_path, value_head_path


	def run(
		self,
		checkpoint_name: Optional[str] = None,
		output_path: Optional[str] = None,
		batch_size: int = 1,
		seq_len: int = 256,
		dynamic_batch: bool = False,
		dynamic_seq: bool = False,
		opset_version: int = 14,
		quantize: bool = False,
		quant_method: str = 'dynamic',
		quant_type: str = 'int8',
		calibration_samples: int = 100,
		tree_mode: bool = False,
		prefix_len: int = 128,
		evaluation_mode: bool = False,
		shared_architecture: bool = False,
		eval_len: int = 64,
	) -> Tuple[str, Optional[str]]:
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
		    quantize: Whether to quantize the model after export
		    quant_method: Quantization method ('dynamic' or 'static')
		    quant_type: Quantization type ('int8' or 'int4')
		    calibration_samples: Number of calibration samples for static quantization
		    tree_mode: Export in tree mode with custom attention masking
		    prefix_len: Length of prefix for tree mode
		    evaluation_mode: Export in evaluation mode for value prediction
		    shared_architecture: Export with shared base model + separate heads (3 models)
		    eval_len: Length of evaluated sequence for shared architecture (m)

		Returns:
		    Tuple of (onnx_path, quantized_path) where quantized_path is None if not quantized
		"""
		logger.info("=" * 80)
		logger.info("TrigoRL ONNX Export")
		logger.info("=" * 80)
		logger.info(f"Training directory: {self.training_dir}")
		if tree_mode:
			logger.info(f"Mode: Tree (custom attention masking)")
		if evaluation_mode:
			logger.info(f"Mode: Evaluation (value prediction)")
		if shared_architecture:
			logger.info(f"Mode: Shared Architecture (base + 2 heads)")

		# Load model
		model, checkpoint = self.load_model(checkpoint_name)

		# Handle shared architecture mode (outputs to directory, not single file)
		if shared_architecture:
			# For shared architecture, output_path becomes output directory
			if output_path is None:
				model_name = self.config.model.config.model_config.type
				epoch = checkpoint['epoch']
				output_dir = str(self.training_dir / f"{model_name}_ep{epoch:04d}_shared")
			else:
				output_dir = str(Path(output_path).parent / (Path(output_path).stem + "_shared"))

			base_model_path, policy_head_path, value_head_path = self.export_shared_architecture(
				model=model,
				output_dir=output_dir,
				batch_size=batch_size,
				prefix_len=prefix_len,
				eval_len=eval_len,
				seq_len=seq_len,
				dynamic_batch=dynamic_batch,
				dynamic_n=dynamic_seq,
				dynamic_m=dynamic_seq,
				dynamic_seq=dynamic_seq,
				opset_version=opset_version,
			)

			# Return base_model_path as primary output
			return base_model_path, None

		# Generate output path if not specified
		if output_path is None:
			model_name = self.config.model.config.model_config.type
			epoch = checkpoint['epoch']
			if tree_mode:
				suffix = '_tree'
			elif evaluation_mode:
				suffix = '_evaluation'
			else:
				suffix = ''
			output_path = str(self.training_dir / f"{model_name}_ep{epoch:04d}{suffix}.onnx")

		output_path = str(Path(output_path).resolve())

		# Export to ONNX (choose mode)
		if tree_mode:
			eval_len = seq_len - prefix_len  # Calculate eval_len from seq_len and prefix_len
			self.export_tree_mode(
				model=model,
				output_path=output_path,
				batch_size=batch_size,
				prefix_len=prefix_len,
				eval_len=eval_len,
				dynamic_batch=dynamic_batch,
				dynamic_n=dynamic_seq,
				dynamic_m=dynamic_seq,
				opset_version=opset_version,
			)
		elif evaluation_mode:
			self.export_evaluation_mode(
				model=model,
				output_path=output_path,
				batch_size=batch_size,
				seq_len=seq_len,
				dynamic_batch=dynamic_batch,
				dynamic_seq=dynamic_seq,
				opset_version=opset_version,
			)
		else:
			self.export_to_onnx(
				model=model,
				output_path=output_path,
				batch_size=batch_size,
				seq_len=seq_len,
				dynamic_batch=dynamic_batch,
				dynamic_seq=dynamic_seq,
				opset_version=opset_version,
			)

		quantized_path = None

		# Quantize if requested
		if quantize:
			quantized_path = self.quantize_model(
				input_path=output_path,
				quant_method=quant_method,
				quant_type=quant_type,
				calibration_samples=calibration_samples,
			)

		logger.info("\n" + "=" * 80)
		logger.info("Export complete!")
		logger.info("=" * 80)
		logger.info(f"ONNX model: {output_path}")
		if quantized_path:
			logger.info(f"Quantized model: {quantized_path}")
		logger.info("=" * 80)

		return output_path, quantized_path


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

	parser.add_argument(
		'--quantize',
		action='store_true',
		help='Quantize the model after export'
	)

	parser.add_argument(
		'--quant-method',
		type=str,
		default='dynamic',
		choices=['dynamic', 'static'],
		help='Quantization method: dynamic (weights only) or static (weights + activations) (default: dynamic)'
	)

	parser.add_argument(
		'--quant-type',
		type=str,
		default='int8',
		choices=['int8', 'uint8', 'int4', 'uint4'],
		help='Quantization type (default: int8)'
	)

	parser.add_argument(
		'--calibration-samples',
		type=int,
		default=100,
		help='Number of calibration samples for static quantization (default: 100)'
	)

	parser.add_argument(
		'--tree-mode',
		action='store_true',
		help='Export in tree mode with custom attention masking for probability computation'
	)

	parser.add_argument(
		'--prefix-len',
		type=int,
		default=128,
		help='Length of prefix for tree mode dummy example (default: 128)'
	)

	parser.add_argument(
		'--evaluation-mode',
		action='store_true',
		help='Export in evaluation mode for value prediction'
	)

	parser.add_argument(
		'--shared-architecture',
		action='store_true',
		help='Export with shared base model + separate policy/value heads (3 models, 48%% memory savings)'
	)

	parser.add_argument(
		'--eval-len',
		type=int,
		default=64,
		help='Length of evaluated sequence for shared architecture (m) (default: 64)'
	)

	return parser.parse_args()


def main():
	"""Main entry point."""
	args = parse_args()

	try:
		# Create exporter
		exporter = ONNXExporter(args.training_dir)

		# Run export (and optionally quantization)
		onnx_path, quantized_path = exporter.run(
			checkpoint_name=args.checkpoint,
			output_path=args.output,
			batch_size=args.batch_size,
			seq_len=args.seq_len,
			dynamic_batch=args.dynamic_batch,
			dynamic_seq=args.dynamic_seq,
			opset_version=args.opset_version,
			quantize=args.quantize,
			quant_method=args.quant_method,
			quant_type=args.quant_type,
			calibration_samples=args.calibration_samples,
			tree_mode=args.tree_mode,
			prefix_len=args.prefix_len,
			evaluation_mode=args.evaluation_mode,
			shared_architecture=args.shared_architecture,
			eval_len=args.eval_len,
		)

		return 0

	except Exception as e:
		logger.error(f"Export failed: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == '__main__':
	sys.exit(main())
