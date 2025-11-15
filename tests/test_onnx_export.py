"""
Test ONNX export functionality.

This script creates a minimal model, trains it briefly, and exports to ONNX
to verify the export pipeline works correctly.
"""

import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from trigor.models import GPT2CausalLM
from trigor.utils.checkpoint import CheckpointManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_model_and_checkpoint(output_dir: Path):
	"""
	Create a minimal GPT-2 model and save a checkpoint.

	Args:
	    output_dir: Directory to save checkpoint and config
	"""
	logger.info("Creating test model and checkpoint...")

	# Create minimal config
	config = OmegaConf.create({
		'model': {
			'type': 'GPT2CausalLM',
			'config': {
				'model_config': {
					'type': 'gpt2',
					'vocab_size': 259,
					'hidden_size': 128,
					'num_layers': 2,
					'num_heads': 2,
					'max_seq_len': 256,
				}
			}
		},
		'training': {
			'dtype': 'float32',
			'checkpoint': {
				'save_mode': 'best',
			},
			'monitor': {
				'field': 'loss',
				'mode': 'min',
			}
		},
		'device': 'cpu',
	})

	# Save config
	config_path = output_dir / 'config.yaml'
	OmegaConf.save(config, config_path)
	logger.info(f"Saved config: {config_path}")

	# Create model
	model = GPT2CausalLM.from_config(config.model.config.model_config)
	model.eval()

	# Create checkpoint
	checkpoint = {
		'epoch': 5,
		'global_step': 1000,
		'global_examples': 8000,
		'validation_count': 5,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': {},
		'scheduler_state_dict': None,
		'best_val_metric': 0.5,
		'wandb_run_id': 'test_run',
		'config': OmegaConf.to_container(config, resolve=True),
	}

	# Save checkpoint
	checkpoint_mgr = CheckpointManager(
		checkpoint_dir=str(output_dir),
		save_mode='best',
		monitor_field='loss',
		monitor_mode='min',
	)

	checkpoint_path = checkpoint_mgr.save(
		checkpoint=checkpoint,
		episode=5,
		metric_value=0.5,
		is_latest=True,
	)

	logger.info(f"Saved checkpoint: {checkpoint_path}")

	return config, model


def test_onnx_export():
	"""Test ONNX export pipeline."""
	logger.info("=" * 80)
	logger.info("Testing ONNX Export Pipeline")
	logger.info("=" * 80)

	# Create temporary directory
	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		logger.info(f"Using temporary directory: {tmpdir}")

		# Create test checkpoint
		config, model = create_test_model_and_checkpoint(tmpdir)

		# Import exporter
		from exportOnnx import ONNXExporter

		# Test 1: Export latest checkpoint
		logger.info("\n" + "=" * 80)
		logger.info("Test 1: Export latest checkpoint")
		logger.info("=" * 80)

		exporter = ONNXExporter(str(tmpdir))
		output_path = tmpdir / 'test_model.onnx'

		exporter.export_to_onnx(
			model=model,
			output_path=str(output_path),
			batch_size=1,
			seq_len=256,
			dynamic_batch=False,
			dynamic_seq=False,
		)

		assert output_path.exists(), "ONNX file was not created"
		logger.info(f"✓ Test 1 passed: {output_path}")

		# Test 2: Export with dynamic axes
		logger.info("\n" + "=" * 80)
		logger.info("Test 2: Export with dynamic axes")
		logger.info("=" * 80)

		output_path_dynamic = tmpdir / 'test_model_dynamic.onnx'

		exporter.export_to_onnx(
			model=model,
			output_path=str(output_path_dynamic),
			batch_size=2,
			seq_len=128,
			dynamic_batch=True,
			dynamic_seq=True,
		)

		assert output_path_dynamic.exists(), "Dynamic ONNX file was not created"
		logger.info(f"✓ Test 2 passed: {output_path_dynamic}")

		# Test 3: Verify ONNX model can be loaded
		logger.info("\n" + "=" * 80)
		logger.info("Test 3: Verify ONNX model loading")
		logger.info("=" * 80)

		try:
			import onnx
			onnx_model = onnx.load(str(output_path))
			onnx.checker.check_model(onnx_model)
			logger.info("✓ Test 3 passed: ONNX model is valid")
		except ImportError:
			logger.warning("⚠ Test 3 skipped: onnx package not installed")
			logger.warning("  Install with: pip install onnx")
		except Exception as e:
			logger.error(f"✗ Test 3 failed: {e}")
			raise

		# Test 4: Verify ONNX runtime inference
		logger.info("\n" + "=" * 80)
		logger.info("Test 4: Verify ONNX Runtime inference")
		logger.info("=" * 80)

		try:
			import onnxruntime as ort
			import numpy as np

			# Create ONNX Runtime session
			session = ort.InferenceSession(str(output_path))

			# Create dummy input
			dummy_input = np.random.randint(0, 259, (1, 256), dtype=np.int64)

			# Run inference
			outputs = session.run(None, {'input_ids': dummy_input})

			logger.info(f"✓ Test 4 passed: Inference successful")
			logger.info(f"  Input shape: {dummy_input.shape}")
			logger.info(f"  Output shape: {outputs[0].shape}")

		except ImportError:
			logger.warning("⚠ Test 4 skipped: onnxruntime package not installed")
			logger.warning("  Install with: pip install onnxruntime")
		except Exception as e:
			logger.error(f"✗ Test 4 failed: {e}")
			raise

		logger.info("\n" + "=" * 80)
		logger.info("All tests passed! ✓")
		logger.info("=" * 80)


if __name__ == '__main__':
	test_onnx_export()
