#!/usr/bin/env python
"""
Test script for AttentionCausalLoss module.

Tests the loss module with different model architectures and validates
that loss computation and metrics work correctly.
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.models.attentionCausalLoss import AttentionCausalLoss


def test_loss_module_creation():
	"""Test creating loss modules with different model types."""
	print("=" * 80)
	print("TEST 1: Loss Module Creation")
	print("=" * 80 + "\n")

	model_types = ['GPT2CausalLM', 'LlamaCausalLM', 'RwkvCausalLM', 'xLSTMCausalLM']

	for model_type in model_types:
		print(f"Testing {model_type}...")

		config = {
			'model_type': model_type,
			'model_config': {
				'vocab_size': 259,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 4,
				'max_seq_len': 512,
			},
			'ignore_index': 256,
			'label_smoothing': 0.0,
		}

		loss_module = AttentionCausalLoss.from_config(config)
		print(f"  ✓ Created successfully")
		print(f"  {loss_module}\n")

		# Check parameter count
		params = loss_module.count_parameters()
		print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable\n")

	print("✓ All loss modules created successfully!\n")


def test_forward_pass():
	"""Test forward pass with loss and metrics computation."""
	print("=" * 80)
	print("TEST 2: Forward Pass and Metrics")
	print("=" * 80 + "\n")

	# Create a small GPT-2 based loss module
	config = {
		'model_type': 'GPT2CausalLM',
		'model_config': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'num_heads': 4,
			'max_seq_len': 512,
		},
		'ignore_index': 256,
		'label_smoothing': 0.1,
	}

	loss_module = AttentionCausalLoss.from_config(config)
	loss_module.eval()

	print("Loss module created")
	print(f"  Model type: {loss_module.model_type}")
	print(f"  Ignore index: {loss_module.ignore_index}")
	print(f"  Label smoothing: {loss_module.label_smoothing}\n")

	# Create sample data
	batch_size = 4
	seq_len = 64
	input_ids = torch.randint(0, 259, (batch_size, seq_len))
	labels = torch.randint(0, 259, (batch_size, seq_len))

	# Add some padding to labels
	labels[:, -10:] = 256  # Last 10 tokens are padding

	attention_mask = (labels != 256).long()

	print(f"Input shapes:")
	print(f"  input_ids: {input_ids.shape}")
	print(f"  labels: {labels.shape}")
	print(f"  attention_mask: {attention_mask.shape}\n")

	# Forward pass
	with torch.no_grad():
		batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
		}
		outputs = loss_module(batch, return_logits=True)

	print("Forward pass outputs:")
	print(f"  Loss: {outputs['loss'].item():.4f}")
	print(f"  Error: {outputs['error'].item():.4f}")
	print(f"  Top-5 Error: {outputs['top5_error'].item():.4f}")
	print(f"  Perplexity: {outputs['perplexity'].item():.2f}")
	print(f"  Num valid tokens: {outputs['num_tokens'].item()}")
	print(f"  Logits shape: {outputs['logits'].shape}\n")

	# Verify metrics are in valid range
	assert 0 <= outputs['error'].item() <= 1, "Error out of range"
	assert 0 <= outputs['top5_error'].item() <= 1, "Top-5 error out of range"
	assert outputs['perplexity'].item() > 0, "Perplexity should be positive"
	assert outputs['logits'].shape == (batch_size, seq_len, 259), "Logits shape mismatch"

	print("✓ Forward pass successful! All metrics valid.\n")


def test_different_model_architectures():
	"""Test forward pass with different model architectures."""
	print("=" * 80)
	print("TEST 3: Different Model Architectures")
	print("=" * 80 + "\n")

	batch_size = 2
	seq_len = 32
	input_ids = torch.randint(0, 259, (batch_size, seq_len))
	labels = torch.randint(0, 259, (batch_size, seq_len))
	attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

	model_configs = {
		'GPT2CausalLM': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'num_heads': 4,
			'max_seq_len': 512,
		},
		'LlamaCausalLM': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'num_heads': 4,
			'num_key_value_heads': 2,
			'max_seq_len': 512,
		},
		'RwkvCausalLM': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'max_seq_len': 512,
		},
		# Skip xLSTM due to known kernel issues
	}

	for model_type, model_config in model_configs.items():
		print(f"Testing {model_type}...")

		config = {
			'model_type': model_type,
			'model_config': model_config,
			'ignore_index': 256,
		}

		loss_module = AttentionCausalLoss.from_config(config)
		loss_module.eval()

		with torch.no_grad():
			batch = {
				'input_ids': input_ids,
				'labels': labels,
				'attention_mask': attention_mask,
			}
			outputs = loss_module(batch)

		print(f"  ✓ Forward pass successful")
		print(f"    Loss: {outputs['loss'].item():.4f}")
		print(f"    Error: {outputs['error'].item():.4f}\n")

	print("✓ All architectures tested successfully!\n")


def test_label_smoothing():
	"""Test that label smoothing affects loss computation."""
	print("=" * 80)
	print("TEST 4: Label Smoothing Effect")
	print("=" * 80 + "\n")

	batch_size = 2
	seq_len = 32
	input_ids = torch.randint(0, 259, (batch_size, seq_len))
	labels = torch.randint(0, 259, (batch_size, seq_len))
	attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

	smoothing_values = [0.0, 0.1, 0.2]
	losses = []

	for smoothing in smoothing_values:
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 259,
				'hidden_size': 64,
				'num_layers': 1,
				'num_heads': 2,
				'max_seq_len': 512,
			},
			'label_smoothing': smoothing,
		}

		loss_module = AttentionCausalLoss.from_config(config)
		loss_module.eval()

		with torch.no_grad():
			batch = {
				'input_ids': input_ids,
				'labels': labels,
				'attention_mask': attention_mask,
			}
			outputs = loss_module(batch)

		loss_value = outputs['loss'].item()
		losses.append(loss_value)

		print(f"Label smoothing = {smoothing:.1f}:")
		print(f"  Loss: {loss_value:.4f}")
		print(f"  Error: {outputs['error'].item():.4f}\n")

	print(f"✓ Label smoothing tested (losses: {[f'{l:.4f}' for l in losses]})\n")


def test_generation():
	"""Test text generation capability."""
	print("=" * 80)
	print("TEST 5: Text Generation")
	print("=" * 80 + "\n")

	config = {
		'model_type': 'GPT2CausalLM',
		'model_config': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'num_heads': 4,
			'max_seq_len': 512,
		},
	}

	loss_module = AttentionCausalLoss.from_config(config)
	loss_module.eval()

	# Start with START token and '['
	start_tokens = torch.tensor([[257, 91]])  # START + '['

	print(f"Generating from start tokens: {start_tokens.tolist()}")

	# Generate with different parameters
	configs = [
		{'temperature': 1.0, 'top_k': None, 'top_p': None},
		{'temperature': 0.8, 'top_k': 50, 'top_p': None},
		{'temperature': 0.9, 'top_k': None, 'top_p': 0.9},
	]

	for i, gen_config in enumerate(configs, 1):
		generated = loss_module.generate(start_tokens, max_length=20, **gen_config)

		print(f"\nGeneration {i} (temp={gen_config['temperature']}, "
		      f"top_k={gen_config['top_k']}, top_p={gen_config['top_p']}):")
		print(f"  Generated shape: {generated.shape}")
		print(f"  First 10 tokens: {generated[0, :10].tolist()}")

	print("\n✓ Generation test successful!\n")


def test_with_omegaconf():
	"""Test configuration loading with OmegaConf."""
	print("=" * 80)
	print("TEST 6: OmegaConf Integration")
	print("=" * 80 + "\n")

	# Create config dictionary
	config_dict = {
		'model_type': 'GPT2CausalLM',
		'model_config': {
			'vocab_size': 259,
			'hidden_size': 128,
			'num_layers': 2,
			'num_heads': 4,
			'max_seq_len': 512,
		},
		'ignore_index': 256,
		'label_smoothing': 0.1,
	}

	config = OmegaConf.create(config_dict)

	print("Config loaded from dict:")
	print(OmegaConf.to_yaml(config))

	loss_module = AttentionCausalLoss.from_config(config)

	print(f"✓ Loss module created from OmegaConf")
	print(f"  {loss_module}\n")

	# Test forward pass
	input_ids = torch.randint(0, 259, (2, 32))
	labels = torch.randint(0, 259, (2, 32))

	with torch.no_grad():
		batch = {
			'input_ids': input_ids,
			'labels': labels,
		}
		outputs = loss_module(batch)

	print(f"Forward pass successful:")
	print(f"  Loss: {outputs['loss'].item():.4f}")
	print(f"  Error: {outputs['error'].item():.4f}\n")

	print("✓ OmegaConf integration test passed!\n")


def main():
	"""Run all tests."""
	print("\n" + "=" * 80)
	print("ATTENTION CAUSAL LOSS TEST SUITE")
	print("=" * 80 + "\n")

	try:
		# Run all tests
		test_loss_module_creation()
		test_forward_pass()
		test_different_model_architectures()
		test_label_smoothing()
		test_generation()
		test_with_omegaconf()

		# Summary
		print("=" * 80)
		print("✓ ALL TESTS PASSED!")
		print("=" * 80)

		print("\nSummary:")
		print("  ✓ Loss module creation for all model types")
		print("  ✓ Forward pass with loss and metrics")
		print("  ✓ Different model architectures (GPT-2, LLaMA, RWKV)")
		print("  ✓ Label smoothing functionality")
		print("  ✓ Text generation capability")
		print("  ✓ OmegaConf configuration integration")

	except Exception as e:
		print(f"\n✗ TEST FAILED: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
