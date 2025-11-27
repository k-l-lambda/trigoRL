"""
Test backward pass for ValueCausalLoss with different dtypes.

Tests that gradients are computed correctly for both float32 and bfloat16.
"""

import pytest
import torch

from trigor.models import ValueCausalLoss


class TestBackwardPass:
	"""Test backward pass with different dtypes."""

	def test_backward_float32(self):
		"""Test backward pass with float32 (default)."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model.train()

		# Create batch
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.randn(batch_size),
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward pass
		outputs = model(batch)
		loss = outputs['loss']

		assert loss.dtype == torch.float32
		assert loss.requires_grad

		# Backward pass
		model.zero_grad()
		loss.backward()

		# Check gradients exist
		for name, param in model.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"
				assert param.grad.dtype == torch.float32, f"Wrong grad dtype for {name}"


	def test_backward_bfloat16(self):
		"""Test backward pass with bfloat16."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model = model.to(dtype=torch.bfloat16)
		model.train()

		# Create batch
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.randn(batch_size).to(dtype=torch.bfloat16),
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward pass
		outputs = model(batch)
		loss = outputs['loss']

		assert loss.dtype == torch.bfloat16
		assert loss.requires_grad

		# Backward pass
		model.zero_grad()
		loss.backward()

		# Check gradients exist and match parameter dtype
		for name, param in model.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"
				assert param.grad.dtype == param.dtype, f"Gradient dtype mismatch for {name}: grad={param.grad.dtype}, param={param.dtype}"


	def test_backward_float16(self):
		"""Test backward pass with float16."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model = model.to(dtype=torch.float16)
		model.train()

		# Create batch
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.randn(batch_size).to(dtype=torch.float16),
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward pass
		outputs = model(batch)
		loss = outputs['loss']

		assert loss.dtype == torch.float16
		assert loss.requires_grad

		# Backward pass
		model.zero_grad()
		loss.backward()

		# Check gradients exist and match parameter dtype
		for name, param in model.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"
				assert param.grad.dtype == param.dtype, f"Gradient dtype mismatch for {name}: grad={param.grad.dtype}, param={param.dtype}"


	def test_gradient_flow(self):
		"""Test that gradients flow through both base model and value head."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model.train()

		# Create batch
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.randn(batch_size),
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward + backward
		model.zero_grad()
		outputs = model(batch)
		outputs['loss'].backward()

		# Collect gradient norms
		base_model_grad_norm = 0.0
		value_head_grad_norm = 0.0

		for name, param in model.named_parameters():
			if param.requires_grad and param.grad is not None:
				grad_norm = param.grad.norm().item()
				if 'value_head' in name:
					value_head_grad_norm += grad_norm ** 2
				else:
					base_model_grad_norm += grad_norm ** 2

		base_model_grad_norm = base_model_grad_norm ** 0.5
		value_head_grad_norm = value_head_grad_norm ** 0.5

		# Both should have gradients
		assert base_model_grad_norm > 0, "Base model should have gradients"
		assert value_head_grad_norm > 0, "Value head should have gradients"


	def test_multiple_backward_passes(self):
		"""Test multiple forward/backward cycles."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model.train()

		batch_size = 2
		seq_len = 20

		for i in range(3):
			batch = {
				'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
				'labels': torch.randint(4, 128, (batch_size, seq_len)),
				'attention_mask': torch.ones(batch_size, seq_len),
				'value_score': torch.randn(batch_size),
				'move_end_positions': [
					torch.tensor([5, 10, 15]),
					torch.tensor([7, 14]),
				],
			}

			# Forward + backward
			model.zero_grad()
			outputs = model(batch)
			loss = outputs['loss']
			loss.backward()

			# Check gradients exist after each iteration
			for name, param in model.named_parameters():
				if param.requires_grad:
					assert param.grad is not None, f"No gradient for {name} in iteration {i}"


	def test_backward_with_zero_score(self):
		"""Test backward pass with zero score (draw/tie game)."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model.train()

		# Create batch with zero score (draw/tie)
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.tensor([0.0, 1.5]),  # First game is draw
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward pass
		outputs = model(batch)
		loss = outputs['loss']

		# Loss should be finite (not nan or inf)
		assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
		assert loss.requires_grad

		# Backward pass should work without errors
		model.zero_grad()
		loss.backward()

		# Check gradients exist and are finite
		for name, param in model.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"
				assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


	def test_backward_with_dtype_mismatch(self):
		"""Test backward with value_score dtype different from model dtype.

		This simulates the real scenario where TGNValueDataset returns float32
		value_score but model is trained in bfloat16/float16.
		"""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
		}
		model = ValueCausalLoss.from_config(config)
		model = model.to(dtype=torch.bfloat16)  # Model in bfloat16
		model.train()

		# Create batch with float32 value_score (like real dataset)
		batch_size = 2
		seq_len = 20
		batch = {
			'input_ids': torch.randint(4, 128, (batch_size, seq_len)),
			'labels': torch.randint(4, 128, (batch_size, seq_len)),
			'attention_mask': torch.ones(batch_size, seq_len),
			'value_score': torch.tensor([6.0, 1.5], dtype=torch.float32),  # float32!
			'move_end_positions': [
				torch.tensor([5, 10, 15]),
				torch.tensor([7, 14]),
			],
		}

		# Forward pass should auto-convert value_score to bfloat16
		outputs = model(batch)
		loss = outputs['loss']

		# Loss should match model dtype (bfloat16)
		assert loss.dtype == torch.bfloat16, f"Loss dtype should be bfloat16, got {loss.dtype}"
		assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
		assert loss.requires_grad

		# Backward pass should work without dtype errors
		model.zero_grad()
		loss.backward()

		# Check gradients exist and match parameter dtype
		for name, param in model.named_parameters():
			if param.requires_grad:
				assert param.grad is not None, f"No gradient for {name}"
				assert param.grad.dtype == param.dtype, f"Gradient dtype mismatch for {name}"
				assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


