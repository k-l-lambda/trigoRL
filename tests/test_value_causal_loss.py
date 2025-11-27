"""
Unit tests for ValueCausalLoss module.

Tests dual-head loss module with VALUE token injection, custom attention masks,
RL discount, and combined policy + value loss computation.
"""

import pytest
import torch
from omegaconf import OmegaConf

from trigor.models import ValueCausalLoss, make_model, list_models


class TestValueCausalLossCreation:
	"""Test module creation and initialization."""

	def test_module_creation_with_gpt2(self):
		"""Test creating ValueCausalLoss with GPT2 base model."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 256,
				'num_layers': 2,
				'num_heads': 4,
			},
			'value_head_config': {
				'hidden_dim': 256,
				'intermediate_dim': 128,
				'bottleneck_dim': 32,
			},
			'lambda_policy': 1.0,
			'lambda_value': 0.5,
			'gamma': 0.99,
		}

		model = ValueCausalLoss.from_config(config)

		assert model.model_type == 'GPT2CausalLM'
		assert model.lambda_policy == 1.0
		assert model.lambda_value == 0.5
		assert model.gamma == 0.99


	def test_from_config_dict(self):
		"""Test from_config with plain dictionary."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {
				'hidden_dim': 128,
			},
		}

		model = ValueCausalLoss.from_config(config)
		assert isinstance(model, ValueCausalLoss)


	def test_from_config_omegaconf(self):
		"""Test from_config with OmegaConf."""
		config = OmegaConf.create({
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {
				'hidden_dim': 128,
			},
		})

		model = ValueCausalLoss.from_config(config)
		assert isinstance(model, ValueCausalLoss)


	def test_value_head_integration(self):
		"""Test that ValueHead is properly initialized."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 256,
				'num_layers': 2,
				'num_heads': 4,
			},
			'value_head_config': {
				'hidden_dim': 256,
				'intermediate_dim': 128,
				'bottleneck_dim': 32,
			},
		}

		model = ValueCausalLoss.from_config(config)

		# Check value_head exists and has correct parameters
		assert hasattr(model, 'value_head')
		assert model.value_head.hidden_dim == 256
		assert model.value_head.intermediate_dim == 128
		assert model.value_head.bottleneck_dim == 32


class TestSequenceConstruction:
	"""Test VALUE token injection."""

	def test_inject_value_tokens_basic(self):
		"""Test basic VALUE token injection."""
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

		input_ids = torch.randint(4, 128, (2, 10))  # [batch=2, seq_len=10]
		move_end_positions = [
			torch.tensor([2, 5, 8]),  # 3 moves
			torch.tensor([3, 7]),  # 2 moves
		]

		result = model._inject_value_tokens(input_ids, move_end_positions)

		# Should append max(3, 2) = 3 VALUE tokens
		assert result.shape[0] == 2  # batch size preserved
		assert result.shape[1] == 13  # 10 + 3 VALUE tokens
		assert (result[0, 10:13] == 3).all()  # First game has 3 VALUE tokens
		assert (result[1, 10:12] == 3).all()  # Second game has 2 VALUE tokens
		assert result[1, 12] == 0  # Padding for second game


	def test_inject_value_tokens_variable_moves(self):
		"""Test injection with variable number of moves per game."""
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

		input_ids = torch.randint(4, 128, (3, 20))
		move_end_positions = [
			torch.tensor([5, 10, 15, 19]),  # 4 moves
			torch.tensor([8]),  # 1 move
			torch.tensor([3, 7, 11, 15, 19]),  # 5 moves
		]

		result = model._inject_value_tokens(input_ids, move_end_positions)

		# Should append max(4, 1, 5) = 5 VALUE tokens
		assert result.shape == (3, 25)  # 20 + 5


	def test_inject_value_tokens_empty_moves(self):
		"""Test injection with empty move list."""
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

		input_ids = torch.randint(4, 128, (2, 10))
		move_end_positions = [
			torch.tensor([]),  # No moves
			torch.tensor([5]),  # 1 move
		]

		result = model._inject_value_tokens(input_ids, move_end_positions)

		# Should append max(0, 1) = 1 VALUE token
		assert result.shape == (2, 11)  # 10 + 1
		assert result[0, 10] == 0  # First game gets padding (no VALUE tokens)
		assert result[1, 10] == 3  # Second game gets 1 VALUE token


	def test_sequence_shape_validation(self):
		"""Test tensor shapes throughout sequence construction."""
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

		batch_size = 4
		seq_len = 30
		input_ids = torch.randint(4, 128, (batch_size, seq_len))
		move_end_positions = [
			torch.tensor([10, 20]),
			torch.tensor([5, 15, 25]),
			torch.tensor([12]),
			torch.tensor([8, 18, 28]),
		]

		result = model._inject_value_tokens(input_ids, move_end_positions)

		assert result.shape[0] == batch_size
		assert result.shape[1] > seq_len  # Expanded with VALUE tokens
		assert result.dtype == input_ids.dtype


class TestAttentionMaskConstruction:
	"""Test custom attention mask creation."""

	def test_standard_causal_mask_for_non_value_tokens(self):
		"""Test that non-VALUE tokens use standard causal attention."""
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

		# Create sequence with VALUE tokens at the end
		input_ids = torch.tensor([[1, 10, 20, 30, 2, 3, 3, 3]])  # 1 game, 3 VALUE tokens
		move_end_positions = [torch.tensor([2, 3, 4])]  # 3 moves ending at positions 2, 3, 4

		mask = model._create_value_attention_mask(input_ids, move_end_positions)

		# Check shape
		assert mask.shape == (1, 1, 8, 8)  # [batch, 1, seq, seq]

		# Check standard causal for non-VALUE tokens (positions 0-4)
		for i in range(5):
			for j in range(8):
				if j <= i:
					assert mask[0, 0, i, j] == 1  # Can attend to previous
				else:
					assert mask[0, 0, i, j] == 0  # Cannot attend to future


	def test_value_token_custom_mask(self):
		"""Test that VALUE tokens attend only up to their move position."""
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

		# Position:  0  1  2  3  4  5  6  7
		# Token:    [S m₀ m₁ m₂ E V₀ V₁ V₂]
		input_ids = torch.tensor([[1, 10, 20, 30, 2, 3, 3, 3]])
		move_end_positions = [torch.tensor([1, 2, 3])]  # Moves end at positions 1, 2, 3

		mask = model._create_value_attention_mask(input_ids, move_end_positions)

		# VALUE token 0 (position 5) should attend up to position 1
		assert mask[0, 0, 5, 0] == 1  # Position 0
		assert mask[0, 0, 5, 1] == 1  # Position 1 (move end)
		assert mask[0, 0, 5, 2] == 0  # Position 2 (after move end)

		# VALUE token 1 (position 6) should attend up to position 2
		assert mask[0, 0, 6, 0] == 1
		assert mask[0, 0, 6, 1] == 1
		assert mask[0, 0, 6, 2] == 1  # Position 2 (move end)
		assert mask[0, 0, 6, 3] == 0  # Position 3 (after move end)

		# VALUE token 2 (position 7) should attend up to position 3
		assert mask[0, 0, 7, 0] == 1
		assert mask[0, 0, 7, 1] == 1
		assert mask[0, 0, 7, 2] == 1
		assert mask[0, 0, 7, 3] == 1  # Position 3 (move end)
		assert mask[0, 0, 7, 4] == 0  # Position 4 (after move end)


	def test_mask_4d_shape(self):
		"""Test attention mask has correct 4D shape."""
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

		batch_size = 3
		seq_len = 15
		input_ids = torch.randint(4, 128, (batch_size, seq_len))
		# Set some tokens to VALUE
		input_ids[:, -3:] = 3

		move_end_positions = [
			torch.tensor([5, 8, 11]),
			torch.tensor([6, 9, 12]),
			torch.tensor([4, 7, 10]),
		]

		mask = model._create_value_attention_mask(input_ids, move_end_positions)

		# Should be [batch, 1, seq_len, seq_len]
		assert mask.shape == (batch_size, 1, seq_len, seq_len)
		assert mask.dtype == torch.float32


	def test_move_end_position_alignment(self):
		"""Test mask correctly uses move_end_positions."""
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

		input_ids = torch.tensor([[1, 10, 20, 30, 40, 2, 3, 3]])  # 2 VALUE tokens
		move_end_positions = [torch.tensor([2, 4])]  # Move 0 ends at 2, move 1 at 4

		mask = model._create_value_attention_mask(input_ids, move_end_positions)

		# VALUE token 0 (position 6) attends up to position 2
		assert mask[0, 0, 6, :3].sum() == 3  # Positions 0, 1, 2
		assert mask[0, 0, 6, 3:].sum() == 0

		# VALUE token 1 (position 7) attends up to position 4
		assert mask[0, 0, 7, :5].sum() == 5  # Positions 0, 1, 2, 3, 4
		assert mask[0, 0, 7, 5:].sum() == 0


class TestHiddenStateExtraction:
	"""Test extracting hidden states at VALUE positions."""

	def test_extract_value_hiddens_basic(self):
		"""Test basic extraction of VALUE hidden states."""
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

		hidden_states = torch.randn(2, 10, 128)  # [batch=2, seq=10, hidden=128]
		input_ids = torch.randint(4, 128, (2, 10))
		# Set VALUE tokens at positions [8, 9] for both games
		input_ids[:, 8:10] = 3

		value_hiddens, value_indices = model._extract_value_hidden_states(hidden_states, input_ids)

		# Should extract 4 VALUE tokens (2 per game)
		assert value_hiddens.shape == (4, 128)
		assert value_indices.shape == (4, 2)

		# Check indices are correct
		assert (value_indices[:, 1] >= 8).all()  # Position indices >= 8


	def test_value_positions_tracking(self):
		"""Test that indices correctly track VALUE positions."""
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

		hidden_states = torch.randn(2, 15, 128)
		input_ids = torch.randint(4, 128, (2, 15))
		# Game 0: VALUE at positions [12, 13, 14]
		# Game 1: VALUE at positions [10, 11]
		input_ids[0, 12:15] = 3
		input_ids[1, 10:12] = 3

		value_hiddens, value_indices = model._extract_value_hidden_states(hidden_states, input_ids)

		# Should extract 5 VALUE tokens total
		assert value_hiddens.shape == (5, 128)
		assert value_indices.shape == (5, 2)

		# Check batch indices
		assert (value_indices[:3, 0] == 0).all()  # First 3 from game 0
		assert (value_indices[3:, 0] == 1).all()  # Last 2 from game 1


	def test_no_value_tokens(self):
		"""Test extraction with no VALUE tokens."""
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

		hidden_states = torch.randn(2, 10, 128)
		input_ids = torch.randint(4, 128, (2, 10))  # No VALUE tokens

		value_hiddens, value_indices = model._extract_value_hidden_states(hidden_states, input_ids)

		# Should return empty tensors
		assert value_hiddens.shape == (0, 128)
		assert value_indices.shape == (0, 2)


class TestDiscountComputation:
	"""Test RL discount weight computation."""

	def test_discount_weights_formula(self):
		"""Test gamma^(N-k-1) formula for 5 moves."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
			'gamma': 0.99,
		}
		model = ValueCausalLoss.from_config(config)

		move_end_positions = [torch.tensor([5, 10, 15, 20, 25])]  # 5 moves

		weights = model._compute_discount_weights(move_end_positions)

		# Expected: [gamma^4, gamma^3, gamma^2, gamma^1, gamma^0]
		expected = torch.tensor([0.99**4, 0.99**3, 0.99**2, 0.99**1, 1.0])

		assert weights.shape == (5,)
		assert torch.allclose(weights, expected, atol=1e-6)


	def test_discount_weights_shapes(self):
		"""Test discount weights match number of VALUE tokens."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
			'gamma': 0.99,
		}
		model = ValueCausalLoss.from_config(config)

		move_end_positions = [
			torch.tensor([5, 10, 15]),  # 3 moves
			torch.tensor([8, 16, 24, 32, 40]),  # 5 moves
			torch.tensor([12, 24]),  # 2 moves
		]

		weights = model._compute_discount_weights(move_end_positions)

		# Should have 3 + 5 + 2 = 10 weights
		assert weights.shape == (10,)


	def test_discount_edge_case_gamma_1(self):
		"""Test discount with gamma=1.0 (no discount)."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
			'gamma': 1.0,
		}
		model = ValueCausalLoss.from_config(config)

		move_end_positions = [torch.tensor([5, 10, 15])]

		weights = model._compute_discount_weights(move_end_positions)

		# All weights should be 1.0
		assert torch.allclose(weights, torch.ones(3))


	def test_discount_single_move(self):
		"""Test discount with single move (gamma^0 = 1)."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
			'gamma': 0.95,
		}
		model = ValueCausalLoss.from_config(config)

		move_end_positions = [torch.tensor([10])]  # Single move

		weights = model._compute_discount_weights(move_end_positions)

		# Should be gamma^0 = 1.0
		assert weights.shape == (1,)
		assert torch.allclose(weights, torch.tensor([1.0]))


class TestLossComputation:
	"""Test policy and value loss computation."""

	def test_policy_loss_ignores_value_tokens(self):
		"""Test that policy loss ignores VALUE tokens."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Check that policy loss is computed
		assert 'policy_loss' in outputs
		assert outputs['policy_loss'].item() > 0


	def test_value_loss_with_discounts(self):
		"""Test value loss computation with discount weighting."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Check that value loss is computed
		assert 'value_loss' in outputs
		assert outputs['value_loss'].item() >= 0


	def test_combined_loss_weighting(self):
		"""Test that combined loss uses lambda weights."""
		config = {
			'model_type': 'GPT2CausalLM',
			'model_config': {
				'vocab_size': 128,
				'hidden_size': 128,
				'num_layers': 2,
				'num_heads': 2,
			},
			'value_head_config': {'hidden_dim': 128},
			'lambda_policy': 2.0,
			'lambda_value': 0.5,
		}
		model = ValueCausalLoss.from_config(config)
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Total loss should be approximately: 2.0 * policy_loss + 0.5 * value_loss
		expected_loss = 2.0 * outputs['policy_loss'] + 0.5 * outputs['value_loss']
		assert torch.allclose(outputs['loss'], expected_loss, atol=1e-5)


class TestForwardPass:
	"""Test complete forward pass."""

	def test_forward_basic(self):
		"""Test basic forward pass."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Check all expected outputs
		assert 'loss' in outputs
		assert 'policy_loss' in outputs
		assert 'value_loss' in outputs
		assert 'policy_error' in outputs
		assert 'value_mae' in outputs
		assert 'value_mse' in outputs
		assert 'num_policy_tokens' in outputs
		assert 'num_value_predictions' in outputs


	def test_forward_with_batch(self):
		"""Test forward pass with batch size > 1."""
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
		model.eval()

		batch_size = 4
		input_ids = torch.randint(4, 128, (batch_size, 30))
		labels = torch.randint(4, 128, (batch_size, 30))
		attention_mask = torch.ones(batch_size, 30)
		value_score = torch.randn(batch_size)
		move_end_positions = [torch.randint(5, 25, (3,)) for _ in range(batch_size)]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		assert outputs['loss'].item() > 0
		assert outputs['num_value_predictions'].item() == batch_size * 3  # 4 games * 3 moves each


	def test_forward_variable_games(self):
		"""Test forward with different numbers of moves per game."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (3, 25))
		labels = torch.randint(4, 128, (3, 25))
		attention_mask = torch.ones(3, 25)
		value_score = torch.tensor([1.0, -0.5, 0.8])
		move_end_positions = [
			torch.tensor([5, 10, 15, 20]),  # 4 moves
			torch.tensor([8, 16]),  # 2 moves
			torch.tensor([6, 12, 18]),  # 3 moves
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Should have 4 + 2 + 3 = 9 value predictions
		assert outputs['num_value_predictions'].item() == 9


	def test_backward_pass(self):
		"""Test that backward pass computes gradients."""
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

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)
		loss = outputs['loss']
		loss.backward()

		# Check that gradients are computed
		for param in model.parameters():
			if param.requires_grad:
				assert param.grad is not None


class TestMetrics:
	"""Test metric computation."""

	def test_policy_metrics(self):
		"""Test policy accuracy/error computation."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Check policy metrics
		assert 'policy_error' in outputs
		assert 0.0 <= outputs['policy_error'].item() <= 1.0


	def test_value_metrics(self):
		"""Test value MAE/MSE computation."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Check value metrics
		assert 'value_mae' in outputs
		assert 'value_mse' in outputs
		assert outputs['value_mae'].item() >= 0
		assert outputs['value_mse'].item() >= 0


	def test_metric_ranges(self):
		"""Test that all metrics are in valid ranges."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		# Policy error in [0, 1]
		assert 0.0 <= outputs['policy_error'].item() <= 1.0

		# Value MAE and MSE >= 0
		assert outputs['value_mae'].item() >= 0
		assert outputs['value_mse'].item() >= 0

		# Loss values should be positive
		assert outputs['loss'].item() >= 0
		assert outputs['policy_loss'].item() >= 0
		assert outputs['value_loss'].item() >= 0


class TestIntegration:
	"""Integration tests."""

	def test_registry_integration(self):
		"""Test that ValueCausalLoss is registered."""
		from trigor.models import list_models

		models = list_models()
		assert 'ValueCausalLoss' in models


	def test_make_model_factory(self):
		"""Test creating via make_model factory."""
		from trigor.models import make_model

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

		model = make_model('ValueCausalLoss', config)
		assert isinstance(model, ValueCausalLoss)


	def test_evaluation_mode(self):
		"""Test model in eval mode."""
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

		# Switch to eval mode
		model.eval()
		assert not model.training

		# Forward pass should work
		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		assert outputs['loss'].item() >= 0


class TestEdgeCases:
	"""Test edge cases and error handling."""

	def test_single_move_game(self):
		"""Test with single move per game."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 10))
		labels = torch.randint(4, 128, (2, 10))
		attention_mask = torch.ones(2, 10)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5]),  # Single move
			torch.tensor([6]),  # Single move
		]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		assert outputs['num_value_predictions'].item() == 2


	def test_homogeneous_batch(self):
		"""Test batch where all games have same number of moves."""
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
		model.eval()

		batch_size = 4
		input_ids = torch.randint(4, 128, (batch_size, 20))
		labels = torch.randint(4, 128, (batch_size, 20))
		attention_mask = torch.ones(batch_size, 20)
		value_score = torch.randn(batch_size)
		# All games have exactly 3 moves
		move_end_positions = [torch.tensor([5, 10, 15]) for _ in range(batch_size)]

		with torch.no_grad():
			batch = {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'value_score': value_score,
			'move_end_positions': move_end_positions,
		}
		outputs = model(batch)

		assert outputs['num_value_predictions'].item() == batch_size * 3


	def test_return_logits(self):
		"""Test returning logits and predictions."""
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
		model.eval()

		input_ids = torch.randint(4, 128, (2, 20))
		labels = torch.randint(4, 128, (2, 20))
		attention_mask = torch.ones(2, 20)
		value_score = torch.tensor([1.0, -0.5])
		move_end_positions = [
			torch.tensor([5, 10, 15]),
			torch.tensor([7, 14]),
		]

		with torch.no_grad():
			batch = {
				'input_ids': input_ids,
				'labels': labels,
				'attention_mask': attention_mask,
				'value_score': value_score,
				'move_end_positions': move_end_positions,
			}
			outputs = model(batch, return_logits=True)

		# Check that logits and predictions are returned
		assert 'logits' in outputs
		assert 'value_predictions' in outputs
		assert outputs['logits'].shape[0] == 2  # batch size
		assert outputs['logits'].shape[2] == 128  # vocab size
