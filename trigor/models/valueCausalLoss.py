"""
Value Causal Loss module for dual-head training (policy + value).

This module combines causal language modeling with value prediction using RL discount.
Extends AttentionCausalLoss pattern to work with TGNValueDataset.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from trigor.models.registry import make_model, register_model
from trigor.models.valueHead import ValueHead


@register_model('ValueCausalLoss')
class ValueCausalLoss(nn.Module):
	"""
	Dual-head loss module: Policy (next-token prediction) + Value (game outcome prediction).

	Combines causal language modeling with value prediction for game outcome estimation.
	Uses custom attention masks and RL discount (gamma) for temporal credit assignment.

	Architecture:
	- base_model: CausalLM (GPT2/LLaMA/RWKV/xLSTM)
	- value_head: ValueHead network (hidden_dim → [-1, 1])
	- Appends VALUE tokens to input_ids for value prediction
	- Labels are padded (not injected) with ignore_index at VALUE/PAD positions
	- Custom attention mask: Each VALUE token attends up to its corresponding move
	- RL discount: gamma^(N-k-1) for move k in game with N moves

	Args:
	    model_type: Type of base model ('GPT2CausalLM', 'LlamaCausalLM', etc.)
	    model_config: Configuration for the base model
	    value_head_config: Configuration for ValueHead
	    lambda_policy: Policy loss weight (default: 1.0)
	    lambda_value: Value loss weight (default: 0.5)
	    gamma: RL discount factor (default: 0.99)
	    territory_value_factor: Weight for log(|score|) term in value target (default: 1)
	    ignore_index: Token ID to ignore in loss (default: 0 = PAD)
	    label_smoothing: Label smoothing for policy loss (default: 0.0)
	    value_id: VALUE token ID (default: 3)
	    pad_id: PAD token ID (default: 0)

	Value Target Formula:
	    target = sign(score) + log(|score|) * territory_value_factor

	    This separates win/loss direction (sign) from magnitude (log term).
	    The logarithm helps balance targets when scores have different scales.

	Example:
	    >>> config = {
	    ...     'model_type': 'GPT2CausalLM',
	    ...     'model_config': {'vocab_size': 128, 'hidden_size': 256, ...},
	    ...     'value_head_config': {'hidden_dim': 256, ...},
	    ...     'lambda_policy': 1.0,
	    ...     'lambda_value': 0.5,
	    ...     'gamma': 0.99,
	    ... }
	    >>> loss_module = ValueCausalLoss.from_config(config)
	    >>> batch = {...}  # Contains input_ids, labels, value_score, move_end_positions
	    >>> outputs = loss_module(batch)
	    >>> print(outputs['loss'], outputs['policy_loss'], outputs['value_loss'])
	"""

	def __init__(
		self,
		model_type: str,
		model_config: Union[Dict, DictConfig],
		value_head_config: Union[Dict, DictConfig],
		lambda_policy: float = 1.0,
		lambda_value: float = 0.5,
		gamma: float = 0.99,
		territory_value_factor: float = 1.,
		ignore_index: int = 0,
		label_smoothing: float = 0.0,
		value_id: int = 3,
		pad_id: int = 0,
	):
		"""Initialize ValueCausalLoss with dual-head architecture."""
		super().__init__()

		# Store configuration
		self.model_type = model_type
		self.lambda_policy = lambda_policy
		self.lambda_value = lambda_value
		self.gamma = gamma
		self.territory_value_factor = territory_value_factor
		self.ignore_index = ignore_index
		self.label_smoothing = label_smoothing
		self.value_id = value_id
		self.pad_id = pad_id

		# Construct base model using factory
		self.model = make_model(model_type, model_config)

		# Construct value head
		self.value_head = ValueHead.from_config(value_head_config)

		# Create loss functions
		self.policy_loss_fn = nn.CrossEntropyLoss(
			ignore_index=ignore_index,
			label_smoothing=label_smoothing,
			reduction='mean',
		)


	@classmethod
	def from_config(cls, config: Union[Dict, DictConfig]) -> 'ValueCausalLoss':
		"""
		Create ValueCausalLoss from configuration.

		Supports two configuration formats:

		Format 1 (Nested - Recommended):
		    config:
		        model_config:
		            type: GPT2CausalLM
		            config:
		                vocab_size: 128
		                hidden_size: 256
		                ...
		        value_head_config:
		            hidden_dim: 256
		            ...
		        lambda_policy: 1.0
		        lambda_value: 0.5
		        gamma: 0.99

		Format 2 (Flat - Backward Compatible):
		    model_type: GPT2CausalLM
		    model_config:
		        vocab_size: 128
		        hidden_size: 256
		        ...
		    value_head_config:
		        hidden_dim: 256
		        ...
		    lambda_policy: 1.0
		    lambda_value: 0.5
		    gamma: 0.99

		Args:
		    config: Configuration dictionary or DictConfig

		Returns:
		    Initialized ValueCausalLoss module

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> config = OmegaConf.load('config/model/value_causal_loss.yaml')
		    >>> loss_module = ValueCausalLoss.from_config(config)
		"""
		# Convert to DictConfig if plain dict
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Detect which format is being used
		if 'model_config' in config and 'type' in config.model_config:
			# Format 1: Nested structure
			model_type = config.model_config.type
			model_config = config.model_config.config
		elif 'model_type' in config:
			# Format 2: Flat structure
			model_type = config.model_type
			model_config = config.model_config
		else:
			raise ValueError(
				"Config must contain either 'model_config.type' (nested format) "
				"or 'model_type' (flat format)"
			)

		# Extract value head config
		value_head_config = config.get('value_head_config', {})

		# Extract loss parameters with defaults
		lambda_policy = config.get('lambda_policy', 1.0)
		lambda_value = config.get('lambda_value', 0.5)
		gamma = config.get('gamma', 0.99)
		territory_value_factor = config.get('territory_value_factor', 0.01)
		ignore_index = config.get('ignore_index', 0)
		label_smoothing = config.get('label_smoothing', 0.0)
		value_id = config.get('value_id', 3)
		pad_id = config.get('pad_id', 0)

		return cls(
			model_type=model_type,
			model_config=model_config,
			value_head_config=value_head_config,
			lambda_policy=lambda_policy,
			lambda_value=lambda_value,
			gamma=gamma,
			territory_value_factor=territory_value_factor,
			ignore_index=ignore_index,
			label_smoothing=label_smoothing,
			value_id=value_id,
			pad_id=pad_id,
		)


	def _inject_value_tokens(
		self,
		input_ids: torch.Tensor,  # [batch, seq_len]
		move_end_positions: List[torch.Tensor],  # Variable length per sample
	) -> torch.Tensor:
		"""
		Append VALUE tokens to sequence tail.

		Injects VALUE tokens at the end of each sequence. The number of VALUE tokens
		equals the number of moves in the game. The i-th VALUE token corresponds to
		the i-th move and will be used to predict the game value from that position.

		Args:
		    input_ids: Input token IDs [batch, seq_len]
		    move_end_positions: List of tensors, each containing move end positions

		Returns:
		    [batch, new_seq_len] with VALUE tokens appended

		Example:
		    Input:  [START, move1, move2, END, PAD, PAD]
		    Moves:  3 moves in game
		    Output: [START, move1, move2, END, VALUE, VALUE, VALUE]
		"""
		batch_size = input_ids.shape[0]
		device = input_ids.device
		new_sequences = []

		for i in range(batch_size):
			seq = input_ids[i]
			num_moves = len(move_end_positions[i])

			# Append num_moves VALUE tokens
			value_tokens = torch.full(
				(num_moves,),
				self.value_id,
				dtype=torch.long,
				device=device
			)
			new_seq = torch.cat([seq, value_tokens], dim=0)
			new_sequences.append(new_seq)

		# Pad to max length in batch
		max_len = max(seq.shape[0] for seq in new_sequences)
		padded_sequences = []
		pad_value = self.pad_id

		for seq in new_sequences:
			padding_len = max_len - seq.shape[0]
			if padding_len > 0:
				seq = F.pad(seq, (0, padding_len), value=pad_value)
			padded_sequences.append(seq)

		return torch.stack(padded_sequences, dim=0)


	def _pad_labels(
		self,
		labels: torch.Tensor,  # [batch, seq_len]
		input_ids_with_values: torch.Tensor,  # [batch, extended_seq_len]
	) -> torch.Tensor:
		"""
		Pad labels to match extended input_ids length (without injecting VALUE tokens).

		The positions corresponding to VALUE tokens and padding will be set to ignore_index.

		Args:
		    labels: Original labels [batch, seq_len]
		    input_ids_with_values: Input IDs with VALUE tokens appended [batch, extended_seq_len]

		Returns:
		    Padded labels [batch, extended_seq_len] with ignore_index at VALUE/PAD positions
		"""
		batch_size = labels.shape[0]
		extended_seq_len = input_ids_with_values.shape[1]
		device = labels.device

		# Create padded labels filled with ignore_index
		labels_padded = torch.full(
			(batch_size, extended_seq_len),
			self.ignore_index,
			dtype=labels.dtype,
			device=device
		)

		# Copy original labels to the beginning
		original_seq_len = labels.shape[1]
		labels_padded[:, :original_seq_len] = labels

		return labels_padded


	def _create_value_attention_mask(
		self,
		input_ids: torch.Tensor,  # [batch, total_seq_len]
		move_end_positions: List[torch.Tensor],  # List of [num_moves_i]
	) -> torch.Tensor:
		"""
		Create custom attention mask for VALUE tokens.

		Creates a causal attention mask where:
		- Non-VALUE tokens use standard causal attention (attend to all previous tokens)
		- VALUE token i attends only up to move_end_positions[i] (prevents future info leakage)

		Args:
		    input_ids: Input token IDs with VALUE tokens [batch, total_seq_len]
		    move_end_positions: List of move end positions per sample

		Returns:
		    Attention mask [batch, 1, total_seq_len, total_seq_len]

		Visualization:
		    Position:  0  1  2  3  4  5  6  7
		    Token:    [S  m₀ m₁ m₂ E  V₀ V₁ V₂]
		    move_end_positions = [1, 2, 3]

		    V₀ row: [1  1  0  0  0  0  0  0]  ← Attends up to move 0 end (pos 1)
		    V₁ row: [1  1  1  0  0  0  0  0]  ← Attends up to move 1 end (pos 2)
		    V₂ row: [1  1  1  1  0  0  0  0]  ← Attends up to move 2 end (pos 3)
		"""
		batch_size, total_seq_len = input_ids.shape
		device = input_ids.device

		# Start with standard causal mask
		causal_mask = torch.tril(
			torch.ones(total_seq_len, total_seq_len, device=device)
		)
		attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

		# Modify for VALUE tokens
		for batch_idx in range(batch_size):
			# Find VALUE token positions
			value_positions = (input_ids[batch_idx] == self.value_id).nonzero(as_tuple=True)[0]
			move_ends = move_end_positions[batch_idx]

			for move_idx, value_pos in enumerate(value_positions):
				if move_idx >= len(move_ends):
					# Safety: if more VALUE tokens than moves, use causal mask
					continue

				move_end_pos = move_ends[move_idx].item()

				# Clear the row and set up to move_end_pos
				attention_mask[batch_idx, value_pos, :] = 0
				attention_mask[batch_idx, value_pos, :move_end_pos+1] = 1

		# Add head dimension: [batch, 1, seq_len, seq_len]
		return attention_mask.unsqueeze(1)


	def _extract_value_hidden_states(
		self,
		hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
		input_ids: torch.Tensor,  # [batch, seq_len]
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Extract hidden states at VALUE token positions.

		Args:
		    hidden_states: Model hidden states [batch, seq_len, hidden_dim]
		    input_ids: Input token IDs [batch, seq_len]

		Returns:
		    value_hiddens: [total_value_tokens, hidden_dim]
		    value_indices: [total_value_tokens, 2] (batch_idx, pos_idx pairs)

		Example:
		    hidden_states: [batch=4, seq=50, hidden=256]
		    Game 0: 3 VALUE tokens at positions [47, 48, 49]
		    Game 1: 5 VALUE tokens at positions [45, 46, 47, 48, 49]
		    Game 2: 2 VALUE tokens at positions [48, 49]
		    Game 3: 4 VALUE tokens at positions [46, 47, 48, 49]

		    value_indices: [14, 2]  (14 total VALUE tokens)
		    value_hiddens: [14, 256]
		"""
		# Find all VALUE token positions
		value_mask = (input_ids == self.value_id)  # [batch, seq_len]
		value_indices = value_mask.nonzero(as_tuple=False)  # [num_values, 2]

		if len(value_indices) == 0:
			# No VALUE tokens found
			return torch.empty(0, hidden_states.shape[-1], device=hidden_states.device), \
			       torch.empty(0, 2, dtype=torch.long, device=hidden_states.device)

		# Gather hidden states
		batch_idxs = value_indices[:, 0]
		pos_idxs = value_indices[:, 1]
		value_hiddens = hidden_states[batch_idxs, pos_idxs]  # [num_values, hidden_dim]

		return value_hiddens, value_indices


	def _expand_value_targets(
		self,
		value_score: torch.Tensor,  # [batch]
		move_end_positions: List[torch.Tensor],  # Variable length per sample
	) -> torch.Tensor:
		"""
		Expand scalar value_score to per-move targets using logarithmic transformation.

		Formula: target = sign(score) + log(|score|) * territory_value_factor

		This separates win/loss direction from magnitude:
		- sign(score) captures the win/loss direction
		- log(|score|) * factor adds magnitude information (balanced for different scales)

		Args:
		    value_score: Final game outcome per sample [batch]
		    move_end_positions: List of move end positions per sample

		Returns:
		    [total_value_tokens] with expanded targets

		Example (territory_value_factor=0.1):
		    value_score: [10.0, -5.0, 20.0]

		    Targets:
		    - 10.0  -> sign=1.0,  log(10)=2.30, target = 1.0 + 2.30*0.1 = 1.23
		    - -5.0  -> sign=-1.0, log(5)=1.61,  target = -1.0 - 1.61*0.1 = -1.161
		    - 20.0  -> sign=1.0,  log(20)=3.00, target = 1.0 + 3.00*0.1 = 1.3
		"""
		value_targets = []

		for batch_idx, move_positions in enumerate(move_end_positions):
			num_moves = len(move_positions)
			score = value_score[batch_idx]
			target_base = torch.sgn(score).item()
			target = target_base * (1 + torch.log(torch.abs(score)).item()) * self.territory_value_factor

			# Repeat for all moves
			targets = torch.full(
				(num_moves,),
				target,
				dtype=torch.float32,
				device=value_score.device
			)
			value_targets.append(targets)

		if len(value_targets) == 0:
			return torch.tensor([], dtype=torch.float32, device=value_score.device)

		return torch.cat(value_targets, dim=0)


	def _compute_discount_weights(
		self,
		move_end_positions: List[torch.Tensor],
	) -> torch.Tensor:
		"""
		Compute RL discount factors for all VALUE tokens.

		Uses exponential decay from game end: gamma^(N-k-1) for move k.
		These factors are multiplied into value targets to represent discounted future rewards.

		Args:
		    move_end_positions: List of move end positions per sample

		Returns:
		    [total_value_tokens] with discount factors

		Example (gamma=0.99, N=5 moves):
		    Move 0 (first):  gamma^4 = 0.9606  (most discounted)
		    Move 1:          gamma^3 = 0.9703
		    Move 2:          gamma^2 = 0.9801
		    Move 3:          gamma^1 = 0.9900
		    Move 4 (last):   gamma^0 = 1.0000  (full weight)

		The value target at move k becomes: V(s_k) = gamma^(N-k-1) * z
		where z is the final game outcome.
		"""
		all_weights = []

		for move_positions in move_end_positions:
			num_moves = len(move_positions)

			if num_moves == 0:
				continue

			# Compute gamma^(N-k-1) for k in [0, N-1]
			exponents = torch.arange(
				num_moves - 1, -1, -1,
				dtype=torch.float32,
				device=move_positions.device if len(move_positions) > 0 else 'cpu'
			)
			weights = self.gamma ** exponents  # [num_moves]

			all_weights.append(weights)

		if len(all_weights) == 0:
			return torch.tensor([], dtype=torch.float32)

		return torch.cat(all_weights, dim=0)


	def forward(
		self,
		batch: Optional[Dict[str, torch.Tensor]] = None,
		return_logits: bool = False,
		**kwargs
	) -> Dict[str, torch.Tensor]:
		"""
		Dual-head forward pass with custom VALUE token attention.

		Args:
		    batch: Dictionary containing:
		        - input_ids: Input token IDs [batch, seq_len]
		        - labels: Target token IDs [batch, seq_len]
		        - attention_mask: Attention mask [batch, seq_len]
		        - value_score: Final game outcomes [batch]
		        - move_end_positions: List of move end positions per sample
		    return_logits: Whether to return logits and predictions (default: False)
		    **kwargs: Alternative way to pass individual arguments (backward compatibility)

		Returns:
		    Dictionary containing:
		        - loss: Total weighted loss (scalar)
		        - policy_loss: Policy CE loss (scalar)
		        - value_loss: Value MSE loss (scalar)
		        - policy_error: 1 - policy_accuracy (scalar)
		        - value_mae: Value mean absolute error (scalar)
		        - value_mse: Value mean squared error (scalar)
		        - num_policy_tokens: Number of valid policy tokens (scalar)
		        - num_value_predictions: Number of VALUE predictions (scalar)
		        - logits: Model logits [batch, seq_len, vocab_size] (if return_logits=True)
		        - value_predictions: Value predictions [total_value_tokens] (if return_logits=True)

		Example:
		    >>> # New API (preferred)
		    >>> batch = {
		    ...     'input_ids': torch.randint(4, 128, (2, 20)),
		    ...     'labels': torch.randint(4, 128, (2, 20)),
		    ...     'attention_mask': torch.ones(2, 20),
		    ...     'value_score': torch.tensor([1.0, -0.5]),
		    ...     'move_end_positions': [torch.tensor([5, 10, 15]), torch.tensor([7, 14])],
		    ... }
		    >>> outputs = model(batch)
		    >>> loss = outputs['loss']
		    >>> loss.backward()
		"""

		input_ids = batch['input_ids']
		labels = batch['labels']
		value_score = batch['value_score']
		move_end_positions = batch['move_end_positions']

		# Step 1: Inject VALUE tokens into input_ids only
		input_ids_with_values = self._inject_value_tokens(input_ids, move_end_positions)

		# Step 2: Pad labels to match extended sequence (without VALUE token injection)
		labels_padded = self._pad_labels(labels, input_ids_with_values)

		# Step 3: Create custom attention mask
		attention_mask_custom = self._create_value_attention_mask(
			input_ids_with_values,
			move_end_positions
		)

		# Step 4: Forward through base model
		outputs = self.model(
			input_ids_with_values,
			attention_mask=attention_mask_custom,
			output_hidden_states=True
		)
		logits = outputs.logits  # [batch, extended_seq_len, vocab_size]
		hidden_states = outputs.hidden_states[-1]  # [batch, extended_seq_len, hidden_dim]

		# Step 5: Extract VALUE hidden states
		value_hiddens, _ = self._extract_value_hidden_states(
			hidden_states,
			input_ids_with_values
		)

		# Step 6: Predict values
		if len(value_hiddens) > 0:
			value_predictions = self.value_head(value_hiddens)  # [total_value_tokens]
		else:
			value_predictions = torch.tensor([], device=logits.device)

		# Step 7: Compute policy loss (labels_padded already has ignore_index at VALUE/PAD positions)
		logits_flat = logits.view(-1, logits.shape[-1])
		labels_flat = labels_padded.view(-1)

		policy_loss = self.policy_loss_fn(logits_flat, labels_flat)

		# Step 8: Compute value loss
		if len(value_predictions) > 0:
			# Expand value_score to per-move targets
			value_targets = self._expand_value_targets(value_score, move_end_positions)

			# Compute discount weights and apply to targets
			discount_weights = self._compute_discount_weights(move_end_positions)
			discounted_targets = value_targets * discount_weights

			# MSE loss with discounted targets
			value_loss = F.mse_loss(value_predictions, discounted_targets)
		else:
			value_loss = torch.tensor(0.0, device=logits.device)

		# Step 9: Combine losses
		total_loss = self.lambda_policy * policy_loss + self.lambda_value * value_loss

		# Step 10: Compute metrics
		with torch.no_grad():
			# Policy metrics
			predictions = torch.argmax(logits_flat, dim=-1)
			valid_mask = labels_flat != self.ignore_index
			if valid_mask.sum() > 0:
				policy_accuracy = ((predictions == labels_flat) & valid_mask).float().sum() / valid_mask.sum()
			else:
				policy_accuracy = torch.tensor(0.0, device=logits.device)

			# Value metrics
			if len(value_predictions) > 0:
				value_mae = (value_predictions - discounted_targets).abs().mean()
				value_mse = ((value_predictions - discounted_targets) ** 2).mean()
			else:
				value_mae = torch.tensor(0.0, device=logits.device)
				value_mse = torch.tensor(0.0, device=logits.device)

		# Return outputs
		output_dict = {
			'loss': total_loss,
			'policy_loss': policy_loss,
			'value_loss': value_loss,
			'policy_error': 1 - policy_accuracy,
			'value_mae': value_mae,
			'value_mse': value_mse,
			'num_policy_tokens': valid_mask.sum(),
			'num_value_predictions': torch.tensor(len(value_predictions), device=logits.device),
		}

		if return_logits:
			output_dict['logits'] = logits
			output_dict['value_predictions'] = value_predictions

		return output_dict


	def get_model_info(self) -> Dict:
		"""
		Get information about the wrapped models.

		Returns:
		    Dictionary with model information
		"""
		info = {
			'model_type': self.model_type,
			'lambda_policy': self.lambda_policy,
			'lambda_value': self.lambda_value,
			'gamma': self.gamma,
			'territory_value_factor': self.territory_value_factor,
			'ignore_index': self.ignore_index,
			'label_smoothing': self.label_smoothing,
			'value_id': self.value_id,
			'pad_id': self.pad_id,
		}

		# Add model-specific info if available
		if hasattr(self.model, 'get_model_info'):
			info['base_model_info'] = self.model.get_model_info()

		if hasattr(self.value_head, 'get_model_info'):
			info['value_head_info'] = self.value_head.get_model_info()

		return info


	def count_parameters(self) -> Dict[str, int]:
		"""
		Count model parameters.

		Returns:
		    Dictionary with parameter counts for each component
		"""
		# Base model parameters
		base_total = sum(p.numel() for p in self.model.parameters())
		base_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

		# Value head parameters
		value_total = sum(p.numel() for p in self.value_head.parameters())
		value_trainable = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)

		# Total
		total = base_total + value_total
		trainable = base_trainable + value_trainable

		return {
			'total': total,
			'trainable': trainable,
			'non_trainable': total - trainable,
			'base_model': {
				'total': base_total,
				'trainable': base_trainable,
			},
			'value_head': {
				'total': value_total,
				'trainable': value_trainable,
			},
		}


	def __repr__(self) -> str:
		"""String representation."""
		params = self.count_parameters()
		return (
			f"ValueCausalLoss(\n"
			f"  model_type={self.model_type},\n"
			f"  total_parameters={params['total']:,},\n"
			f"  base_model_parameters={params['base_model']['total']:,},\n"
			f"  value_head_parameters={params['value_head']['total']:,},\n"
			f"  lambda_policy={self.lambda_policy},\n"
			f"  lambda_value={self.lambda_value},\n"
			f"  gamma={self.gamma},\n"
			f"  territory_value_factor={self.territory_value_factor}\n"
			f")"
		)
