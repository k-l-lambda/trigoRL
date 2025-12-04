"""
Tree Language Model for ONNX Export.

This model wraps a base CausalLM model to support tree-mode inference where:
1. Input is prefix_ids (context) + evaluated_ids (targets to evaluate)
2. Causal attention mask is applied by default
3. Custom evaluated_mask can override attention in the evaluated region (bottom-right m×m)
4. Returns logits for last prefix position + all evaluated positions (m+1 total)

This is designed specifically for ONNX export and efficient inference in JavaScript.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .registry import register_model


@register_model("tree")
class TreeLM(nn.Module):
	"""
	Tree mode wrapper for causal language models.

	This model is designed for ONNX export and supports flexible attention masking
	for computing probabilities over evaluation sequences given a prefix context.

	ONNX Input Signature:
	    - prefix_ids: [batch_size, n] - Prefix token sequence
	    - evaluated_ids: [batch_size, m] - Tokens to evaluate
	    - evaluated_mask: [batch_size, m, m] - Custom attention pattern for evaluated region

	ONNX Output:
	    - logits: [batch_size, m + 1, vocab_size] - Logits for last prefix + all evaluated positions

	Args:
	    base_model: Underlying CausalLM model (GPT2, LLaMA, RWKV, xLSTM)
	"""

	def __init__(self, base_model: nn.Module):
		super().__init__()
		self.model = base_model


	def forward(
		self,
		prefix_ids: torch.Tensor,
		evaluated_ids: torch.Tensor,
		evaluated_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		Forward pass with tree mode masking.

		Args:
		    prefix_ids: Prefix token IDs [batch_size, n]
		    evaluated_ids: Evaluated token IDs [batch_size, m]
		    evaluated_mask: Custom attention mask for evaluated region [batch_size, m, m]
		                    This mask overwrites the bottom-right m×m region of the causal mask

		Returns:
		    logits: [batch_size, m+1, vocab_size] - Logits for last prefix + all evaluated positions
		"""
		batch_size, n = prefix_ids.shape
		_, m = evaluated_ids.shape

		# Concatenate prefix and evaluated tokens to form full input sequence
		input_ids = torch.cat([prefix_ids, evaluated_ids], dim=1)  # [batch, n+m]

		# CRITICAL: Calculate position_ids based on tree structure
		# Each evaluated token's position depends on how many tokens it can attend to
		# evaluated_mask[i, :].sum() = number of evaluated tokens that token i can see
		# Total visible tokens = n (prefix) + evaluated_mask[i, :].sum() (evaluated)
		# Position = total_visible - 1 (because position is 0-indexed)
		prefix_positions = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)  # [batch, n]

		# For each evaluated token, compute: position = n + mask_row_sum - 1
		mask_row_sums = evaluated_mask.sum(dim=2)  # [batch, m] - sum over last dimension
		evaluated_positions = (n + mask_row_sums - 1).long()  # [batch, m] - convert to long for embedding

		position_ids = torch.cat([prefix_positions, evaluated_positions], dim=1)  # [batch, n+m]

		# Build base causal mask for entire sequence (n+m) × (n+m)
		total_len = n + m
		causal_mask = torch.tril(
			torch.ones(total_len, total_len, device=input_ids.device, dtype=torch.float32)
		)  # [n+m, n+m]

		# Expand to batch dimension
		combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()  # Clone for in-place modification

		# Overwrite the bottom-right m×m region with evaluated_mask
		# This replaces the default causal pattern in the evaluated region
		combined_mask[:, n:, n:] = evaluated_mask  # [batch, m, m] overwrites tail

		# Convert 0/1 mask to log-space format (0 = attend, -inf = mask)
		# This is the correct format expected by GPT2: mask is added to attention weights
		# Reference: transformers/models/gpt2/modeling_gpt2.py
		mask_value = -float("inf")
		combined_mask = torch.where(
			combined_mask == 1.0,
			torch.tensor(0.0, dtype=torch.float32, device=input_ids.device),
			torch.tensor(mask_value, dtype=torch.float32, device=input_ids.device)
		)

		# Convert to 4D attention mask: [batch_size, 1, seq_len, seq_len]
		attention_mask = combined_mask.unsqueeze(1)

		# Get base model (unwrap if wrapped in AttentionCausalLoss or similar)
		if hasattr(self.model, 'model'):
			base = self.model.model
		else:
			base = self.model

		# Forward pass with custom attention mask and position_ids
		model_outputs = base(input_ids, attention_mask=attention_mask, position_ids=position_ids)

		# Extract logits
		if hasattr(model_outputs, 'logits'):
			logits = model_outputs.logits  # [batch_size, n+m, vocab_size]
		elif isinstance(model_outputs, torch.Tensor):
			logits = model_outputs
		elif isinstance(model_outputs, dict):
			logits = model_outputs['logits']
		elif isinstance(model_outputs, (list, tuple)):
			logits = model_outputs[0]
		else:
			raise ValueError(f"Unsupported model output type: {type(model_outputs)}")

		# Return logits for: last prefix position + all evaluated positions
		# Shape: [batch_size, m+1, vocab_size]
		# Positions: [n-1, n, n+1, ..., n+m-1]
		return logits[:, n-1:, :]


	def get_model_info(self) -> Dict[str, any]:
		"""Get information about the wrapped model."""
		if hasattr(self.model, 'get_model_info'):
			base_info = self.model.get_model_info()
		else:
			base_info = {}

		return {
			'model_class': 'TreeLM',
			'base_model': base_info.get('model_type', 'unknown'),
			'mode': 'tree',
			'onnx_compatible': True,
		}


	@classmethod
	def from_base_model(cls, base_model: nn.Module) -> 'TreeLM':
		"""
		Create TreeLM from a base CausalLM model.

		Args:
		    base_model: Instance of GPT2CausalLM, LlamaCausalLM, RwkvCausalLM, xLSTMCausalLM,
		                or AttentionCausalLoss wrapper

		Returns:
		    TreeLM instance
		"""
		return cls(base_model)


	@classmethod
	def from_config(cls, config: Union[Dict, DictConfig], base_model: nn.Module) -> 'TreeLM':
		"""
		Create TreeLM with configuration.

		Args:
		    config: Configuration dict (currently unused, for compatibility)
		    base_model: Base CausalLM model instance

		Returns:
		    TreeLM instance
		"""
		return cls(base_model)


	def __repr__(self) -> str:
		base_repr = repr(self.model) if self.model else "None"
		return f"TreeLM(\n  base_model={base_repr}\n)"


def create_causal_evaluated_mask(
	m: int,
	device: Optional[torch.device] = None,
	dtype: torch.dtype = torch.float32
) -> torch.Tensor:
	"""
	Create a causal attention mask for evaluated region (standard autoregressive).

	Args:
	    m: Number of evaluated tokens
	    device: Device to create tensor on
	    dtype: Data type for mask

	Returns:
	    Attention mask [1, m, m] with lower triangular pattern
	"""
	# Lower triangular - causal
	mask = torch.tril(torch.ones(m, m, device=device, dtype=dtype))

	# Add batch dimension
	mask = mask.unsqueeze(0)  # [1, m, m]

	return mask


def create_diagonal_evaluated_mask(
	m: int,
	device: Optional[torch.device] = None,
	dtype: torch.dtype = torch.float32
) -> torch.Tensor:
	"""
	Create a diagonal attention mask for evaluated region (each position only attends to itself).

	Args:
	    m: Number of evaluated tokens
	    device: Device to create tensor on
	    dtype: Data type for mask

	Returns:
	    Attention mask [1, m, m] with diagonal pattern (identity matrix)
	"""
	# Diagonal only
	mask = torch.eye(m, device=device, dtype=dtype)

	# Add batch dimension
	mask = mask.unsqueeze(0)  # [1, m, m]

	return mask
