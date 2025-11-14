"""
Attention Causal Loss module for language modeling.

This module combines a causal language model with cross-entropy loss computation,
providing both loss and accuracy metrics for training and evaluation.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from trigor.models.registry import make_model, register_model


@register_model('AttentionCausalLoss')
class AttentionCausalLoss(nn.Module):
	"""
	Attention-based Causal Language Modeling Loss module.

	This class wraps a causal language model (GPT-2, LLaMA, RWKV, xLSTM)
	and provides loss computation and accuracy metrics for next-token prediction.

	The module:
	1. Constructs the model using the model factory
	2. Computes cross-entropy loss between predictions and targets
	3. Calculates accuracy metrics (token-level and sequence-level)
	4. Handles padding tokens properly (ignored in loss and metrics)

	Args:
	    model_type: Type of model to use (e.g., 'GPT2CausalLM', 'LlamaCausalLM')
	    model_config: Configuration for the model
	    ignore_index: Token ID to ignore in loss computation (typically PAD token, default: 256)
	    label_smoothing: Label smoothing factor for cross-entropy loss (default: 0.0)

	Example:
	    >>> config = {
	    ...     'model_type': 'GPT2CausalLM',
	    ...     'model_config': {
	    ...         'vocab_size': 259,
	    ...         'hidden_size': 256,
	    ...         'num_layers': 6,
	    ...         'num_heads': 8,
	    ...         'max_seq_len': 2048,
	    ...     },
	    ...     'ignore_index': 256,  # PAD token
	    ... }
	    >>> loss_module = AttentionCausalLoss.from_config(config)
	    >>> outputs = loss_module(input_ids, labels, attention_mask)
	    >>> print(outputs['loss'], outputs['accuracy'])
	"""

	def __init__(
		self,
		model_type: str,
		model_config: Union[Dict, DictConfig],
		ignore_index: int = 256,
		label_smoothing: float = 0.0,
	):
		"""Initialize the AttentionCausalLoss module."""
		super().__init__()

		# Store configuration
		self.model_type = model_type
		self.ignore_index = ignore_index
		self.label_smoothing = label_smoothing

		# Construct the model using factory
		self.model = make_model(model_type, model_config)

		# Create loss function
		self.loss_fn = nn.CrossEntropyLoss(
			ignore_index=ignore_index,
			label_smoothing=label_smoothing,
			reduction='mean',
		)

	@classmethod
	def from_config(cls, config: Union[Dict, DictConfig]) -> 'AttentionCausalLoss':
		"""
		Create AttentionCausalLoss from configuration.

		Supports two configuration formats:

		Format 1 (Nested - Recommended):
		    config:
		        model_config:
		            type: GPT2CausalLM
		            config:
		                vocab_size: 259
		                hidden_size: 256
		                ...
		        ignore_index: 256
		        label_smoothing: 0.1

		Format 2 (Flat - Backward Compatible):
		    model_type: GPT2CausalLM
		    model_config:
		        vocab_size: 259
		        hidden_size: 256
		        ...
		    ignore_index: 256
		    label_smoothing: 0.1

		Args:
		    config: Configuration dictionary or DictConfig

		Returns:
		    Initialized AttentionCausalLoss module

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> config = OmegaConf.create({
		    ...     'model_config': {
		    ...         'type': 'GPT2CausalLM',
		    ...         'config': {
		    ...             'vocab_size': 259,
		    ...             'hidden_size': 256,
		    ...             'num_layers': 6,
		    ...             'num_heads': 8,
		    ...         }
		    ...     },
		    ...     'ignore_index': 256,
		    ...     'label_smoothing': 0.1,
		    ... })
		    >>> loss_module = AttentionCausalLoss.from_config(config)
		"""
		# Convert to DictConfig if plain dict
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Detect which format is being used
		if 'model_config' in config and 'type' in config.model_config:
			# Format 1: Nested structure (model_config.type + model_config.config)
			model_type = config.model_config.type
			model_config = config.model_config.config
		elif 'model_type' in config:
			# Format 2: Flat structure (backward compatible)
			model_type = config.model_type
			model_config = config.model_config
		else:
			raise ValueError(
				"Config must contain either 'model_config.type' (nested format) "
				"or 'model_type' (flat format)"
			)

		# Extract loss parameters with defaults
		ignore_index = config.get('ignore_index', 256)
		label_smoothing = config.get('label_smoothing', 0.0)

		return cls(
			model_type=model_type,
			model_config=model_config,
			ignore_index=ignore_index,
			label_smoothing=label_smoothing,
		)

	def forward(
		self,
		input_ids: torch.Tensor,
		labels: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		return_logits: bool = False,
	) -> Dict[str, torch.Tensor]:
		"""
		Forward pass: compute loss and metrics.

		Args:
		    input_ids: Input token IDs [batch_size, seq_len]
		    labels: Target token IDs [batch_size, seq_len]
		    attention_mask: Attention mask [batch_size, seq_len] (optional)
		    return_logits: Whether to return model logits (default: False)

		Returns:
		    Dictionary containing:
		        - loss: Cross-entropy loss (scalar)
		        - accuracy: Token-level accuracy (scalar)
		        - perplexity: Perplexity metric (scalar)
		        - top5_accuracy: Top-5 token accuracy (scalar)
		        - num_tokens: Number of valid tokens (scalar)
		        - logits: Model output logits [batch_size, seq_len, vocab_size] (if return_logits=True)

		Example:
		    >>> input_ids = torch.randint(0, 259, (4, 512))
		    >>> labels = torch.randint(0, 259, (4, 512))
		    >>> attention_mask = torch.ones(4, 512)
		    >>> outputs = loss_module(input_ids, labels, attention_mask)
		    >>> loss = outputs['loss']
		    >>> accuracy = outputs['accuracy']
		"""
		# Get model predictions
		model_outputs = self.model(input_ids, attention_mask=attention_mask)
		logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]

		batch_size, seq_len, vocab_size = logits.shape

		# Reshape for loss computation
		# logits: [batch_size * seq_len, vocab_size]
		# labels: [batch_size * seq_len]
		logits_flat = logits.view(-1, vocab_size)
		labels_flat = labels.view(-1)

		# Compute cross-entropy loss
		loss = self.loss_fn(logits_flat, labels_flat)

		# Compute metrics
		with torch.no_grad():
			# Get predictions (most likely token)
			predictions = torch.argmax(logits_flat, dim=-1)

			# Create mask for valid tokens (not padding)
			valid_mask = labels_flat != self.ignore_index

			# Token-level accuracy
			correct = (predictions == labels_flat) & valid_mask
			accuracy = correct.sum().float() / valid_mask.sum().float()
			error = 1 - accuracy

			# Top-5 accuracy
			top5_predictions = torch.topk(logits_flat, k=5, dim=-1).indices
			top5_correct = (top5_predictions == labels_flat.unsqueeze(-1)).any(dim=-1) & valid_mask
			top5_accuracy = top5_correct.sum().float() / valid_mask.sum().float()
			top5_error = 1 - top5_accuracy

			# Perplexity
			perplexity = torch.exp(loss)

			# Number of valid tokens
			num_tokens = valid_mask.sum()

		# Build output dictionary
		outputs = {
			'loss': loss,
			'error': error,
			'perplexity': perplexity,
			'top5_error': top5_error,
			'num_tokens': num_tokens,
		}

		if return_logits:
			outputs['logits'] = logits

		return outputs

	def generate(
		self,
		input_ids: torch.Tensor,
		max_length: int = 100,
		temperature: float = 1.0,
		top_k: Optional[int] = None,
		top_p: Optional[float] = None,
	) -> torch.Tensor:
		"""
		Generate text autoregressively.

		Args:
		    input_ids: Starting token IDs [batch_size, seq_len]
		    max_length: Maximum number of tokens to generate
		    temperature: Sampling temperature (higher = more random)
		    top_k: Keep only top-k tokens for sampling
		    top_p: Nucleus sampling probability threshold

		Returns:
		    Generated token IDs [batch_size, seq_len + max_length]

		Example:
		    >>> start_tokens = torch.tensor([[257, 91]])  # START + '['
		    >>> generated = loss_module.generate(start_tokens, max_length=50)
		"""
		self.model.eval()

		batch_size, start_len = input_ids.shape
		device = input_ids.device

		# Initialize generated sequence with input
		generated = input_ids.clone()

		with torch.no_grad():
			for _ in range(max_length):
				# Get logits for next token
				outputs = self.model(generated)
				logits = outputs.logits  # [batch_size, current_len, vocab_size]

				# Get logits for last position
				next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]

				# Apply top-k filtering
				if top_k is not None:
					indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
					next_token_logits[indices_to_remove] = float('-inf')

				# Apply top-p (nucleus) filtering
				if top_p is not None:
					sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
					cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

					# Remove tokens with cumulative probability above threshold
					sorted_indices_to_remove = cumulative_probs > top_p
					# Shift the indices to the right to keep first token above threshold
					sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
					sorted_indices_to_remove[..., 0] = 0

					# Scatter sorted indices back to original order
					indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
					next_token_logits[indices_to_remove] = float('-inf')

				# Sample next token
				probs = F.softmax(next_token_logits, dim=-1)
				next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

				# Append to generated sequence
				generated = torch.cat([generated, next_token], dim=-1)

				# Stop if all sequences have generated END token (258)
				if (next_token == 258).all():
					break

		return generated

	def get_model_info(self) -> Dict:
		"""
		Get information about the wrapped model.

		Returns:
		    Dictionary with model information
		"""
		info = {
			'model_type': self.model_type,
			'ignore_index': self.ignore_index,
			'label_smoothing': self.label_smoothing,
		}

		# Add model-specific info if available
		if hasattr(self.model, 'get_model_info'):
			info['model_info'] = self.model.get_model_info()

		return info

	def count_parameters(self) -> Dict[str, int]:
		"""
		Count model parameters.

		Returns:
		    Dictionary with parameter counts
		"""
		total = sum(p.numel() for p in self.model.parameters())
		trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

		return {
			'total': total,
			'trainable': trainable,
			'non_trainable': total - trainable,
		}

	def __repr__(self) -> str:
		"""String representation."""
		params = self.count_parameters()
		return (
			f"AttentionCausalLoss(\n"
			f"  model_type={self.model_type},\n"
			f"  parameters={params['total']:,},\n"
			f"  ignore_index={self.ignore_index},\n"
			f"  label_smoothing={self.label_smoothing}\n"
			f")"
		)
