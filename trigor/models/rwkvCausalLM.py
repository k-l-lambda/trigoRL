"""RWKV CausalLM wrapper for TrigoRL."""

from typing import Any, Dict, Union

import torch

try:
	from omegaconf import DictConfig, OmegaConf
except ImportError:
	DictConfig = None
	OmegaConf = None

from transformers import RwkvConfig, RwkvForCausalLM

from trigor.models.registry import register_model


@register_model('RwkvCausalLM')
class RwkvCausalLM(RwkvForCausalLM):
	"""
	RWKV Causal Language Model wrapper for TrigoRL.

	This class wraps HuggingFace's RwkvForCausalLM with OmegaConf support
	and additional utility methods for model introspection.

	Features:
	- Linear attention (O(N·D²) complexity instead of O(N²·D))
	- Time-mixing and channel-mixing blocks
	- Exponential time decay mechanism
	- Recurrent state for autoregressive generation
	- No explicit attention heads

	Args:
	    config: RwkvConfig instance

	Example:
	    >>> config = {
	    ...     'vocab_size': 259,
	    ...     'hidden_size': 256,
	    ...     'num_layers': 6,
	    ...     'max_seq_len': 2048,
	    ... }
	    >>> model = RwkvCausalLM.from_config(config)
	"""

	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], 'DictConfig']) -> 'RwkvCausalLM':
		"""
		Create RwkvCausalLM from Hydra configuration.

		Converts Hydra-style config to RwkvConfig and instantiates the model.
		Supports both plain dict and OmegaConf DictConfig.

		Args:
		    config: Configuration dictionary or DictConfig with keys:
		        - vocab_size: Vocabulary size (default: 259)
		        - hidden_size: Hidden dimension (default: 256)
		        - num_layers: Number of RWKV blocks (default: 6)
		        - max_seq_len: Maximum sequence length (default: 2048)
		        - intermediate_size: FFN intermediate dimension (default: 4 * hidden_size)
		        - layer_norm_eps: Layer norm epsilon (default: 1e-5)
		        - use_cache: Use cache for recurrent state (default: True)

		Returns:
		    Instantiated RwkvCausalLM model

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> cfg = OmegaConf.create({
		    ...     'vocab_size': 259,
		    ...     'hidden_size': 256,
		    ...     'num_layers': 6,
		    ... })
		    >>> model = RwkvCausalLM.from_config(cfg)
		"""
		# Convert plain dict to DictConfig for unified API
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Extract parameters with defaults
		vocab_size = config.get('vocab_size', 259)
		hidden_size = config.get('hidden_size', 256)
		num_layers = config.get('num_layers', 6)
		max_seq_len = config.get('max_seq_len', 2048)
		intermediate_size = config.get('intermediate_size', 4 * hidden_size)
		layer_norm_eps = config.get('layer_norm_eps', 1e-5)
		use_cache = config.get('use_cache', True)

		# Create RwkvConfig
		model_config = RwkvConfig(
			vocab_size=vocab_size,
			context_length=max_seq_len,
			hidden_size=hidden_size,
			num_hidden_layers=num_layers,
			attention_hidden_size=hidden_size,  # Usually same as hidden_size
			intermediate_size=intermediate_size,
			layer_norm_epsilon=layer_norm_eps,
			use_cache=use_cache,
		)

		return cls(model_config)

	def get_model_info(self) -> Dict[str, Any]:
		"""
		Get model architecture information.

		Returns:
		    Dictionary containing model configuration details
		"""
		return {
			'model_type': 'rwkv',
			'vocab_size': self.config.vocab_size,
			'hidden_size': self.config.hidden_size,
			'num_layers': self.config.num_hidden_layers,
			'attention_hidden_size': self.config.attention_hidden_size,
			'max_seq_len': self.config.context_length,
			'intermediate_size': self.config.intermediate_size,
			'attention_type': 'Linear (RWKV)',
			'total_parameters': self.count_parameters()['total'],
			'trainable_parameters': self.count_parameters()['trainable'],
		}

	def count_parameters(self) -> Dict[str, int]:
		"""
		Count total and trainable parameters.

		Returns:
		    Dictionary with 'total' and 'trainable' parameter counts
		"""
		total = sum(p.numel() for p in self.parameters())
		trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return {'total': total, 'trainable': trainable}

	def get_memory_footprint(self, batch_size: int = 1, seq_len: int = 2048) -> Dict[str, float]:
		"""
		Estimate memory footprint in MB.

		Args:
		    batch_size: Batch size for estimation
		    seq_len: Sequence length for estimation

		Returns:
		    Dictionary with memory estimates (parameters, activations, total)
		"""
		# Parameter memory (FP32)
		param_memory = self.count_parameters()['total'] * 4 / (1024**2)

		# Activation memory estimate (rough)
		# RWKV has linear complexity for attention
		hidden_size = self.config.hidden_size
		num_layers = self.config.num_hidden_layers
		intermediate_size = self.config.intermediate_size

		# Time-mixing: linear attention operations
		time_mixing_memory = 2 * batch_size * seq_len * hidden_size
		# Channel-mixing: FFN-like operations
		channel_mixing_memory = 2 * batch_size * seq_len * intermediate_size
		# Per layer activation
		layer_activation = (time_mixing_memory + channel_mixing_memory) * 4 / (1024**2)
		total_activation = layer_activation * num_layers

		return {
			'parameters_mb': param_memory,
			'activations_mb': total_activation,
			'total_mb': param_memory + total_activation,
		}

	def __repr__(self) -> str:
		"""Return readable string representation."""
		info = self.get_model_info()
		return (
			f"RwkvCausalLM(\n"
			f"  vocab_size={info['vocab_size']},\n"
			f"  hidden_size={info['hidden_size']},\n"
			f"  num_layers={info['num_layers']},\n"
			f"  attention_type={info['attention_type']},\n"
			f"  max_seq_len={info['max_seq_len']},\n"
			f"  parameters={info['total_parameters']:,}\n"
			f")"
		)
