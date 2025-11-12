"""GPT-2 CausalLM wrapper for TrigoRL."""

from typing import Any, Dict, Union

import torch

try:
	from omegaconf import DictConfig, OmegaConf
except ImportError:
	DictConfig = None
	OmegaConf = None

from transformers import GPT2Config, GPT2LMHeadModel

from trigor.models.registry import register_model


@register_model('GPT2CausalLM')
class GPT2CausalLM(GPT2LMHeadModel):
	"""
	GPT-2 Causal Language Model wrapper for TrigoRL.

	This class wraps HuggingFace's GPT2LMHeadModel with OmegaConf support
	and additional utility methods for model introspection.

	Features:
	- Standard multi-head attention (MHA)
	- GELU activation
	- Learned positional embeddings
	- Layer normalization

	Args:
	    config: GPT2Config instance

	Example:
	    >>> config = {
	    ...     'vocab_size': 259,
	    ...     'hidden_size': 256,
	    ...     'num_layers': 6,
	    ...     'num_heads': 8,
	    ...     'max_seq_len': 2048,
	    ... }
	    >>> model = GPT2CausalLM.from_config(config)
	"""

	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], 'DictConfig']) -> 'GPT2CausalLM':
		"""
		Create GPT2CausalLM from Hydra configuration.

		Converts Hydra-style config to GPT2Config and instantiates the model.
		Supports both plain dict and OmegaConf DictConfig.

		Args:
		    config: Configuration dictionary or DictConfig with keys:
		        - vocab_size: Vocabulary size (default: 259)
		        - hidden_size: Hidden dimension (default: 256)
		        - num_layers: Number of transformer layers (default: 6)
		        - num_heads: Number of attention heads (default: 8)
		        - max_seq_len: Maximum sequence length (default: 2048)
		        - intermediate_size: FFN intermediate dimension (default: 4 * hidden_size)
		        - activation: Activation function (default: 'gelu_new')
		        - dropout: Dropout rate (default: 0.1)
		        - layer_norm_eps: Layer norm epsilon (default: 1e-5)
		        - use_cache: Use KV cache for inference (default: True)

		Returns:
		    Instantiated GPT2CausalLM model

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> cfg = OmegaConf.create({'vocab_size': 259, 'hidden_size': 256, 'num_layers': 6})
		    >>> model = GPT2CausalLM.from_config(cfg)
		"""
		# Convert plain dict to DictConfig for unified API
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Extract parameters with defaults
		vocab_size = config.get('vocab_size', 259)
		hidden_size = config.get('hidden_size', 256)
		num_layers = config.get('num_layers', 6)
		num_heads = config.get('num_heads', 8)
		max_seq_len = config.get('max_seq_len', 2048)
		intermediate_size = config.get('intermediate_size', 4 * hidden_size)
		activation = config.get('activation', 'gelu_new')
		dropout = config.get('dropout', 0.1)
		layer_norm_eps = config.get('layer_norm_eps', 1e-5)
		use_cache = config.get('use_cache', True)

		# Create GPT2Config with proper parameter names
		model_config = GPT2Config(
			vocab_size=vocab_size,
			n_positions=max_seq_len,
			n_embd=hidden_size,
			n_layer=num_layers,
			n_head=num_heads,
			n_inner=intermediate_size,
			activation_function=activation,
			resid_pdrop=dropout,
			embd_pdrop=dropout,
			attn_pdrop=dropout,
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
			'model_type': 'gpt2',
			'vocab_size': self.config.vocab_size,
			'hidden_size': self.config.n_embd,
			'num_layers': self.config.n_layer,
			'num_heads': self.config.n_head,
			'max_seq_len': self.config.n_positions,
			'intermediate_size': self.config.n_inner,
			'activation': self.config.activation_function,
			'dropout': self.config.resid_pdrop,
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
		# Per layer: attention (Q, K, V, O) + FFN (intermediate, output) + residuals
		hidden_size = self.config.n_embd
		num_layers = self.config.n_layer
		intermediate_size = self.config.n_inner

		# Attention: 4 * (batch * seq * hidden)
		attention_memory = 4 * batch_size * seq_len * hidden_size
		# FFN: 2 * (batch * seq * intermediate)
		ffn_memory = 2 * batch_size * seq_len * intermediate_size
		# Per layer activation
		layer_activation = (attention_memory + ffn_memory) * 4 / (1024**2)
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
			f"GPT2CausalLM(\n"
			f"  vocab_size={info['vocab_size']},\n"
			f"  hidden_size={info['hidden_size']},\n"
			f"  num_layers={info['num_layers']},\n"
			f"  num_heads={info['num_heads']},\n"
			f"  max_seq_len={info['max_seq_len']},\n"
			f"  parameters={info['total_parameters']:,}\n"
			f")"
		)
