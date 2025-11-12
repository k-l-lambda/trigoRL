"""LLaMA CausalLM wrapper for TrigoRL."""

from typing import Any, Dict, Union

import torch

try:
	from omegaconf import DictConfig, OmegaConf
except ImportError:
	DictConfig = None
	OmegaConf = None

from transformers import LlamaConfig, LlamaForCausalLM

from trigor.models.registry import register_model


@register_model('LlamaCausalLM')
class LlamaCausalLM(LlamaForCausalLM):
	"""
	LLaMA Causal Language Model wrapper for TrigoRL.

	This class wraps HuggingFace's LlamaForCausalLM with OmegaConf support
	and additional utility methods for model introspection.

	Features:
	- Multi-head attention (MHA), Grouped Query Attention (GQA), or Multi-Query Attention (MQA)
	- RoPE (Rotary Position Embedding)
	- RMSNorm instead of LayerNorm
	- SiLU (Swish) activation
	- No bias in linear layers

	Args:
	    config: LlamaConfig instance

	Example:
	    >>> config = {
	    ...     'vocab_size': 259,
	    ...     'hidden_size': 256,
	    ...     'num_layers': 6,
	    ...     'num_heads': 8,
	    ...     'num_key_value_heads': 2,  # GQA with 4 groups
	    ...     'max_seq_len': 2048,
	    ... }
	    >>> model = LlamaCausalLM.from_config(config)
	"""

	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], 'DictConfig']) -> 'LlamaCausalLM':
		"""
		Create LlamaCausalLM from Hydra configuration.

		Converts Hydra-style config to LlamaConfig and instantiates the model.
		Supports both plain dict and OmegaConf DictConfig.

		Args:
		    config: Configuration dictionary or DictConfig with keys:
		        - vocab_size: Vocabulary size (default: 259)
		        - hidden_size: Hidden dimension (default: 256)
		        - num_layers: Number of transformer layers (default: 6)
		        - num_heads: Number of query attention heads (default: 8)
		        - num_key_value_heads: Number of K/V heads for GQA/MQA (default: num_heads)
		            * Set to num_heads for MHA (standard)
		            * Set to 2-4 for GQA (grouped query attention)
		            * Set to 1 for MQA (multi-query attention)
		        - max_seq_len: Maximum sequence length (default: 2048)
		        - intermediate_size: FFN intermediate dimension (default: hidden_size * 11008 // 4096)
		        - activation: Activation function (default: 'silu')
		        - rms_norm_eps: RMSNorm epsilon (default: 1e-6)
		        - use_cache: Use KV cache for inference (default: True)

		Returns:
		    Instantiated LlamaCausalLM model

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> cfg = OmegaConf.create({
		    ...     'vocab_size': 259,
		    ...     'hidden_size': 256,
		    ...     'num_layers': 6,
		    ...     'num_heads': 8,
		    ...     'num_key_value_heads': 2,  # GQA with 4 groups
		    ... })
		    >>> model = LlamaCausalLM.from_config(cfg)
		"""
		# Convert plain dict to DictConfig for unified API
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Extract parameters with defaults
		vocab_size = config.get('vocab_size', 259)
		hidden_size = config.get('hidden_size', 256)
		num_layers = config.get('num_layers', 6)
		num_heads = config.get('num_heads', 8)
		num_key_value_heads = config.get('num_key_value_heads', num_heads)  # Default to MHA
		max_seq_len = config.get('max_seq_len', 2048)

		# LLaMA's default intermediate_size is ~2.7x hidden_size (11008/4096)
		# Scale proportionally for smaller models
		default_intermediate = int(hidden_size * 11008 / 4096)
		intermediate_size = config.get('intermediate_size', default_intermediate)

		activation = config.get('activation', 'silu')
		rms_norm_eps = config.get('rms_norm_eps', 1e-6)
		use_cache = config.get('use_cache', True)

		# Create LlamaConfig
		model_config = LlamaConfig(
			vocab_size=vocab_size,
			hidden_size=hidden_size,
			intermediate_size=intermediate_size,
			num_hidden_layers=num_layers,
			num_attention_heads=num_heads,
			num_key_value_heads=num_key_value_heads,
			hidden_act=activation,
			max_position_embeddings=max_seq_len,
			rms_norm_eps=rms_norm_eps,
			attention_dropout=0.0,  # LLaMA default
			use_cache=use_cache,
		)

		return cls(model_config)

	def get_model_info(self) -> Dict[str, Any]:
		"""
		Get model architecture information.

		Returns:
		    Dictionary containing model configuration details
		"""
		# Determine attention type
		num_heads = self.config.num_attention_heads
		num_kv_heads = self.config.num_key_value_heads
		if num_kv_heads == num_heads:
			attention_type = 'MHA'
		elif num_kv_heads == 1:
			attention_type = 'MQA'
		else:
			attention_type = f'GQA (groups={num_heads // num_kv_heads})'

		return {
			'model_type': 'llama',
			'vocab_size': self.config.vocab_size,
			'hidden_size': self.config.hidden_size,
			'num_layers': self.config.num_hidden_layers,
			'num_heads': num_heads,
			'num_key_value_heads': num_kv_heads,
			'attention_type': attention_type,
			'max_seq_len': self.config.max_position_embeddings,
			'intermediate_size': self.config.intermediate_size,
			'activation': self.config.hidden_act,
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
		hidden_size = self.config.hidden_size
		num_layers = self.config.num_hidden_layers
		intermediate_size = self.config.intermediate_size

		# Attention: Q, K, V, O
		attention_memory = 4 * batch_size * seq_len * hidden_size
		# FFN: gate, up, down projections
		ffn_memory = 3 * batch_size * seq_len * intermediate_size
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
			f"LlamaCausalLM(\n"
			f"  vocab_size={info['vocab_size']},\n"
			f"  hidden_size={info['hidden_size']},\n"
			f"  num_layers={info['num_layers']},\n"
			f"  num_heads={info['num_heads']},\n"
			f"  attention_type={info['attention_type']},\n"
			f"  max_seq_len={info['max_seq_len']},\n"
			f"  parameters={info['total_parameters']:,}\n"
			f")"
		)
