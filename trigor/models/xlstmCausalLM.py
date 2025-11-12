"""xLSTM CausalLM wrapper for TrigoRL."""

from typing import Any, Dict, Union

import torch

try:
	from omegaconf import DictConfig, OmegaConf
except ImportError:
	DictConfig = None
	OmegaConf = None

from transformers import xLSTMConfig, xLSTMForCausalLM

from trigor.models.registry import register_model


@register_model('xLSTMCausalLM')
class xLSTMCausalLM(xLSTMForCausalLM):
	"""
	xLSTM Causal Language Model wrapper for TrigoRL.

	This class wraps HuggingFace's xLSTMForCausalLM with OmegaConf support
	and additional utility methods for model introspection.

	Features:
	- Extended LSTM with matrix-valued cell states
	- Exponential gating (log-space) instead of sigmoid
	- Multi-head architecture
	- Chunk-wise parallelization for efficiency
	- Modern LSTM variant with better performance

	Args:
	    config: xLSTMConfig instance

	Example:
	    >>> config = {
	    ...     'vocab_size': 259,
	    ...     'hidden_size': 256,
	    ...     'num_layers': 6,
	    ...     'num_heads': 8,
	    ...     'max_seq_len': 2048,
	    ... }
	    >>> model = xLSTMCausalLM.from_config(config)
	"""

	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], 'DictConfig']) -> 'xLSTMCausalLM':
		"""
		Create xLSTMCausalLM from Hydra configuration.

		Converts Hydra-style config to xLSTMConfig and instantiates the model.
		Supports both plain dict and OmegaConf DictConfig.

		Args:
		    config: Configuration dictionary or DictConfig with keys:
		        - vocab_size: Vocabulary size (default: 259)
		        - hidden_size: Hidden dimension (default: 256)
		        - num_layers: Number of xLSTM blocks (default: 6)
		        - num_heads: Number of heads (default: 8)
		        - max_seq_len: Maximum sequence length (default: 2048)
		        - chunk_size: Chunk size for parallelization (default: 64)
		        - qk_dim_factor: Q/K dimension scaling factor (default: 0.5)
		        - v_dim_factor: Value dimension scaling factor (default: 1.0)
		        - norm_eps: Normalization epsilon (default: 1e-6)
		        - mode: 'train' or 'inference' (default: 'inference')
		        - use_cache: Use cache for recurrent state (default: True)

		Returns:
		    Instantiated xLSTMCausalLM model

		Example:
		    >>> from omegaconf import OmegaConf
		    >>> cfg = OmegaConf.create({
		    ...     'vocab_size': 259,
		    ...     'hidden_size': 256,
		    ...     'num_layers': 6,
		    ...     'num_heads': 8,
		    ... })
		    >>> model = xLSTMCausalLM.from_config(cfg)
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
		chunk_size = config.get('chunk_size', 64)
		qk_dim_factor = config.get('qk_dim_factor', 0.5)
		v_dim_factor = config.get('v_dim_factor', 1.0)
		norm_eps = config.get('norm_eps', 1e-6)
		mode = config.get('mode', 'inference')
		use_cache = config.get('use_cache', True)

		# Create xLSTMConfig
		model_config = xLSTMConfig(
			vocab_size=vocab_size,
			hidden_size=hidden_size,
			num_hidden_layers=num_layers,
			num_heads=num_heads,
			norm_eps=norm_eps,
			qk_dim_factor=qk_dim_factor,
			v_dim_factor=v_dim_factor,
			chunk_size=chunk_size,
			chunkwise_kernel="chunkwise--native_autograd",
			mode=mode,
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
			'model_type': 'xlstm',
			'vocab_size': self.config.vocab_size,
			'hidden_size': self.config.hidden_size,
			'num_layers': self.config.num_hidden_layers,
			'num_heads': self.config.num_heads,
			'chunk_size': self.config.chunk_size,
			'qk_dim_factor': self.config.qk_dim_factor,
			'v_dim_factor': self.config.v_dim_factor,
			'mode': self.config.mode,
			'architecture': 'Extended LSTM (Matrix-valued)',
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
		# xLSTM has recurrent state + multi-head operations
		hidden_size = self.config.hidden_size
		num_layers = self.config.num_hidden_layers
		num_heads = self.config.num_heads

		# LSTM cell states (matrix-valued): more complex than standard LSTM
		# Q, K, V projections per head
		qk_dim = int(hidden_size * self.config.qk_dim_factor)
		v_dim = int(hidden_size * self.config.v_dim_factor)
		projection_memory = (2 * qk_dim + v_dim) * num_heads * batch_size * seq_len

		# Cell states and gates
		cell_memory = hidden_size * num_heads * batch_size * seq_len

		# Per layer activation
		layer_activation = (projection_memory + cell_memory) * 4 / (1024**2)
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
			f"xLSTMCausalLM(\n"
			f"  vocab_size={info['vocab_size']},\n"
			f"  hidden_size={info['hidden_size']},\n"
			f"  num_layers={info['num_layers']},\n"
			f"  num_heads={info['num_heads']},\n"
			f"  chunk_size={info['chunk_size']},\n"
			f"  parameters={info['total_parameters']:,}\n"
			f")"
		)
