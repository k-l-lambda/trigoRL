"""Model registry for TrigoRL."""

from typing import Any, Dict, Type, Union

import torch.nn as nn

try:
	from omegaconf import DictConfig, OmegaConf
except ImportError:
	DictConfig = None
	OmegaConf = None


# Model registry mapping model type names to classes
MODELS: Dict[str, Type[nn.Module]] = {}


def register_model(name: str, model_class: Type[nn.Module] = None):
	"""
	Register a new model type. Can be used as a decorator or function.

	Usage as decorator:
	    @register_model('GPT2')
	    class GPT2CausalLM(GPT2LMHeadModel):
	        ...

	Usage as function:
	    register_model('GPT2', GPT2CausalLM)

	Args:
	    name: Model type name (e.g., 'gpt2', 'llama', 'rwkv', 'xlstm')
	    model_class: Model class (must inherit from torch.nn.Module)

	Returns:
	    When used as decorator, returns the class unchanged
	"""

	def _register(cls: Type[nn.Module]):
		"""Inner function to perform the actual registration."""
		if not issubclass(cls, nn.Module):
			raise ValueError(f"Model class {cls} must inherit from torch.nn.Module")

		MODELS[name] = cls
		print(f"Registered model: {name}")
		return cls

	# If called with class (function call style), register immediately
	if model_class is not None:
		return _register(model_class)

	# If called without class (decorator style), return the decorator
	return _register


def make_model(model_type: str, config: Union[Dict[str, Any], 'DictConfig']) -> nn.Module:
	"""
	Factory function to create a model from configuration.

	If the model class has a `from_config` classmethod, it will be used.
	Otherwise, the config dictionary is passed directly as kwargs to __init__.

	Args:
	    model_type: Model type name (must be registered)
	    config: Model-specific configuration dictionary or DictConfig

	Returns:
	    Instantiated model

	Raises:
	    ValueError: If model_type is not registered

	Example:
	    >>> config = {
	    ...     'type': 'gpt2',
	    ...     'vocab_size': 259,
	    ...     'hidden_size': 256,
	    ...     'num_layers': 6,
	    ...     'num_heads': 8,
	    ... }
	    >>> model = make_model('gpt2', config)
	"""
	if model_type not in MODELS:
		available = ", ".join(MODELS.keys())
		raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")

	model_class = MODELS[model_type]

	# Check if model class has a from_config classmethod
	if hasattr(model_class, 'from_config') and callable(getattr(model_class, 'from_config')):
		# from_config handles both dict and DictConfig
		return model_class.from_config(config)

	# Generic fallback: pass config as kwargs to __init__
	# This requires a plain dict, so convert DictConfig if needed
	if DictConfig is not None and not isinstance(config, dict):
		config = OmegaConf.to_container(config, resolve=True)
	return model_class(**config)


def list_models() -> list:
	"""Return list of all registered model types."""
	return list(MODELS.keys())
