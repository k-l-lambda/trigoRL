"""Dataset registry for TrigoRL."""

from typing import Any, Dict, Type, Union

from torch.utils.data import Dataset

try:
	from omegaconf import DictConfig
except ImportError:
	DictConfig = None

# Dataset registry mapping dataset type names to classes
DATASETS: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str, dataset_class: Type[Dataset] = None):
	"""
	Register a new dataset type. Can be used as a decorator or function.

	Usage as decorator:
	    @register_dataset('MyDataset')
	    class MyDataset(Dataset):
	        ...

	Usage as function:
	    register_dataset('MyDataset', MyDataset)

	Args:
	    name: Dataset type name (e.g., 'TGNDataset', 'CustomDataset')
	    dataset_class: Dataset class (must inherit from torch.utils.data.Dataset)

	Returns:
	    When used as decorator, returns the class unchanged
	"""

	def _register(cls: Type[Dataset]):
		"""Inner function to perform the actual registration."""
		if not issubclass(cls, Dataset):
			raise ValueError(f"Dataset class {cls} must inherit from torch.utils.data.Dataset")

		DATASETS[name] = cls
		print(f"Registered dataset: {name}")
		return cls

	# If called with class (function call style), register immediately
	if dataset_class is not None:
		return _register(dataset_class)

	# If called without class (decorator style), return the decorator
	return _register


def make_dataset(dataset_type: str, config: Union[Dict[str, Any], 'DictConfig']) -> Dataset:
	"""
	Factory function to create a dataset from configuration.

	If the dataset class has a `from_config` classmethod, it will be used.
	Otherwise, the config dictionary is passed directly as kwargs to __init__.

	Args:
	    dataset_type: Dataset type name (must be registered)
	    config: Dataset-specific configuration dictionary or DictConfig

	Returns:
	    Instantiated dataset

	Raises:
	    ValueError: If dataset_type is not registered

	Example:
	    >>> config = {
	    ...     'data_dir': 'data/tgn_games',
	    ...     'max_length': 512,
	    ... }
	    >>> dataset = make_dataset('TGNDataset', config)
	"""
	if dataset_type not in DATASETS:
		available = ", ".join(DATASETS.keys())
		raise ValueError(f"Unknown dataset type '{dataset_type}'. Available: {available}")

	dataset_class = DATASETS[dataset_type]

	# Check if dataset class has a from_config classmethod
	if hasattr(dataset_class, 'from_config') and callable(getattr(dataset_class, 'from_config')):
		# from_config handles both dict and DictConfig
		return dataset_class.from_config(config)

	# Generic fallback: pass config as kwargs to __init__
	# This requires a plain dict, so convert DictConfig if needed
	if DictConfig is not None and not isinstance(config, dict):
		from omegaconf import OmegaConf

		config = OmegaConf.to_container(config, resolve=True)
	return dataset_class(**config)


def list_datasets() -> list:
	"""Return list of all registered dataset types."""
	return list(DATASETS.keys())
