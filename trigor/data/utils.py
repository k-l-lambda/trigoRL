"""
Utility functions for data processing.
"""

import re
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig


def parse_split(split: str) -> Tuple[List[int], int, bool]:
	"""
	Parse split string into phases, cycle, and shuffle flag.

	Supports both comma-separated phases and range syntax with '..'.

	Args:
	    split: Split specification string (e.g., "*0..2/5" or "3,4/5")

	Returns:
	    Tuple of (phases, cycle, shuffle)
	    - phases: List of phase indices to include
	    - cycle: Total number of phases
	    - shuffle: Whether to shuffle the dataset

	Examples:
	    >>> parse_split("*0..2/5")
	    ([0, 1, 2], 5, True)
	    >>> parse_split("3,4/5")
	    ([3, 4], 5, False)
	    >>> parse_split("*0..3,7,8/10")
	    ([0, 1, 2, 3, 7, 8], 10, True)
	    >>> parse_split("0..2,5..7/10")
	    ([0, 1, 2, 5, 6, 7], 10, False)
	"""
	shuffle = split.startswith('*')
	if shuffle:
		split = split[1:]

	phases_str, cycle_str = split.split('/')

	# Parse phases: support comma-separated list with optional range syntax
	parts = phases_str.split(',')
	phases = []

	for part in parts:
		# Check if this part is a range (e.g., "0..7")
		range_match = re.match(r'^(\d+)\.\.(\d+)$', part)
		if range_match:
			# Expand range: "0..7" -> [0,1,2,3,4,5,6,7]
			start = int(range_match.group(1))
			end = int(range_match.group(2))
			phases.extend(range(start, end + 1))
		else:
			# Single phase number
			phases.append(int(part))

	cycle = int(cycle_str)

	return phases, cycle, shuffle


def make_dataloader(
	dataset_type: str,
	config: Union[Dict[str, Any], DictConfig],
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 0,
	pin_memory: bool = True,
	drop_last: bool = False,
	**dataloader_kwargs
) -> DataLoader:
	"""
	Create a dataset and DataLoader with the correct collate function.

	This is a convenience function that:
	1. Creates a dataset using make_dataset()
	2. Automatically selects the correct collate function for the dataset type
	3. Creates a DataLoader with the specified parameters

	Args:
	    dataset_type: Dataset type name (e.g., 'TGNDataset', 'TGNValueDataset')
	    config: Dataset configuration dictionary or DictConfig
	    batch_size: Batch size for DataLoader (default: 32)
	    shuffle: Whether to shuffle the dataset (default: True)
	    num_workers: Number of worker processes for data loading (default: 0)
	    pin_memory: Whether to pin memory for faster GPU transfer (default: True)
	    drop_last: Whether to drop the last incomplete batch (default: False)
	    **dataloader_kwargs: Additional keyword arguments for DataLoader

	Returns:
	    DataLoader instance with the dataset and correct collate function

	Example:
	    >>> config = {
	    ...     'data_dir': 'data/selfplay',
	    ...     'max_length': 1024,
	    ...     'split': '*0..7/10',
	    ... }
	    >>> train_loader = make_dataloader(
	    ...     'TGNValueDataset',
	    ...     config,
	    ...     batch_size=16,
	    ...     shuffle=True,
	    ... )
	    >>> for batch in train_loader:
	    ...     # batch has correct collate_batch applied
	    ...     pass
	"""
	from trigor.data.registry import make_dataset

	# Create dataset
	dataset = make_dataset(dataset_type, config)

	# Get the appropriate collate function
	# If the dataset class has a collate_batch method, use it
	collate_fn = None
	if hasattr(dataset, 'collate_batch') and callable(getattr(dataset, 'collate_batch')):
		collate_fn = dataset.collate_batch
	elif hasattr(dataset.__class__, 'collate_batch'):
		collate_fn = dataset.__class__.collate_batch

	# Create DataLoader
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		collate_fn=collate_fn,
		drop_last=drop_last,
		**dataloader_kwargs
	)

	return loader

