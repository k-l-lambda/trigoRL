"""
PyTorch dataset for TGN (Trigo Game Notation) files.

Loads TGN files from a directory and provides byte-tokenized sequences
for transformer-based sequence modeling.
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

try:
	from omegaconf import DictConfig
except ImportError:
	DictConfig = None

from trigor.data.registry import register_dataset
from trigor.data.tokenizer import TGNByteTokenizer
from trigor.data.utils import parse_split


@register_dataset('TGNDataset')
class TGNDataset(Dataset):
	"""
	PyTorch dataset for TGN files with byte-level tokenization.

	Loads TGN game notation files and converts them to byte token sequences
	suitable for next-token prediction or sequence modeling tasks.

	Args:
	    data_dir: Directory containing .tgn files
	    tokenizer: TGNByteTokenizer instance
	    max_length: Maximum sequence length (default: 2048)
	    min_length: Minimum file size in bytes to include (default: 10)
	    max_file_size: Maximum file size in bytes to include (default: 10000)
	    split: Optional split specification (default: None = use all files)
	           Format: "phases/cycle" or "*phases/cycle" (* for shuffle)
	           Phases can be:
	             - Comma-separated: "0,1,2,3/10"
	             - Range with ..: "0..3/10" (equivalent to "0,1,2,3/10")
	             - Mixed: "0..3,7,8/10"
	           Example: "*0..7/10" = use 80% for training (phases 0-7 out of 10)
	                    "8,9/10" = use 20% for validation (phases 8-9 out of 10)
	                    "*0..2/5" = use 60% for training with shuffle
	    filter_fn: Optional function to filter files (receives Path, returns bool)

	Split Format:
	    The split parameter uses a "phases/cycle" format to deterministically split data:
	    - Files are hashed and assigned to one of 'cycle' groups (0 to cycle-1)
	    - Only files in the specified 'phases' are included
	    - Prefix with '*' to enable shuffling
	    - Phases can use range syntax with '..' for convenience

	    Examples:
	        "*0..8/10" - Training set (90% of data, shuffled)
	        "9/10" - Validation set (10% of data, no shuffle)
	        "*0..3/5" - Training set (80% of data, shuffled)
	        "4/5" - Validation set (20% of data, no shuffle)
	"""

	@staticmethod
	def file_to_phase(file_path: Path, cycle: int) -> int:
		"""
		Deterministically assign a file to a phase based on its path hash.

		Args:
		    file_path: Path to the file
		    cycle: Total number of phases

		Returns:
		    Phase index (0 to cycle-1)
		"""
		# Use MD5 hash of filename for deterministic assignment
		file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
		hash_int = int(file_hash, 16)
		return hash_int % cycle

	@classmethod
	def from_config(cls, config: Union[Dict[str, Any], 'DictConfig']) -> 'TGNDataset':
		"""
		Create TGNDataset from configuration dictionary or DictConfig.

		This method handles tokenizer creation and parameter extraction,
		allowing the dataset to be created directly from config without
		special handling in the factory function. Supports both plain
		dictionaries and OmegaConf DictConfig objects.

		Args:
		    config: Configuration dictionary or DictConfig with keys:
		        - data_dir: Path to directory containing .tgn files
		        - tokenizer_config: Optional dict with tokenizer settings (default: {})
		        - max_length: Maximum sequence length (default: 2048)
		        - min_length: Minimum file size in bytes (default: 10)
		        - max_file_size: Maximum file size in bytes (default: 10000)
		        - split: Optional split specification (default: None)
		        - filter_fn: Optional callable to filter files (default: None)

		Returns:
		    Instantiated TGNDataset

		Example:
		    >>> config = {
		    ...     'data_dir': 'data/tgn_games',
		    ...     'max_length': 512,
		    ...     'split': '*0,1,2,3/5',  # 80% for training
		    ... }
		    >>> dataset = TGNDataset.from_config(config)
		"""
		# Convert plain dict to DictConfig for unified API
		if isinstance(config, dict):
			from omegaconf import OmegaConf

			config = OmegaConf.create(config)

		# Now use DictConfig API uniformly
		# Create tokenizer (TGNByteTokenizer is stateless, so config not needed)
		tokenizer_config = config.get('tokenizer_config', {})
		tokenizer = TGNByteTokenizer(**tokenizer_config)

		# Extract dataset parameters using DictConfig API
		dataset_params = {
			'data_dir': config.data_dir,
			'tokenizer': tokenizer,
			'max_length': config.get('max_length', 2048),
			'min_length': config.get('min_length', 10),
			'max_file_size': config.get('max_file_size', 10000),
			'split': config.get('split', None),
		}

		# Optional filter function
		if 'filter_fn' in config:
			dataset_params['filter_fn'] = config.filter_fn

		return cls(**dataset_params)

	def __init__(
		self,
		data_dir: str,
		tokenizer: TGNByteTokenizer,
		max_length: int = 8192,
		min_length: int = 10,
		max_file_size: int = 10000,
		split: Optional[str] = None,
		filter_fn: Optional[Callable[[Path], bool]] = None,
	):
		"""Initialize TGN dataset."""
		self.data_dir = Path(data_dir)
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.min_length = min_length
		self.max_file_size = max_file_size
		self.split = split

		# Parse split if provided
		if split is not None:
			self.phases, self.cycle, self.shuffle = parse_split(split)
		else:
			self.phases = None
			self.cycle = None
			self.shuffle = False

		# Find all .tgn files
		all_files = sorted(self.data_dir.glob("*.tgn"))

		if not all_files:
			raise ValueError(f"No .tgn files found in {data_dir}")

		# Filter by file size
		all_files = [f for f in all_files if self.min_length <= f.stat().st_size <= self.max_file_size]

		# Apply custom filter if provided
		if filter_fn is not None:
			all_files = [f for f in all_files if filter_fn(f)]

		# Apply split filter if specified
		if self.split is not None:
			# Assign each file to a phase based on hash
			self.files = [
				f for f in all_files
				if self.file_to_phase(f, self.cycle) in self.phases
			]

			if not self.files:
				raise ValueError(
					f"No files in split {self.split} from {data_dir}. "
					f"Total files before split: {len(all_files)}"
				)

			print(
				f"Loaded {len(self.files)} / {len(all_files)} TGN files from {data_dir} "
				f"(split: {self.split})"
			)
		else:
			self.files = all_files
			print(f"Loaded {len(self.files)} TGN files from {data_dir}")

		# Shuffle indices if required (but keep files list stable for reproducibility)
		if self.shuffle:
			import random
			# Use a fixed seed based on split string for reproducible shuffling
			seed = hash(self.split) % (2**32)
			random.Random(seed).shuffle(self.files)

	def __len__(self) -> int:
		"""Return number of games in dataset."""
		return len(self.files)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		"""
		Get a single training example.

		Args:
		    idx: Index of the game to load

		Returns:
		    Dictionary containing:
		        - input_ids: Token sequence for input [max_length]
		        - labels: Token sequence for targets [max_length]
		        - attention_mask: Mask for valid tokens [max_length]
		"""
		# Load TGN file
		file_path = self.files[idx]
		try:
			text = file_path.read_text(encoding='utf-8')
		except UnicodeDecodeError:
			# Fallback for files with different encodings
			text = file_path.read_text(encoding='utf-8', errors='replace')

		# Tokenize
		tokens = self.tokenizer.encode(
			text,
			max_length=self.max_length,
			add_special_tokens=True,
			padding=True,
			truncation=True,
		)

		# Create input and target sequences for next-token prediction
		# Input:  [START, tok1, tok2, ..., tokN-1]
		# Target: [tok1, tok2, ..., tokN-1, END]
		input_ids = tokens[:-1]
		labels = tokens[1:]

		# Create attention mask (1 for real tokens, 0 for padding)
		attention_mask = (tokens != self.tokenizer.PAD_ID).long()
		attention_mask = attention_mask[:-1]  # Align with input_ids

		return {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
		}

	def get_file_path(self, idx: int) -> Path:
		"""Get file path for a specific index."""
		return self.files[idx]

	def get_text(self, idx: int) -> str:
		"""Get raw text for a specific index."""
		return self.files[idx].read_text(encoding='utf-8')

	def get_file_info(self, idx: int) -> Dict:
		"""
		Get metadata about a file.

		Returns:
		    Dictionary with file information
		"""
		file_path = self.files[idx]
		return {
			'path': str(file_path),
			'name': file_path.name,
			'size_bytes': file_path.stat().st_size,
		}

	def get_stats(self) -> Dict:
		"""
		Get statistics about the dataset.

		Returns:
		    Dictionary with dataset statistics
		"""
		file_sizes = [f.stat().st_size for f in self.files]

		return {
			'num_files': len(self.files),
			'total_bytes': sum(file_sizes),
			'avg_bytes': sum(file_sizes) / len(file_sizes) if file_sizes else 0,
			'min_bytes': min(file_sizes) if file_sizes else 0,
			'max_bytes': max(file_sizes) if file_sizes else 0,
			'max_length': self.max_length,
			'vocab_size': self.tokenizer.VOCAB_SIZE,
		}

	@staticmethod
	def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		"""
		Collate function for DataLoader.

		This static method can be used with PyTorch DataLoader:
		    DataLoader(dataset, collate_fn=TGNDataset.collate_batch)

		Args:
		    batch: List of dataset items

		Returns:
		    Batched tensors
		"""
		return {
			'input_ids': torch.stack([item['input_ids'] for item in batch]),
			'labels': torch.stack([item['labels'] for item in batch]),
			'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
		}

	def __repr__(self) -> str:
		return (
			f"TGNDataset(num_files={len(self.files)}, " f"max_length={self.max_length}, " f"data_dir='{self.data_dir}')"
		)
