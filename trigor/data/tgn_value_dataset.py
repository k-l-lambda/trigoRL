"""
PyTorch dataset for TGN (Trigo Game Notation) files with value score extraction.

Extends TGNDataset to parse game structure and extract final scores
for dual-head network training (policy + value heads).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from trigor.data.registry import register_dataset
from trigor.data.tgn_dataset import TGNDataset
from trigor.data.tokenizer import TGNByteTokenizer


def parse_tgn_file(text: str, tokenizer: TGNByteTokenizer) -> Tuple[List[str], float, List[int]]:
	"""
	Parse TGN file to extract moves, score, and token positions.

	TGN Format:
		[Board 2x3x3]

		1. z00 zaa
		2. aaz aaa
		3. zza az0
		; -18

	Args:
	    text: Raw TGN file content
	    tokenizer: Tokenizer to compute move end positions

	Returns:
	    Tuple of (moves, score, move_end_positions):
	        - moves: List of move strings (e.g., ["z00", "zaa", "aaz", "aaa"])
	        - score: Final game score (negative = Black win, positive = White win)
	        - move_end_positions: List of token positions where each move ends
	                              Positions are relative to the tokenized full text

	Score Interpretation:
	    - Negative score: Black wins (e.g., -18 means Black won by 18 points)
	    - Positive score: White wins (e.g., +12 means White won by 12 points)
	    - Zero: Draw or missing score

	Move Token Positions:
	    The positions array indicates where each move ends in the tokenized full text.
	    Uses regex pattern \b[0a-zPR]+\b to match individual moves robustly.

	Examples:
	    >>> tokenizer = TGNByteTokenizer()
	    >>> text = "[Board 2x2x2]\\n1. z00 zaa\\n2. aaa\\n; -5"
	    >>> moves, score, positions = parse_tgn_file(text, tokenizer)
	    >>> moves
	    ['z00', 'zaa', 'aaa']
	    >>> score
	    -5.0
	    >>> len(positions)
	    3
	"""
	score = 0.0

	# Score comment: "; -18" or "; +12" or "; 0"
	score_pattern = re.compile(r';\s*([+-]?\d+(?:\.\d+)?)')
	score_match = score_pattern.search(text)
	if score_match:
		score = float(score_match.group(1))

	# Match all moves using robust regex pattern
	# Pattern \b[0a-zPR]+\b matches move coordinates
	# Only consider moves that appear after move numbers (e.g., "1. ")
	round_pattern = re.compile(r'\d+\.\s+([0a-zPR\s]+)')

	moves = []
	move_positions_in_text = []  # Store (start, end) character positions

	# Find all move sequences (e.g., "1. z00 zaa" captures "z00 zaa")
	for match in round_pattern.finditer(text):
		move_sequence = match.group(1).strip()
		# Split the sequence into individual moves
		individual_moves = re.findall(r'\b[0a-zPR]+\b', move_sequence)

		# Track character position of each individual move
		start_offset = match.start(1)
		for move in individual_moves:
			# Find this move's position within the captured group
			move_start = text.find(move, start_offset)
			if move_start != -1:
				move_end = move_start + len(move)
				moves.append(move)
				move_positions_in_text.append(move_end)  # Character position where move ends
				start_offset = move_end

	# Convert character positions to token positions
	move_end_positions = []
	for char_pos in move_positions_in_text:
		# Tokenize text up to this character position
		text_up_to_move = text[:char_pos]
		tokens_up_to_move = tokenizer.encode(
			text_up_to_move,
			add_special_tokens=True,  # Includes [START] token
			padding=False,
			truncation=False,
		)
		# Token position is the last token index (0-indexed)
		move_end_positions.append(len(tokens_up_to_move) - 1)

	return moves, score, move_end_positions


@register_dataset('TGNValueDataset')
class TGNValueDataset(TGNDataset):
	"""
	TGN dataset with value score extraction for dual-head training.

	Extends TGNDataset to add:
	- TGN structure parsing (moves, score)
	- value_score field in output (final game score)
	- move_end_positions field (token positions where moves end)

	The base class handles:
	- File discovery and filtering
	- Train/val splitting
	- Tokenization
	- Standard output (input_ids, labels, attention_mask)

	Args:
	    data_dir: Directory containing .tgn files
	    tokenizer: TGNByteTokenizer instance
	    max_length: Maximum sequence length (default: 8192)
	    min_length: Minimum file size in bytes to include (default: 10)
	    max_file_size: Maximum file size in bytes to include (default: 10000)
	    split: Optional split specification (default: None = use all files)
	    filter_fn: Optional function to filter files
	    parse_value: Enable value parsing (default: True)

	Output Format:
	    {
	        'input_ids': torch.Tensor,           # [max_length-1]
	        'labels': torch.Tensor,               # [max_length-1]
	        'attention_mask': torch.Tensor,       # [max_length-1]
	        'value_score': torch.Tensor,          # scalar float32
	        'move_end_positions': torch.Tensor,   # [variable] int64
	    }

	Usage:
	    >>> from trigor.data.tokenizer import TGNByteTokenizer
	    >>> tokenizer = TGNByteTokenizer()
	    >>> dataset = TGNValueDataset(
	    ...     data_dir='data/selfplay',
	    ...     tokenizer=tokenizer,
	    ...     max_length=2048,
	    ...     split='*0..7/10'  # 80% for training
	    ... )
	    >>> sample = dataset[0]
	    >>> print(sample['value_score'])         # Final game score
	    >>> print(len(sample['move_end_positions']))  # Number of moves
	    >>> print(sample['move_end_positions'])  # Token positions where moves end
	"""

	@classmethod
	def from_config(cls, config: Dict) -> 'TGNValueDataset':
		"""
		Create TGNValueDataset from configuration dictionary.

		Extends parent's from_config to support parse_value parameter.

		Args:
		    config: Configuration dictionary with keys from TGNDataset.from_config
		            plus optional:
		        - parse_value: Enable value parsing (default: True)

		Returns:
		    Instantiated TGNValueDataset

		Example:
		    >>> config = {
		    ...     'data_dir': 'data/selfplay',
		    ...     'max_length': 2048,
		    ...     'split': '*0..7/10',
		    ...     'parse_value': True,
		    ... }
		    >>> dataset = TGNValueDataset.from_config(config)
		"""
		# Call parent's from_config to get base parameters
		# But we need to instantiate using cls (TGNValueDataset) not parent class
		from omegaconf import OmegaConf

		# Convert plain dict to DictConfig for unified API
		if isinstance(config, dict):
			config = OmegaConf.create(config)

		# Create tokenizer
		tokenizer_config = config.get('tokenizer_config', {})
		tokenizer = TGNByteTokenizer(**tokenizer_config)

		# Extract dataset parameters
		dataset_params = {
			'data_dir': config.data_dir,
			'tokenizer': tokenizer,
			'max_length': config.get('max_length', 2048),
			'min_length': config.get('min_length', 10),
			'max_file_size': config.get('max_file_size', 10000),
			'split': config.get('split', None),
			'parse_value': config.get('parse_value', True),  # TGNValueDataset-specific
		}

		# Optional filter function
		if 'filter_fn' in config:
			dataset_params['filter_fn'] = config.filter_fn

		return cls(**dataset_params)

	def __init__(
		self,
		data_dir: str,
		tokenizer,
		max_length: int = 8192,
		min_length: int = 10,
		max_file_size: int = 10000,
		split: Optional[str] = None,
		filter_fn = None,
		parse_value: bool = True,
	):
		"""Initialize TGNValueDataset."""
		super().__init__(
			data_dir=data_dir,
			tokenizer=tokenizer,
			max_length=max_length,
			min_length=min_length,
			max_file_size=max_file_size,
			split=split,
			filter_fn=filter_fn,
		)
		self.parse_value = parse_value

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		"""
		Get a single training example with value score and move positions.

		Args:
		    idx: Index of the game to load

		Returns:
		    Dictionary containing:
		        - input_ids: Token sequence [max_length-1]
		        - labels: Target tokens [max_length-1]
		        - attention_mask: 1D mask [max_length-1]
		        - value_score: Scalar game result (float32)
		        - move_end_positions: Token positions [variable] (int64)
		"""
		# Get standard dataset output from parent class
		base_output = super().__getitem__(idx)

		# Parse TGN for value score if enabled
		if self.parse_value:
			file_path = self.files[idx]
			try:
				text = file_path.read_text(encoding='utf-8', errors='replace')
			except Exception as e:
				print(f"Warning: Failed to read {file_path.name}: {e}")
				# Fallback values
				base_output['value_score'] = torch.tensor(0.0, dtype=torch.float32)
				base_output['move_end_positions'] = torch.tensor([], dtype=torch.long)
				return base_output

			try:
				moves, score, move_end_positions = parse_tgn_file(text, self.tokenizer)
				base_output['value_score'] = torch.tensor(score, dtype=torch.float32)
				base_output['move_end_positions'] = torch.tensor(move_end_positions, dtype=torch.long)
			except Exception as e:
				# Fallback for malformed files
				print(f"Warning: Failed to parse {file_path.name}: {e}")
				base_output['value_score'] = torch.tensor(0.0, dtype=torch.float32)
				base_output['move_end_positions'] = torch.tensor([], dtype=torch.long)

		return base_output

	@staticmethod
	def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
		"""
		Collate function for DataLoader.

		Extends parent collate to handle value_score and move_end_positions fields.

		Args:
		    batch: List of dataset outputs (each is a dict)

		Returns:
		    Batched dictionary with stacked tensors:
		        - input_ids, labels, attention_mask: [batch_size, seq_len]
		        - value_score: [batch_size]
		        - move_end_positions: List of tensors (variable length per sample)
		"""
		# Use parent collate for standard fields
		collated = TGNDataset.collate_batch(batch)

		# Stack scalar fields
		if 'value_score' in batch[0]:
			collated['value_score'] = torch.stack([x['value_score'] for x in batch])

		# Keep move_end_positions as list (variable length per sample)
		if 'move_end_positions' in batch[0]:
			collated['move_end_positions'] = [x['move_end_positions'] for x in batch]

		return collated

	def __repr__(self) -> str:
		"""String representation of TGNValueDataset."""
		return (
			f"TGNValueDataset(num_files={len(self.files)}, "
			f"max_length={self.max_length}, "
			f"data_dir='{self.data_dir}', "
			f"parse_value={self.parse_value})"
		)

