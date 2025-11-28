"""
Unit tests for TGNValueDataset.

Tests TGN parsing, dataset output format, and integration with DataLoader.
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from trigor.data.tgn_value_dataset import TGNValueDataset, parse_tgn_file
from trigor.data.tokenizer import TGNByteTokenizer


class TestTGNParsing:
	"""Test TGN file parsing logic."""

	def test_parse_tgn_basic(self):
		"""Test basic TGN parsing with negative score."""
		tokenizer = TGNByteTokenizer()
		text = """
[Board 2x3x3]

1. z00 zaa
2. aaz aaa
3. zza az0
; -18
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert len(moves) == 6  # 6 individual moves (2 per round for rounds 1-3)
		assert 'z00' in moves and 'zaa' in moves
		assert score == -18.0
		assert len(positions) == 6

	def test_parse_tgn_positive_score(self):
		"""Test positive score (White win)."""
		tokenizer = TGNByteTokenizer()
		text = """
1. abc def
2. ghi
; +12
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert len(moves) == 3  # 3 individual moves
		assert score == 12.0
		assert len(positions) == 3

	def test_parse_tgn_zero_score(self):
		"""Test zero score (draw)."""
		tokenizer = TGNByteTokenizer()
		text = """
1. abc def
; 0
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert score == 0.0
		assert len(moves) == 2  # 2 individual moves
		assert len(positions) == 2

	def test_parse_tgn_missing_score(self):
		"""Test fallback when score line is missing."""
		tokenizer = TGNByteTokenizer()
		text = """
1. abc def
2. xyz
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert score == 0.0  # Default to 0 when missing
		assert len(moves) == 3  # 3 individual moves
		assert len(positions) == 3

	def test_parse_tgn_single_move_per_line(self):
		"""Test lines with single move (last move of game)."""
		tokenizer = TGNByteTokenizer()
		text = """
1. z00 zaa
2. aaa
; -5
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert len(moves) == 3  # 3 individual moves
		assert 'z00' in moves and 'zaa' in moves and 'aaa' in moves
		assert score == -5.0
		assert len(positions) == 3

	def test_parse_tgn_float_score(self):
		"""Test fractional scores (if supported)."""
		tokenizer = TGNByteTokenizer()
		text = """
1. abc def
; -7.5
"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert score == -7.5
		assert len(moves) == 2  # 2 individual moves
		assert len(positions) == 2

	def test_parse_tgn_empty_file(self):
		"""Test empty TGN file."""
		tokenizer = TGNByteTokenizer()
		text = ""

		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert len(moves) == 0
		assert score == 0.0
		assert len(positions) == 0

	def test_parse_tgn_extra_whitespace(self):
		"""Test TGN with extra whitespace."""
		tokenizer = TGNByteTokenizer()
		text = """


1. z00 zaa

2. aaz aaa

; -10

"""
		moves, score, positions = parse_tgn_file(text, tokenizer)

		assert len(moves) == 4  # 4 individual moves
		assert score == -10.0
		assert len(positions) == 4


class TestTGNValueDataset:
	"""Test TGNValueDataset class."""

	def test_dataset_output_fields(self, tmp_path):
		"""Test dataset returns correct fields."""
		# Create mock TGN file
		tgn_file = tmp_path / "test.tgn"
		tgn_file.write_text("[Board 2x2x2]\n1. z00 zaa\n2. aaa\n; -5")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
		)

		assert len(dataset) == 1

		sample = dataset[0]

		# Check all required fields exist
		assert 'input_ids' in sample
		assert 'labels' in sample
		assert 'attention_mask' in sample
		assert 'value_score' in sample
		assert 'move_end_positions' in sample

	def test_dataset_value_extraction(self, tmp_path):
		"""Test correct value extraction."""
		tgn_file = tmp_path / "test.tgn"
		tgn_file.write_text("[Board 2x2x2]\n1. z00 zaa\n2. aaa\n; -5")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
		)

		sample = dataset[0]

		assert sample['value_score'].item() == -5.0
		assert len(sample['move_end_positions']) == 3  # 3 individual moves

	def test_dataset_tensor_types(self, tmp_path):
		"""Test output tensor types are correct."""
		tgn_file = tmp_path / "test.tgn"
		tgn_file.write_text("1. abc\n; 10")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
		)

		sample = dataset[0]

		assert sample['input_ids'].dtype == torch.long
		assert sample['labels'].dtype == torch.long
		assert sample['attention_mask'].dtype == torch.long
		assert sample['value_score'].dtype == torch.float32
		assert sample['move_end_positions'].dtype == torch.long

	def test_dataset_parse_value_disabled(self, tmp_path):
		"""Test dataset with parse_value=False."""
		tgn_file = tmp_path / "test.tgn"
		tgn_file.write_text("1. abc\n; 10")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
			parse_value=False,
		)

		sample = dataset[0]

		# Should not have value fields when disabled
		assert 'value_score' not in sample
		assert 'move_end_positions' not in sample

	def test_dataset_malformed_file_fallback(self, tmp_path):
		"""Test fallback for malformed TGN files."""
		tgn_file = tmp_path / "malformed.tgn"
		# File with no parseable content
		tgn_file.write_text("This is not a valid TGN file\nNo moves here\n")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
		)

		sample = dataset[0]

		# Should fallback to 0.0 score and empty positions
		assert sample['value_score'].item() == 0.0
		assert len(sample['move_end_positions']) == 0

	def test_dataset_multiple_files(self, tmp_path):
		"""Test dataset with multiple TGN files."""
		# Create multiple TGN files
		for i, score in enumerate([-10, 5, -3, 15]):
			tgn_file = tmp_path / f"game{i}.tgn"
			tgn_file.write_text(f"1. abc def\n2. xyz\n; {score}")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=128,
		)

		assert len(dataset) == 4

		# Check each file has correct score
		scores = [dataset[i]['value_score'].item() for i in range(4)]
		assert scores == [-10.0, 5.0, -3.0, 15.0]


class TestCollateFunction:
	"""Test collate_batch function."""

	def test_collate_batch_basic(self, tmp_path):
		"""Test collate function stacks tensors correctly."""
		# Create mock files
		for i in range(3):
			tgn_file = tmp_path / f"game{i}.tgn"
			tgn_file.write_text(f"1. abc\n; {i}")

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=64,
		)

		loader = DataLoader(
			dataset,
			batch_size=3,
			collate_fn=TGNValueDataset.collate_batch,
		)

		batch = next(iter(loader))

		# Check batch shapes
		# Note: input_ids and labels are max_length-1 due to next-token prediction shift
		assert batch['input_ids'].shape == (3, 63)
		assert batch['labels'].shape == (3, 63)
		assert batch['attention_mask'].shape == (3, 63)
		assert batch['value_score'].shape == (3,)

		# Check values
		assert batch['value_score'].tolist() == [0.0, 1.0, 2.0]


class TestIntegration:
	"""Integration tests with real-world scenarios."""

	def test_dataloader_integration(self, tmp_path):
		"""Test full DataLoader integration."""
		# Create multiple games
		games = [
			("[Board 3x3x3]\n1. z00 zaa\n2. aaz aaa\n; -18", -18.0, 4),  # 4 individual moves
			("1. abc def\n2. xyz\n3. qqq\n; 7", 7.0, 4),  # 4 individual moves
			("1. single\n; 0", 0.0, 1),  # 1 individual move
		]

		for i, (content, _, _) in enumerate(games):
			(tmp_path / f"game{i}.tgn").write_text(content)

		tokenizer = TGNByteTokenizer()
		dataset = TGNValueDataset(
			data_dir=str(tmp_path),
			tokenizer=tokenizer,
			max_length=256,
		)

		loader = DataLoader(
			dataset,
			batch_size=2,
			shuffle=False,
			collate_fn=TGNValueDataset.collate_batch,
		)

		batches = list(loader)

		# Should have 2 batches (2 + 1)
		assert len(batches) == 2

		# First batch (size 2)
		batch1 = batches[0]
		assert batch1['value_score'].tolist() == [-18.0, 7.0]
		assert len(batch1['move_end_positions'][0]) == 4
		assert len(batch1['move_end_positions'][1]) == 4

		# Second batch (size 1)
		batch2 = batches[1]
		assert batch2['value_score'].tolist() == [0.0]
		assert len(batch2['move_end_positions'][0]) == 1

	def test_from_config_classmethod(self, tmp_path):
		"""Test dataset creation from config dictionary."""
		# Create test file
		(tmp_path / "test.tgn").write_text("1. abc\n; -5")

		config = {
			'data_dir': str(tmp_path),
			'max_length': 128,
			'min_length': 1,
			'parse_value': True,
		}

		dataset = TGNValueDataset.from_config(config)

		assert len(dataset) == 1
		sample = dataset[0]
		assert sample['value_score'].item() == -5.0

	def test_make_dataset_factory(self, tmp_path):
		"""Test dataset creation via make_dataset factory function."""
		from trigor.data import make_dataset

		# Create test files with different scores
		(tmp_path / "game1.tgn").write_text("1. abc def\n; -10")
		(tmp_path / "game2.tgn").write_text("1. xyz\n; 5")

		config = {
			'data_dir': str(tmp_path),
			'max_length': 256,
			'min_length': 1,
			'parse_value': True,
		}

		# Create dataset via factory
		dataset = make_dataset('TGNValueDataset', config)

		# Verify correct type
		assert isinstance(dataset, TGNValueDataset)
		assert dataset.__class__.__name__ == 'TGNValueDataset'

		# Verify dataset works correctly
		assert len(dataset) == 2

		# Check first sample
		sample0 = dataset[0]
		assert 'value_score' in sample0
		assert 'move_end_positions' in sample0
		assert sample0['value_score'].item() == -10.0
		assert len(sample0['move_end_positions']) == 2  # 2 moves

		# Check second sample
		sample1 = dataset[1]
		assert sample1['value_score'].item() == 5.0
		assert len(sample1['move_end_positions']) == 1  # 1 move

		# Verify parse_value attribute
		assert hasattr(dataset, 'parse_value')
		assert dataset.parse_value is True

	def test_make_dataset_with_parse_value_disabled(self, tmp_path):
		"""Test make_dataset with parse_value=False."""
		from trigor.data import make_dataset

		# Create test file
		(tmp_path / "test.tgn").write_text("1. abc\n; 10")

		config = {
			'data_dir': str(tmp_path),
			'max_length': 128,
			'parse_value': False,  # Disable value parsing
		}

		dataset = make_dataset('TGNValueDataset', config)

		# Should be TGNValueDataset instance
		assert isinstance(dataset, TGNValueDataset)

		# But should not have value fields
		sample = dataset[0]
		assert 'value_score' not in sample
		assert 'move_end_positions' not in sample

		# Verify parse_value attribute is False
		assert dataset.parse_value is False

	def test_make_dataloader_utility(self, tmp_path):
		"""Test make_dataloader utility function."""
		from trigor.data import make_dataloader

		# Create test files
		for i in range(5):
			(tmp_path / f"game{i}.tgn").write_text(f"1. abc def\n; {i}")

		config = {
			'data_dir': str(tmp_path),
			'max_length': 128,
		}

		# Create dataloader using utility function
		loader = make_dataloader(
			dataset_type='TGNValueDataset',
			config=config,
			batch_size=2,
			shuffle=False,
			num_workers=0,
		)

		# Verify loader properties
		assert loader is not None
		assert len(loader.dataset) == 5
		assert loader.batch_size == 2
		assert len(loader) == 3  # 5 samples / 2 batch_size = 3 batches

		# Verify collate function is applied
		batch = next(iter(loader))
		assert 'input_ids' in batch
		assert 'value_score' in batch
		assert 'move_end_positions' in batch
		assert batch['value_score'].shape == (2,)  # Batch size
		assert isinstance(batch['move_end_positions'], list)
		assert len(batch['move_end_positions']) == 2

		# Verify dataset type is correct
		assert isinstance(loader.dataset, TGNValueDataset)

