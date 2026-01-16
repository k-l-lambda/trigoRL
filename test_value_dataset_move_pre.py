#!/usr/bin/env python3
"""Test script for TGNValueDataset with move_pre_indices field."""

from trigor.data.tgn_value_dataset import TGNValueDataset, parse_tgn_file
from trigor.data.tokenizer import TGNByteTokenizer


def test_parse_tgn_file():
	"""Test parse_tgn_file function with move_pre_indices."""
	print("=" * 60)
	print("Test 1: parse_tgn_file function")
	print("=" * 60)

	tokenizer = TGNByteTokenizer()
	text = "[Board 2x3x3]\n\n1. z00 zaa\n2. aaz aaa\n; -18"

	moves, score, move_end_positions, move_pre_indices = parse_tgn_file(text, tokenizer)

	print(f"Text: {repr(text)}")
	print(f"\nMoves: {moves}")
	print(f"Score: {score}")
	print(f"Move end positions (token): {move_end_positions}")
	print(f"Move pre indices (token): {move_pre_indices}")

	# Verify lengths match
	assert len(moves) == len(move_end_positions) == len(move_pre_indices), \
		"All arrays should have same length"
	print(f"\n✓ All arrays have length {len(moves)}")

	# Verify move_pre_indices < move_end_positions
	for i, (pre, end) in enumerate(zip(move_pre_indices, move_end_positions)):
		print(f"Move {i+1} '{moves[i]}': pre_idx={pre}, end_pos={end}")
		assert pre < end, f"Pre index ({pre}) should be less than end position ({end})"
	print("✓ All pre indices are before end positions")

	print()


def test_dataset_integration():
	"""Test TGNValueDataset with actual files."""
	print("=" * 60)
	print("Test 2: TGNValueDataset integration")
	print("=" * 60)

	try:
		import os
		from trigor.data.tokenizer import TGNByteTokenizer

		data_dir = 'data/selfplay_10k'

		if not os.path.exists(data_dir):
			print(f"Skipping: {data_dir} not found")
			return

		# Create dataset
		dataset = TGNValueDataset(
			data_dir=data_dir,
			tokenizer=TGNByteTokenizer(),
			max_length=2048,
		)

		print(f"Dataset loaded: {len(dataset)} files")

		# Get first sample
		sample = dataset[0]

		print(f"\nFirst sample:")
		print(f"  Keys: {list(sample.keys())}")
		print(f"  Moves found: {len(sample['move_end_positions'])}")
		print(f"  Value score: {sample['value_score'].item()}")

		# Verify fields exist
		assert 'move_end_positions' in sample, "move_end_positions should exist"
		assert 'move_pre_indices' in sample, "move_pre_indices should exist"
		assert 'value_score' in sample, "value_score should exist"
		print("✓ All expected fields exist")

		# Verify shapes match
		end_len = len(sample['move_end_positions'])
		pre_len = len(sample['move_pre_indices'])
		assert end_len == pre_len, f"Lengths should match: end={end_len}, pre={pre_len}"
		print(f"✓ Both position arrays have length {end_len}")

		# Verify move_pre_indices < move_end_positions
		for i in range(min(5, end_len)):
			pre = sample['move_pre_indices'][i].item()
			end = sample['move_end_positions'][i].item()
			print(f"  Move {i+1}: pre_idx={pre}, end_pos={end}")
			assert pre < end, f"Pre index should be less than end position"
		print("✓ Pre indices are before end positions")

	except Exception as e:
		print(f"Error: {e}")
		import traceback
		traceback.print_exc()


def test_with_manual_text():
	"""Test with manually constructed TGN text."""
	print("=" * 60)
	print("Test 3: Manual text verification")
	print("=" * 60)

	tokenizer = TGNByteTokenizer()
	text = "[Board 2x2x2]\n1. z00 zaa\n; -5"

	print(f"Text: {repr(text)}")

	# Parse
	moves, score, move_end_positions, move_pre_indices = parse_tgn_file(text, tokenizer)

	print(f"\nMoves: {moves}")
	print(f"Score: {score}")

	# Tokenize full text to understand positions
	full_tokens = tokenizer.encode(text, add_special_tokens=True, padding=False, truncation=False)
	print(f"\nFull text tokenized length: {len(full_tokens)}")

	# For each move, show the context
	for i, move in enumerate(moves):
		pre_idx = move_pre_indices[i]
		end_pos = move_end_positions[i]

		print(f"\nMove {i+1}: '{move}'")
		print(f"  Pre index: {pre_idx}")
		print(f"  End position: {end_pos}")

		# Decode tokens around this move
		if pre_idx > 0:
			pre_tokens = full_tokens[:pre_idx+1]
			pre_text = tokenizer.decode(pre_tokens)
			print(f"  Text up to pre_idx: {repr(pre_text)}")

		end_tokens = full_tokens[:end_pos+1]
		end_text = tokenizer.decode(end_tokens)
		print(f"  Text up to end_pos: {repr(end_text)}")

		# Verify move is in the decoded text
		assert move in end_text, f"Move '{move}' should be in decoded text"
		print(f"  ✓ Move found in decoded text")

	print()


if __name__ == '__main__':
	test_parse_tgn_file()
	test_with_manual_text()
	test_dataset_integration()

	print("=" * 60)
	print("All tests completed!")
	print("=" * 60)
