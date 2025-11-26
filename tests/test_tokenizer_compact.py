"""
Test suite for the optimized TGNTokenizer (128-token vocabulary).

Tests cover:
1. Basic encoding/decoding
2. Special tokens (including VALUE_ID)
3. Vocabulary size reduction (128 vs 259)
4. TGN notation compatibility
5. Dual-head network integration
"""

import pytest
import torch

from trigor.data.tokenizer import TGNTokenizer


class TestTGNTokenizerBasics:
	"""Test basic tokenizer functionality."""

	def setup_method(self):
		"""Initialize tokenizer for each test."""
		self.tokenizer = TGNTokenizer()

	def test_vocab_size(self):
		"""Test vocabulary size is 128."""
		assert self.tokenizer.get_vocab_size() == 128
		assert len(self.tokenizer) == 128

	def test_special_tokens(self):
		"""Test special token IDs (0-3 used, 4-7 reserved)."""
		special_tokens = self.tokenizer.get_special_tokens()

		assert special_tokens['pad'] == 0
		assert special_tokens['start'] == 1
		assert special_tokens['end'] == 2
		assert special_tokens['value'] == 3  # VALUE token for dual-head networks
		assert len(special_tokens) == 4  # Only 4 tokens in use

	def test_basic_encoding(self):
		"""Test basic text encoding."""
		text = "B3 000"
		tokens = self.tokenizer.encode(
			text,
			max_length=16,
			add_special_tokens=True,
			padding=True
		)

		# Check tensor properties
		assert isinstance(tokens, torch.Tensor)
		assert tokens.shape == (16,)
		assert tokens.dtype == torch.long

		# Check structure: [START] ... [END] [PAD] [PAD] ...
		assert tokens[0] == self.tokenizer.START_ID
		assert self.tokenizer.END_ID in tokens
		assert (tokens == self.tokenizer.PAD_ID).sum() > 0

	def test_basic_decoding(self):
		"""Test basic decoding back to text."""
		text = "B3 000"
		tokens = self.tokenizer.encode(text, max_length=32, padding=True)
		decoded = self.tokenizer.decode(tokens)

		assert decoded == text

	def test_roundtrip(self):
		"""Test encoding and decoding roundtrip."""
		test_cases = [
			"B3 000",
			"19x19x1",
			"W[B3] B[000] W[111]",
			"AB(x3)",
			"Black wins by 7.5 points",
		]

		for text in test_cases:
			tokens = self.tokenizer.encode(text, max_length=64, padding=True)
			decoded = self.tokenizer.decode(tokens)
			assert decoded == text, f"Roundtrip failed for: {text}"


class TestValueToken:
	"""Test VALUE token functionality for dual-head networks."""

	def setup_method(self):
		"""Initialize tokenizer for each test."""
		self.tokenizer = TGNTokenizer()

	def test_value_token_id(self):
		"""Test VALUE_ID is correctly assigned."""
		assert self.tokenizer.VALUE_ID == 3

	def test_encode_without_value_token(self):
		"""Test standard encoding without VALUE token."""
		text = "B3 000"
		tokens = self.tokenizer.encode(
			text,
			max_length=16,
			add_special_tokens=True,
			add_value_token=False,
			padding=False
		)

		# Structure: [START] ... [END]
		assert tokens[0] == self.tokenizer.START_ID
		assert tokens[-1] == self.tokenizer.END_ID
		assert self.tokenizer.VALUE_ID not in tokens

	def test_encode_with_value_token(self):
		"""Test encoding WITH VALUE token for dual-head training."""
		text = "B3 000"
		tokens = self.tokenizer.encode(
			text,
			max_length=16,
			add_special_tokens=True,
			add_value_token=True,  # ← Enable VALUE token
			padding=False
		)

		# Structure: [VALUE] [START] ... [END]
		assert tokens[0] == self.tokenizer.VALUE_ID
		assert tokens[1] == self.tokenizer.START_ID
		assert tokens[-1] == self.tokenizer.END_ID

	def test_trajectory_encoding(self):
		"""Test encoding trajectory with VALUE tokens after each move."""
		# Simulate a 3-move game
		moves = ["B3 000", "W3 111", "B3 222"]

		# Encode with VALUE token after each move
		trajectory_tokens = []
		for move in moves:
			# Encode move
			move_tokens = self.tokenizer.encode(
				move,
				add_special_tokens=False,
				add_value_token=False,
				padding=False
			)
			trajectory_tokens.extend(move_tokens.tolist())

			# Add VALUE token after move
			trajectory_tokens.append(self.tokenizer.VALUE_ID)

		# Convert to tensor
		trajectory_tensor = torch.tensor(trajectory_tokens, dtype=torch.long)

		# Verify VALUE tokens are at correct positions
		value_positions = (trajectory_tensor == self.tokenizer.VALUE_ID).nonzero(as_tuple=True)[0]
		assert len(value_positions) == 3  # 3 moves → 3 VALUE tokens

	def test_decode_skips_value_token(self):
		"""Test decoding correctly skips VALUE token."""
		text = "B3 000"
		tokens = self.tokenizer.encode(
			text,
			add_value_token=True,
			padding=False
		)

		decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
		assert decoded == text
		assert "[VALUE]" not in decoded


class TestASCIIMapping:
	"""Test ASCII character mapping (tokens 8-127)."""

	def setup_method(self):
		"""Initialize tokenizer for each test."""
		self.tokenizer = TGNTokenizer()

	def test_ascii_printable_range(self):
		"""Test ASCII printable characters (32-127) use direct identity mapping."""
		# Test a few representative ASCII characters with identity mapping
		test_cases = [
			(' ', 32),   # SPACE
			('0', 48),   # Digit
			('A', 65),   # Uppercase
			('a', 97),   # Lowercase
			('~', 126),  # Last printable before DEL
		]

		for char, expected_token in test_cases:
			tokens = self.tokenizer.encode(
				char,
				add_special_tokens=False,
				padding=False
			)
			# Direct identity mapping: token_id = ascii_value
			assert tokens[0] == expected_token == ord(char)

	def test_space_character(self):
		"""Test space character (ASCII 32) maps to token 32 (identity mapping)."""
		text = " "
		tokens = self.tokenizer.encode(
			text,
			add_special_tokens=False,
			padding=False
		)

		# Direct identity mapping: token 32 = ASCII 32
		assert tokens[0] == 32

	def test_tgn_common_characters(self):
		"""Test TGN common characters are all encodable."""
		# Common TGN characters
		tgn_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 []()+-x"

		tokens = self.tokenizer.encode(
			tgn_chars,
			add_special_tokens=False,
			padding=False
		)

		# All should encode successfully (no tokens skipped)
		assert len(tokens) == len(tgn_chars)

	def test_unknown_bytes(self):
		"""Test non-ASCII bytes are skipped (not encoded)."""
		# UTF-8 encoded Chinese character (multi-byte)
		text = "你好"  # Contains bytes > 127

		tokens = self.tokenizer.encode(
			text,
			add_special_tokens=False,
			padding=False
		)

		# Should be empty since unknown bytes are skipped
		assert len(tokens) == 0


class TestBatchOperations:
	"""Test batch encoding/decoding."""

	def setup_method(self):
		"""Initialize tokenizer for each test."""
		self.tokenizer = TGNTokenizer()

	def test_encode_batch(self):
		"""Test batch encoding."""
		texts = ["B3 000", "W3 111", "B3 222"]
		tokens_batch = self.tokenizer.encode_batch(texts, max_length=16)

		assert isinstance(tokens_batch, torch.Tensor)
		assert tokens_batch.shape == (3, 16)  # (batch_size, max_length)

	def test_decode_batch(self):
		"""Test batch decoding."""
		texts = ["B3 000", "W3 111", "B3 222"]
		tokens_batch = self.tokenizer.encode_batch(texts, max_length=16)
		decoded_texts = self.tokenizer.decode_batch(tokens_batch)

		assert len(decoded_texts) == 3
		for original, decoded in zip(texts, decoded_texts):
			assert original == decoded


class TestMemoryEfficiency:
	"""Test memory efficiency improvements vs old tokenizer."""

	def test_vocab_size_reduction(self):
		"""Test vocabulary size reduced from 259 to 128."""
		tokenizer = TGNTokenizer()

		# New tokenizer: 128 tokens
		assert tokenizer.get_vocab_size() == 128

		# Old tokenizer would be: 256 bytes + 3 special = 259
		# Reduction: 259 → 128 = 50.6% reduction

	def test_embedding_layer_size(self):
		"""Calculate embedding layer size reduction."""
		hidden_size = 256
		old_vocab_size = 259
		new_vocab_size = 128

		# Embedding layer: vocab_size × hidden_size × 4 bytes (float32)
		old_size_mb = (old_vocab_size * hidden_size * 4) / (1024 * 1024)
		new_size_mb = (new_vocab_size * hidden_size * 4) / (1024 * 1024)

		print(f"\nEmbedding layer size:")
		print(f"  Old (259 vocab): {old_size_mb:.2f} MB")
		print(f"  New (128 vocab): {new_size_mb:.2f} MB")
		print(f"  Reduction: {(old_size_mb - new_size_mb):.2f} MB ({(1 - new_size_mb/old_size_mb)*100:.1f}%)")

		assert new_size_mb < old_size_mb


class TestTGNNotationCompatibility:
	"""Test compatibility with actual TGN notation."""

	def setup_method(self):
		"""Initialize tokenizer for each test."""
		self.tokenizer = TGNTokenizer()

	def test_trigo_moves(self):
		"""Test encoding Trigo move notation."""
		moves = [
			"B3 000",  # Black at center
			"W3 aaa",  # White at corner
			"B3 zzz",  # Black at opposite corner
			"W3 0ab",  # White at mixed coordinates
		]

		for move in moves:
			tokens = self.tokenizer.encode(move, padding=False)
			decoded = self.tokenizer.decode(tokens)
			assert decoded == move

	def test_game_metadata(self):
		"""Test encoding game metadata."""
		metadata = [
			"19x19x1",
			"Size: 5x5x5",
			"Black: Player1",
			"White: Player2",
			"Komi: 7.5",
		]

		for text in metadata:
			tokens = self.tokenizer.encode(text, padding=False)
			decoded = self.tokenizer.decode(tokens)
			assert decoded == text

	def test_full_game_example(self):
		"""Test encoding a complete game record."""
		game_record = """Size: 5x5x5
Black: Player1
White: Player2
B3 000
W3 aaa
B3 111
W3 zzz
Black wins"""

		tokens = self.tokenizer.encode(game_record, max_length=512, padding=True)
		decoded = self.tokenizer.decode(tokens)

		# Normalize whitespace for comparison
		assert decoded.strip() == game_record.strip()


if __name__ == "__main__":
	pytest.main([__file__, "-v", "-s"])
