"""
Optimized tokenizer for TGN (Trigo Game Notation) files.

This tokenizer uses a reduced vocabulary of 128 tokens:
- 0-3: Special tokens (PAD, START, END, VALUE) - 4-7 reserved for future use
- 8-127: ASCII printable characters + essential whitespace

The reduced vocabulary is designed for:
1. Memory efficiency: Smaller embedding layer (128 vs 259)
2. Training speed: Fewer parameters to learn
3. TGN compatibility: ASCII-based notation fits in token range 10-104
4. Value head support: VALUE token for position evaluation
"""

from typing import List, Union

import torch


class TGNTokenizer:
	"""
	Compact tokenizer for TGN notation with 128-token vocabulary.

	Vocabulary Layout (128 tokens total):
	    0-7:    Special tokens
	            0: PAD    - Padding token
	            1: START  - Beginning of sequence
	            2: END    - End of sequence
	            3: VALUE  - Value evaluation marker (for dual-head network)
	            4-7: unused

	    8-10:   Essential whitespace
	            8  -> 9   (TAB)
	            9  -> 10  (LF/newline)
	            10 -> 32  (SPACE)

	    11-104: ASCII printable (33-126: ! to ~)
	            11 -> 33  (!)
	            12 -> 34  (")
	            ...
	            104 -> 126 (~)

	    127:    DEL character (127)

	This design allows TGN notation (using A-Z, a-z, 0-9, space, punctuation)
	and multi-line game records to be fully represented while keeping vocabulary minimal.
	"""

	# Vocabulary size
	VOCAB_SIZE = 128

	# Special tokens (0-3 used, 4-7 reserved)
	PAD_ID = 0
	START_ID = 1
	END_ID = 2
	VALUE_ID = 3  # For value evaluation in dual-head networks

	# Whitespace token mapping (tokens 8-10)
	# TAB(9) -> 8, LF(10) -> 9, SPACE(32) -> 10
	# ASCII printable (33-126) -> tokens 11-104
	# DEL(127) -> 127

	def __init__(self):
		"""Initialize the compact tokenizer."""
		# Create byte-to-token mapping
		self._build_vocab_map()

	def _build_vocab_map(self):
		"""Build bidirectional mapping between bytes and token IDs."""
		# Byte -> Token ID mapping
		self.byte_to_token = {}

		# Essential whitespace characters (use tokens 8-10)
		self.byte_to_token[9] = 8   # TAB -> token 8
		self.byte_to_token[10] = 9  # LF (newline) -> token 9
		self.byte_to_token[32] = 10 # SPACE -> token 10

		# ASCII printable range (33-126: ! to ~)
		for ascii_val in range(33, 127):  # 33-126 (! to ~)
			token_id = ascii_val - 33 + 11  # Start from token 11
			self.byte_to_token[ascii_val] = token_id

		# DEL character (127) -> token 127
		self.byte_to_token[127] = 127

		# Token ID -> Byte mapping (inverse)
		self.token_to_byte = {v: k for k, v in self.byte_to_token.items()}

	def encode(
		self,
		text: str,
		max_length: int = 2048,
		add_special_tokens: bool = True,
		add_value_token: bool = False,
		padding: bool = True,
		truncation: bool = True,
	) -> torch.Tensor:
		"""
		Encode text to token IDs.

		Args:
		    text: Input text to tokenize
		    max_length: Maximum sequence length
		    add_special_tokens: Add START and END tokens
		    add_value_token: Add VALUE token at beginning (for value head)
		    padding: Pad sequences to max_length
		    truncation: Truncate sequences exceeding max_length

		Returns:
		    Tensor of token IDs [max_length] if padding=True, else [seq_len]

		Example:
		    >>> tokenizer = TGNTokenizer()
		    >>> # Regular encoding
		    >>> tokens = tokenizer.encode("B3 000")
		    >>> # With value token (for dual-head training)
		    >>> tokens = tokenizer.encode("B3 000", add_value_token=True)
		    >>> # Result: [VALUE_ID, START_ID, ...tokens..., END_ID, PAD, PAD, ...]
		"""
		# Convert text to bytes
		byte_values = list(text.encode('utf-8'))

		# Map bytes to token IDs
		tokens = []
		for byte_val in byte_values:
			if byte_val in self.byte_to_token:
				tokens.append(self.byte_to_token[byte_val])
			# Skip out-of-vocabulary bytes (non-ASCII characters)
			# TGN notation should only use mapped ASCII characters

		# Add special tokens
		if add_value_token:
			# For dual-head networks: [VALUE] [START] ... [END]
			if add_special_tokens:
				tokens = [self.VALUE_ID, self.START_ID] + tokens + [self.END_ID]
			else:
				tokens = [self.VALUE_ID] + tokens
		elif add_special_tokens:
			# Standard: [START] ... [END]
			tokens = [self.START_ID] + tokens + [self.END_ID]

		# Truncate if needed
		if truncation and len(tokens) > max_length:
			if add_special_tokens:
				# Keep START (and VALUE if present), truncate middle, add END
				tokens = tokens[: max_length - 1] + [self.END_ID]
			else:
				tokens = tokens[:max_length]

		# Pad if needed
		if padding and len(tokens) < max_length:
			tokens = tokens + [self.PAD_ID] * (max_length - len(tokens))

		return torch.tensor(tokens, dtype=torch.long)


	def encode_batch(self, texts: List[str], max_length: int = 2048, **kwargs) -> torch.Tensor:
		"""
		Encode multiple texts to batch of token tensors.

		Args:
		    texts: List of input texts
		    max_length: Maximum sequence length
		    **kwargs: Additional arguments passed to encode()

		Returns:
		    Tensor of token IDs [batch_size, max_length]
		"""
		return torch.stack([self.encode(text, max_length=max_length, **kwargs) for text in texts])


	def decode(self, tokens: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
		"""
		Decode token IDs back to text.

		Args:
		    tokens: Token IDs to decode
		    skip_special_tokens: Skip special tokens (0-7)

		Returns:
		    Decoded text string
		"""
		# Convert tensor to list if needed
		if isinstance(tokens, torch.Tensor):
			token_list = tokens.tolist()
		else:
			token_list = tokens

		# Filter and convert tokens to bytes
		byte_values = []
		for token_id in token_list:
			# Skip special tokens
			if skip_special_tokens and token_id < 8:
				continue

			# Skip padding
			if token_id == self.PAD_ID:
				continue

			# Convert token to byte
			if token_id in self.token_to_byte:
				byte_values.append(self.token_to_byte[token_id])
			# else: skip unknown tokens

		# Convert bytes to string
		try:
			text = bytes(byte_values).decode('utf-8', errors='replace')
		except Exception:
			# Fallback: ignore decode errors
			text = bytes(byte_values).decode('utf-8', errors='ignore')

		return text


	def decode_batch(self, token_batch: torch.Tensor, **kwargs) -> List[str]:
		"""
		Decode batch of token tensors to texts.

		Args:
		    token_batch: Batch of token IDs [batch_size, seq_len]
		    **kwargs: Additional arguments passed to decode()

		Returns:
		    List of decoded text strings
		"""
		return [self.decode(tokens, **kwargs) for tokens in token_batch]


	def get_vocab_size(self) -> int:
		"""Return vocabulary size."""
		return self.VOCAB_SIZE


	def get_special_tokens(self) -> dict:
		"""
		Return dictionary of special token IDs.

		Returns:
		    Dictionary mapping token names to IDs
		"""
		return {
			'pad': self.PAD_ID,
			'start': self.START_ID,
			'end': self.END_ID,
			'value': self.VALUE_ID,
		}


	def __len__(self) -> int:
		"""Return vocabulary size."""
		return self.VOCAB_SIZE


	def __repr__(self) -> str:
		return (
			f"TGNTokenizer(vocab_size={self.VOCAB_SIZE}, "
			f"special_tokens={list(range(8))}, "
			f"ascii_range=32-127)"
		)


# Legacy alias for backward compatibility
TGNByteTokenizer = TGNTokenizer
