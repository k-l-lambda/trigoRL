"""
Byte-level tokenizer for TGN (Trigo Game Notation) files.

This tokenizer converts TGN text to byte-level tokens, enabling
transformer models to learn directly from raw notation without
requiring structured parsing.
"""

from typing import List, Union

import torch


class TGNByteTokenizer:
	"""
	Byte-level tokenizer for TGN notation.

	Converts TGN text to byte tokens (0-255) with special tokens
	for padding, start, and end markers.

	Vocabulary:
	    0-255: Standard UTF-8 bytes
	    256: PAD token (for sequence padding)
	    257: START token (beginning of sequence)
	    258: END token (end of sequence)
	"""

	# Standard byte range
	BYTE_RANGE = 256

	# Special tokens (use byte values above 255)
	PAD_TOKEN_ID = 256
	START_TOKEN_ID = 257
	END_TOKEN_ID = 258

	# Vocabulary size (256 bytes + 3 special tokens)
	VOCAB_SIZE = 259

	def __init__(self):
		"""Initialize the byte tokenizer."""
		pass

	def encode(
		self,
		text: str,
		max_length: int = 2048,
		add_special_tokens: bool = True,
		padding: bool = True,
		truncation: bool = True,
	) -> torch.Tensor:
		"""
		Encode text to byte tokens.

		Args:
		    text: Input text to tokenize
		    max_length: Maximum sequence length
		    add_special_tokens: Add START and END tokens
		    padding: Pad sequences to max_length
		    truncation: Truncate sequences exceeding max_length

		Returns:
		    Tensor of token IDs [max_length] if padding=True, else [seq_len]
		"""
		# Convert text to UTF-8 bytes
		byte_tokens = list(text.encode('utf-8'))

		# Add special tokens
		if add_special_tokens:
			tokens = [self.START_TOKEN_ID] + byte_tokens + [self.END_TOKEN_ID]
		else:
			tokens = byte_tokens

		# Truncate if needed
		if truncation and len(tokens) > max_length:
			if add_special_tokens:
				# Keep START, truncate middle, add END
				tokens = tokens[: max_length - 1] + [self.END_TOKEN_ID]
			else:
				tokens = tokens[:max_length]

		# Pad if needed
		if padding and len(tokens) < max_length:
			tokens = tokens + [self.PAD_TOKEN_ID] * (max_length - len(tokens))

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
		Decode byte tokens back to text.

		Args:
		    tokens: Token IDs to decode
		    skip_special_tokens: Skip PAD, START, END tokens

		Returns:
		    Decoded text string
		"""
		# Convert tensor to list if needed
		if isinstance(tokens, torch.Tensor):
			token_list = tokens.tolist()
		else:
			token_list = tokens

		# Filter special tokens if requested
		if skip_special_tokens:
			byte_tokens = [t for t in token_list if t < self.BYTE_RANGE]  # Only keep standard bytes
		else:
			byte_tokens = [
				t for t in token_list if t < self.BYTE_RANGE or t in [self.START_TOKEN_ID, self.END_TOKEN_ID]
			]

		# Convert bytes to string
		try:
			text = bytes(byte_tokens).decode('utf-8', errors='replace')
		except Exception as e:
			# Fallback: ignore decode errors
			text = bytes(byte_tokens).decode('utf-8', errors='ignore')

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

	def __len__(self) -> int:
		"""Return vocabulary size."""
		return self.VOCAB_SIZE

	def __repr__(self) -> str:
		return f"TGNByteTokenizer(vocab_size={self.VOCAB_SIZE})"
