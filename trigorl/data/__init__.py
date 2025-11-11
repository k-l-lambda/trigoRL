"""
Data loading module for TGN (Trigo Game Notation) files.

This module provides byte-level tokenization and dataset loading
for transformer-based sequence modeling on Trigo game notation.
"""

from trigorl.data.tgn_dataset import TGNDataset
from trigorl.data.tokenizer import TGNByteTokenizer

__all__ = [
	"TGNByteTokenizer",
	"TGNDataset",
]
