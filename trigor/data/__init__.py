"""
Data loading module for TGN (Trigo Game Notation) files.

This module provides byte-level tokenization and dataset loading
for transformer-based sequence modeling on Trigo game notation.
"""

from trigor.data.registry import DATASETS, list_datasets, make_dataset, register_dataset
from trigor.data.tgn_dataset import TGNDataset
from trigor.data.tokenizer import TGNByteTokenizer
from trigor.data.utils import parse_split

__all__ = [
	"TGNByteTokenizer",
	"TGNDataset",
	"DATASETS",
	"register_dataset",
	"make_dataset",
	"list_datasets",
	"parse_split",
]
