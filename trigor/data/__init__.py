"""
Data loading module for TGN (Trigo Game Notation) files.

This module provides byte-level tokenization and dataset loading
for transformer-based sequence modeling on Trigo game notation.
"""

from trigor.data.registry import DATASETS, list_datasets, make_dataset, register_dataset
from trigor.data.tgn_dataset import TGNDataset
from trigor.data.tgn_value_dataset import TGNValueDataset
from trigor.data.tokenizer import TGNByteTokenizer
from trigor.data.utils import make_dataloader, parse_split

__all__ = [
	"TGNByteTokenizer",
	"TGNDataset",
	"TGNValueDataset",
	"DATASETS",
	"register_dataset",
	"make_dataset",
	"make_dataloader",
	"list_datasets",
	"parse_split",
]
