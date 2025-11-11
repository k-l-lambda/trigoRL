"""
PyTorch dataset for TGN (Trigo Game Notation) files.

Loads TGN files from a directory and provides byte-tokenized sequences
for transformer-based sequence modeling.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from trigorl.data.tokenizer import TGNByteTokenizer


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
        filter_fn: Optional function to filter files (receives Path, returns bool)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: TGNByteTokenizer,
        max_length: int = 2048,
        min_length: int = 10,
        max_file_size: int = 10000,
        filter_fn: Optional[Callable[[Path], bool]] = None,
    ):
        """Initialize TGN dataset."""
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.max_file_size = max_file_size

        # Find all .tgn files
        self.files = sorted(self.data_dir.glob("*.tgn"))

        if not self.files:
            raise ValueError(f"No .tgn files found in {data_dir}")

        # Filter by file size
        self.files = [
            f for f in self.files
            if self.min_length <= f.stat().st_size <= self.max_file_size
        ]

        # Apply custom filter if provided
        if filter_fn is not None:
            self.files = [f for f in self.files if filter_fn(f)]

        if not self.files:
            raise ValueError(f"No .tgn files remaining after filtering in {data_dir}")

        print(f"Loaded {len(self.files)} TGN files from {data_dir}")

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
        attention_mask = (tokens != self.tokenizer.PAD_TOKEN_ID).long()
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
            f"TGNDataset(num_files={len(self.files)}, "
            f"max_length={self.max_length}, "
            f"data_dir='{self.data_dir}')"
        )
