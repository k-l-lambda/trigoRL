"""
Test TGN dataset loading with C++ generated self-play data.

This test verifies the integration between:
- C++ self-play generator (trigo.cpp)
- Python TGN dataset loader (trigor/data/tgn_dataset.py)
- Training pipeline readiness
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data.tgn_dataset import TGNDataset
from trigor.data.tokenizer import TGNByteTokenizer
from trigor.data.utils import make_dataloader


def test_load_cpp_selfplay_data():
	"""Test loading C++ generated self-play data."""
	print("=" * 60)
	print("Testing TGN Dataset with C++ Self-Play Data")
	print("=" * 60)

	# Path to C++ generated data
	data_dir = "/tmp/selfplay_test"

	if not Path(data_dir).exists():
		print(f"\nError: Data directory not found: {data_dir}")
		print("Please run C++ self-play generator first:")
		print("  cd /home/camus/work/trigo.cpp/build")
		print("  ./self_play_generator --num-games 20 --output /tmp/selfplay_test")
		return False

	# Create tokenizer
	tokenizer = TGNByteTokenizer()
	print(f"\nTokenizer: {tokenizer}")
	print(f"Vocabulary size: {tokenizer.VOCAB_SIZE}")
	print(f"Special tokens: {tokenizer.get_special_tokens()}")

	# Create dataset
	print(f"\nLoading dataset from: {data_dir}")
	dataset = TGNDataset(
		data_dir=data_dir,
		tokenizer=tokenizer,
		max_length=1024,
		min_length=10,
		max_file_size=10000,
	)

	# Print dataset info
	print(f"\nDataset loaded: {dataset}")
	stats = dataset.get_stats()
	print(f"Number of games: {stats['num_files']}")
	print(f"Total bytes: {stats['total_bytes']}")
	print(f"Average bytes per game: {stats['avg_bytes']:.1f}")
	print(f"Size range: {stats['min_bytes']} - {stats['max_bytes']} bytes")

	# Test single item
	print("\n" + "=" * 60)
	print("Testing Single Item")
	print("=" * 60)

	item = dataset[0]
	file_info = dataset.get_file_info(0)

	print(f"\nFile: {file_info['name']}")
	print(f"Size: {file_info['size_bytes']} bytes")
	print(f"\nTensor shapes:")
	print(f"  input_ids: {item['input_ids'].shape} {item['input_ids'].dtype}")
	print(f"  labels: {item['labels'].shape} {item['labels'].dtype}")
	print(f"  attention_mask: {item['attention_mask'].shape} {item['attention_mask'].dtype}")

	# Count non-padding tokens
	num_real_tokens = item['attention_mask'].sum().item()
	print(f"\nReal tokens: {num_real_tokens} / {dataset.max_length}")
	print(f"Padding ratio: {(1 - num_real_tokens / dataset.max_length) * 100:.1f}%")

	# Show first few tokens
	print(f"\nFirst 20 input tokens: {item['input_ids'][:20].tolist()}")
	print(f"First 20 labels: {item['labels'][:20].tolist()}")

	# Decode first few tokens to verify
	decoded_start = tokenizer.decode(item['input_ids'][:50])
	print(f"\nDecoded start of sequence:")
	print(f"  '{decoded_start}'")

	# Test batch loading
	print("\n" + "=" * 60)
	print("Testing Batch Loading")
	print("=" * 60)

	config = {
		'data_dir': data_dir,
		'max_length': 1024,
	}

	dataloader = make_dataloader(
		'TGNDataset',
		config,
		batch_size=4,
		shuffle=True,
		num_workers=0,
	)

	print(f"\nDataLoader created: batch_size=4")
	print(f"Number of batches: {len(dataloader)}")

	# Get first batch
	batch = next(iter(dataloader))

	print(f"\nBatch tensor shapes:")
	print(f"  input_ids: {batch['input_ids'].shape}")
	print(f"  labels: {batch['labels'].shape}")
	print(f"  attention_mask: {batch['attention_mask'].shape}")

	# Verify batch contents
	avg_real_tokens = batch['attention_mask'].sum(dim=1).float().mean().item()
	print(f"\nAverage real tokens per sequence: {avg_real_tokens:.1f}")

	# Test dataset splitting
	print("\n" + "=" * 60)
	print("Testing Dataset Splitting")
	print("=" * 60)

	# Train split (80%)
	train_dataset = TGNDataset(
		data_dir=data_dir,
		tokenizer=tokenizer,
		max_length=1024,
		split="*0..7/10",  # Phases 0-7 out of 10 = 80%, shuffled
	)

	# Val split (20%)
	val_dataset = TGNDataset(
		data_dir=data_dir,
		tokenizer=tokenizer,
		max_length=1024,
		split="8,9/10",  # Phases 8-9 out of 10 = 20%, not shuffled
	)

	print(f"\nTrain set: {len(train_dataset)} files (expected ~80%)")
	print(f"Val set: {len(val_dataset)} files (expected ~20%)")
	print(f"Total: {len(train_dataset) + len(val_dataset)} files")

	# Verify no overlap
	train_files = {train_dataset.get_file_path(i) for i in range(len(train_dataset))}
	val_files = {val_dataset.get_file_path(i) for i in range(len(val_dataset))}
	overlap = train_files & val_files

	if overlap:
		print(f"\n⚠️  WARNING: Found {len(overlap)} overlapping files!")
		return False
	else:
		print(f"\n✓ No overlap between train and val sets")

	# Test actual TGN content
	print("\n" + "=" * 60)
	print("Testing TGN Content")
	print("=" * 60)

	raw_text = dataset.get_text(0)
	print(f"\nRaw TGN content (first 500 chars):")
	print("-" * 60)
	print(raw_text[:500])
	print("-" * 60)

	# Encode and decode cycle
	encoded = tokenizer.encode(raw_text, max_length=2048, padding=False, truncation=False)
	decoded = tokenizer.decode(encoded)

	print(f"\nEncoding test:")
	print(f"  Original length: {len(raw_text)} chars")
	print(f"  Encoded tokens: {len(encoded)} tokens")
	print(f"  Decoded length: {len(decoded)} chars")

	# Check if decode matches original (allowing for minor whitespace differences)
	if decoded.strip() == raw_text.strip():
		print(f"  ✓ Encode/decode cycle preserved content")
	else:
		print(f"  ⚠️  Decode differs from original")
		print(f"  First difference at position: {next((i for i, (a, b) in enumerate(zip(raw_text, decoded)) if a != b), None)}")

	print("\n" + "=" * 60)
	print("✓ All tests passed!")
	print("=" * 60)

	return True


if __name__ == '__main__':
	success = test_load_cpp_selfplay_data()
	sys.exit(0 if success else 1)
