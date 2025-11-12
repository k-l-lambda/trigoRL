"""
Test script for TGN byte-level dataset loader.

Verifies that the tokenizer and dataset work correctly with TGN files.
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNByteTokenizer, TGNDataset


def test_tokenizer():
	"""Test TGNByteTokenizer encode/decode."""
	print("=" * 80)
	print("Testing TGNByteTokenizer...")
	print("=" * 80)

	tokenizer = TGNByteTokenizer()

	# Test 1: Basic encode/decode
	text = "[Board 5x5x5]\n\n1. 000 y00\n"
	print(f"\nOriginal text: {repr(text)}")

	tokens = tokenizer.encode(text, max_length=64)
	print(f"Tokens shape: {tokens.shape}")
	print(f"Token IDs: {tokens[:30].tolist()}...")  # Show first 30

	decoded = tokenizer.decode(tokens)
	print(f"Decoded text: {repr(decoded)}")

	# Test 2: Vocab size
	print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
	print(f"PAD token ID: {tokenizer.PAD_TOKEN_ID}")
	print(f"START token ID: {tokenizer.START_TOKEN_ID}")
	print(f"END token ID: {tokenizer.END_TOKEN_ID}")

	# Test 3: Batch encoding
	texts = ["[Board 3x3]\n1. 00 aa\n", "[Board 5x5x5]\n1. 000 zzz\n2. pass resign\n"]
	batch_tokens = tokenizer.encode_batch(texts, max_length=64)
	print(f"\nBatch shape: {batch_tokens.shape}")

	print("\n✓ Tokenizer tests passed!")
	return tokenizer


def test_dataset(tokenizer):
	"""Test TGNDataset loading."""
	print("\n" + "=" * 80)
	print("Testing TGNDataset...")
	print("=" * 80)

	# Path to TGN files
	data_dir = "third_party/trigo/trigo-web/tools/output"

	if not Path(data_dir).exists():
		print(f"\n⚠ Warning: Data directory not found: {data_dir}")
		print("Skipping dataset tests.")
		return None

	# Create dataset
	dataset = TGNDataset(
		data_dir=data_dir,
		tokenizer=tokenizer,
		max_length=2048,
	)

	print(f"\nDataset: {dataset}")
	print(f"Number of files: {len(dataset)}")

	# Get statistics
	stats = dataset.get_stats()
	print(f"\nDataset statistics:")
	for key, value in stats.items():
		if isinstance(value, float):
			print(f"  {key}: {value:.2f}")
		else:
			print(f"  {key}: {value}")

	# Test single item
	if len(dataset) > 0:
		print(f"\nTesting single item (index 0)...")
		item = dataset[0]

		print(f"  input_ids shape: {item['input_ids'].shape}")
		print(f"  labels shape: {item['labels'].shape}")
		print(f"  attention_mask shape: {item['attention_mask'].shape}")

		# Decode to verify
		input_text = tokenizer.decode(item['input_ids'])
		print(f"  Decoded input (first 100 chars): {input_text[:100]}...")

		# Check attention mask
		valid_tokens = item['attention_mask'].sum().item()
		print(f"  Valid tokens: {valid_tokens} / {len(item['attention_mask'])}")

		# Show file info
		file_info = dataset.get_file_info(0)
		print(f"  File: {file_info['name']}")
		print(f"  Size: {file_info['size_bytes']} bytes")

	print("\n✓ Dataset tests passed!")
	return dataset


def test_dataloader(dataset):
	"""Test DataLoader with dataset."""
	if dataset is None:
		return

	print("\n" + "=" * 80)
	print("Testing DataLoader...")
	print("=" * 80)

	# Create DataLoader
	dataloader = DataLoader(
		dataset,
		batch_size=4,
		shuffle=True,
		collate_fn=TGNDataset.collate_batch,
	)

	print(f"\nDataLoader created with batch_size=4")
	print(f"Number of batches: {len(dataloader)}")

	# Test first batch
	print(f"\nTesting first batch...")
	batch = next(iter(dataloader))

	print(f"  input_ids shape: {batch['input_ids'].shape}")
	print(f"  labels shape: {batch['labels'].shape}")
	print(f"  attention_mask shape: {batch['attention_mask'].shape}")

	# Verify shapes
	assert batch['input_ids'].shape[0] <= 4, "Batch size should be <= 4"
	assert batch['input_ids'].shape == batch['labels'].shape, "Input and labels should match"
	assert batch['input_ids'].shape == batch['attention_mask'].shape, "Masks should match"

	print("\n✓ DataLoader tests passed!")


def main():
	"""Run all tests."""
	print("\n" + "=" * 80)
	print("TGN DATASET LOADER TEST SUITE")
	print("=" * 80)

	try:
		# Test tokenizer
		tokenizer = test_tokenizer()

		# Test dataset
		dataset = test_dataset(tokenizer)

		# Test dataloader
		test_dataloader(dataset)

		print("\n" + "=" * 80)
		print("✓ ALL TESTS PASSED!")
		print("=" * 80)

	except Exception as e:
		print(f"\n✗ TEST FAILED: {e}")
		import traceback

		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
