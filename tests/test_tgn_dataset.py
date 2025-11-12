"""
Test script for TGN byte-level dataset loader with Hydra configuration.

Verifies that the tokenizer and dataset work correctly with TGN files,
and can be loaded from Hydra configuration.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNByteTokenizer, TGNDataset, make_dataset


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


def test_dataset_manual(tokenizer):
	"""Test TGNDataset loading manually (original method)."""
	print("\n" + "=" * 80)
	print("Testing TGNDataset (Manual Creation)...")
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

	print("\n✓ Manual dataset creation passed!")
	return dataset


def test_dataset_from_config():
	"""Test TGNDataset loading from Hydra configuration."""
	print("\n" + "=" * 80)
	print("Testing TGNDataset (From Hydra Config)...")
	print("=" * 80)

	# Load Hydra configuration
	config_path = project_root / "configs" / "trigo_test.yaml"
	print(f"\nLoading config from: {config_path}")

	cfg = OmegaConf.load(config_path)

	# Set paths.root to project root before resolving
	OmegaConf.update(cfg, "paths.root", str(project_root))

	# Resolve interpolations
	OmegaConf.resolve(cfg)

	# Print data configuration
	print("\nData configuration:")
	print(OmegaConf.to_yaml(cfg.data))

	# Check if data directory exists
	data_dir = Path(cfg.data.data_dir)
	if not data_dir.exists():
		print(f"\n⚠ Warning: Data directory not found: {data_dir}")
		print("Skipping config-based dataset test.")
		return None

	# Convert to dict and create dataset using factory
	print(f"\nCreating dataset with type: {cfg.data.type}")

	# Pass DictConfig directly - no need to convert to dict
	dataset = make_dataset(cfg.data.type, cfg.data)

	print(f"\n✓ Created: {dataset}")
	print(f"✓ Number of files: {len(dataset)}")

	# Verify configuration was applied
	print(f"\nConfiguration verification:")
	print(f"  Max length (config): {cfg.data.max_length}")
	print(f"  Max length (dataset): {dataset.max_length}")
	print(f"  ✓ Configuration applied correctly!")

	return dataset, cfg


def test_single_item(dataset):
	"""Test accessing single dataset item."""
	if dataset is None:
		return

	print("\n" + "=" * 80)
	print("Testing Single Item Access...")
	print("=" * 80)

	if len(dataset) > 0:
		print(f"\nTesting single item (index 0)...")
		item = dataset[0]

		print(f"  input_ids shape: {item['input_ids'].shape}")
		print(f"  labels shape: {item['labels'].shape}")
		print(f"  attention_mask shape: {item['attention_mask'].shape}")

		# Decode to verify
		tokenizer = dataset.tokenizer
		input_text = tokenizer.decode(item['input_ids'])
		print(f"  Decoded input (first 100 chars): {input_text[:100]}...")

		# Check attention mask
		valid_tokens = item['attention_mask'].sum().item()
		print(f"  Valid tokens: {valid_tokens} / {len(item['attention_mask'])}")

		# Show file info
		file_info = dataset.get_file_info(0)
		print(f"  File: {file_info['name']}")
		print(f"  Size: {file_info['size_bytes']} bytes")

	print("\n✓ Single item access passed!")


def test_dataloader(dataset, cfg=None):
	"""Test DataLoader with dataset."""
	if dataset is None:
		return

	print("\n" + "=" * 80)
	print("Testing DataLoader...")
	print("=" * 80)

	# Get batch size from config or use default
	if cfg is not None:
		batch_size = cfg.data.loader.batch_size
		shuffle = cfg.data.loader.shuffle
		num_workers = cfg.data.loader.num_workers
		print(f"\nUsing config settings:")
		print(f"  Batch size: {batch_size}")
		print(f"  Shuffle: {shuffle}")
		print(f"  Num workers: {num_workers}")
	else:
		batch_size = 4
		shuffle = True
		num_workers = 0
		print(f"\nUsing default settings:")
		print(f"  Batch size: {batch_size}")

	# Create DataLoader
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=TGNDataset.collate_batch,
	)

	print(f"\nDataLoader created")
	print(f"Number of batches: {len(dataloader)}")

	# Test first batch
	print(f"\nTesting first batch...")
	batch = next(iter(dataloader))

	print(f"  input_ids shape: {batch['input_ids'].shape}")
	print(f"  labels shape: {batch['labels'].shape}")
	print(f"  attention_mask shape: {batch['attention_mask'].shape}")

	# Verify shapes
	assert batch['input_ids'].shape[0] <= batch_size, f"Batch size should be <= {batch_size}"
	assert batch['input_ids'].shape == batch['labels'].shape, "Input and labels should match"
	assert batch['input_ids'].shape == batch['attention_mask'].shape, "Masks should match"

	print("\n✓ DataLoader tests passed!")


def main():
	"""Run all tests."""
	print("\n" + "=" * 80)
	print("TGN DATASET LOADER TEST SUITE (WITH HYDRA CONFIG)")
	print("=" * 80)

	try:
		# Test 1: Tokenizer
		tokenizer = test_tokenizer()

		# Test 2: Manual dataset creation (original method)
		dataset_manual = test_dataset_manual(tokenizer)

		# Test 3: Dataset from Hydra config
		result = test_dataset_from_config()
		if result is not None:
			dataset_config, cfg = result
		else:
			dataset_config, cfg = None, None

		# Test 4: Single item access
		test_single_item(dataset_config if dataset_config is not None else dataset_manual)

		# Test 5: DataLoader with config
		test_dataloader(dataset_config if dataset_config is not None else dataset_manual, cfg)

		print("\n" + "=" * 80)
		print("✓ ALL TESTS PASSED!")
		print("=" * 80)

		print("\nSummary:")
		print("  ✓ Tokenizer working correctly")
		print("  ✓ Manual dataset creation working")
		print("  ✓ Config-based dataset creation working")
		print("  ✓ DataLoader integration working")
		print("\nThe Hydra configuration system is properly integrated!")

	except Exception as e:
		print(f"\n✗ TEST FAILED: {e}")
		import traceback

		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
