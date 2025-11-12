#!/usr/bin/env python3
"""
Test script for dataset factory and registry.

Demonstrates creating datasets from config files using the factory pattern.
"""

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import list_datasets, make_dataset


def test_dataset_registry():
	"""Test dataset registry functions."""
	print("=" * 60)
	print("Testing Dataset Registry")
	print("=" * 60)

	# List registered datasets
	datasets = list_datasets()
	print(f"\nRegistered datasets: {datasets}")
	assert "TGNDataset" in datasets, "TGNDataset should be registered"
	print("✓ Registry contains expected datasets")


def test_dataset_from_config():
	"""Test creating dataset from configuration."""
	print("\n" + "=" * 60)
	print("Testing Dataset Creation from Config")
	print("=" * 60)

	# Create a config dictionary (mimics Hydra config)
	config = {
		'type': 'TGNDataset',
		'data_dir': 'third_party/trigo/trigo-web/tools/output',
		'max_length': 512,
		'min_length': 10,
		'max_file_size': 10000,
		'tokenizer_config': {},
	}

	print(f"\nConfig:\n{OmegaConf.create(config)}")

	# Create dataset using factory
	try:
		dataset = make_dataset(dataset_type=config['type'], config=config)
		print(f"\n✓ Created dataset: {dataset}")

		# Check dataset properties
		print(f"\nDataset length: {len(dataset)}")
		assert len(dataset) > 0, "Dataset should have samples"

		# Get dataset stats
		stats = dataset.get_stats()
		print(f"\nDataset statistics:")
		for key, value in stats.items():
			if isinstance(value, float):
				print(f"  {key}: {value:.2f}")
			else:
				print(f"  {key}: {value}")

		return dataset

	except Exception as e:
		print(f"✗ Failed to create dataset: {e}")
		raise


def test_dataset_iteration(dataset):
	"""Test iterating through dataset."""
	print("\n" + "=" * 60)
	print("Testing Dataset Iteration")
	print("=" * 60)

	# Get a single sample
	sample = dataset[0]
	print(f"\nSample keys: {list(sample.keys())}")
	print(f"Input shape: {sample['input_ids'].shape}")
	print(f"Labels shape: {sample['labels'].shape}")
	print(f"Attention mask shape: {sample['attention_mask'].shape}")

	# Check tensor properties
	assert sample['input_ids'].shape == sample['labels'].shape, "Input and labels should have same shape"
	assert sample['input_ids'].shape == sample['attention_mask'].shape, "Input and mask should have same shape"
	print("✓ Tensor shapes are consistent")

	# Check first few tokens
	print(f"\nFirst 10 input tokens: {sample['input_ids'][:10].tolist()}")
	print(f"First 10 label tokens: {sample['labels'][:10].tolist()}")

	# Get file info
	file_info = dataset.get_file_info(0)
	print(f"\nFile info for sample 0:")
	for key, value in file_info.items():
		print(f"  {key}: {value}")


def test_dataloader(dataset):
	"""Test DataLoader with collate function."""
	print("\n" + "=" * 60)
	print("Testing DataLoader Integration")
	print("=" * 60)

	# Import the dataset class to get collate_batch
	from trigor.data import TGNDataset

	# Create DataLoader
	dataloader = DataLoader(
		dataset,
		batch_size=4,
		shuffle=True,
		collate_fn=TGNDataset.collate_batch,
	)

	print(f"DataLoader created with batch_size=4")

	# Get a batch
	batch = next(iter(dataloader))
	print(f"\nBatch keys: {list(batch.keys())}")
	print(f"Batch input shape: {batch['input_ids'].shape}")
	print(f"Batch labels shape: {batch['labels'].shape}")
	print(f"Batch attention mask shape: {batch['attention_mask'].shape}")

	# Check batch dimensions
	batch_size = batch['input_ids'].shape[0]
	assert batch_size == 4, f"Expected batch size 4, got {batch_size}"
	print(f"✓ Batch size correct: {batch_size}")


def test_hydra_config_loading():
	"""Test loading dataset config from YAML files."""
	print("\n" + "=" * 60)
	print("Testing Hydra Config Loading")
	print("=" * 60)

	config_files = [
		'configs/dataset/tgn_default.yaml',
		'configs/dataset/tgn_small.yaml',
		'configs/dataset/tgn_large.yaml',
	]

	for config_file in config_files:
		config_path = Path(config_file)
		if not config_path.exists():
			print(f"⚠ Config file not found: {config_file}")
			continue

		# Load config
		cfg = OmegaConf.load(config_path)
		print(f"\n✓ Loaded config: {config_file}")
		print(f"  Type: {cfg.type}")
		print(f"  Max length: {cfg.max_length}")
		print(f"  Batch size: {cfg.dataloader.batch_size}")


def main():
	"""Run all tests."""
	print("\n" + "=" * 60)
	print("TGN Dataset Factory Test Suite")
	print("=" * 60)

	try:
		# Test 1: Registry
		test_dataset_registry()

		# Test 2: Create dataset from config
		dataset = test_dataset_from_config()

		# Test 3: Dataset iteration
		test_dataset_iteration(dataset)

		# Test 4: DataLoader
		test_dataloader(dataset)

		# Test 5: Hydra config loading
		test_hydra_config_loading()

		print("\n" + "=" * 60)
		print("All tests passed! ✓")
		print("=" * 60)

	except Exception as e:
		print(f"\n✗ Test failed with error: {e}")
		import traceback

		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
