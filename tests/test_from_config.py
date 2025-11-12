#!/usr/bin/env python3
"""
Test script for from_config pattern.

Demonstrates how datasets handle their own construction logic
using the from_config classmethod.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNDataset, list_datasets, make_dataset, register_dataset


def test_tgn_from_config():
	"""Test TGNDataset.from_config() method directly."""
	print("=" * 60)
	print("Test 1: TGNDataset.from_config() Direct Usage")
	print("=" * 60)

	config = {
		'data_dir': 'third_party/trigo/trigo-web/tools/output',
		'max_length': 512,
		'min_length': 10,
		'max_file_size': 10000,
	}

	# Create dataset directly using from_config
	dataset = TGNDataset.from_config(config)
	print(f"\n✓ Created: {dataset}")
	print(f"✓ Length: {len(dataset)}")
	print(f"✓ Max length: {dataset.max_length}")


def test_make_dataset_uses_from_config():
	"""Test that make_dataset automatically uses from_config if available."""
	print("\n" + "=" * 60)
	print("Test 2: make_dataset() Auto-detects from_config")
	print("=" * 60)

	config = {
		'data_dir': 'third_party/trigo/trigo-web/tools/output',
		'max_length': 1024,
	}

	# make_dataset should automatically call TGNDataset.from_config()
	dataset = make_dataset('TGNDataset', config)
	print(f"\n✓ make_dataset created: {dataset}")
	print(f"✓ Max length: {dataset.max_length}")
	print("✓ make_dataset automatically used from_config()")


def test_custom_dataset_with_from_config():
	"""Test custom dataset implementing from_config pattern."""
	print("\n" + "=" * 60)
	print("Test 3: Custom Dataset with from_config")
	print("=" * 60)

	@register_dataset('ConfigurableDataset')
	class ConfigurableDataset(Dataset):
		"""Custom dataset that implements from_config pattern."""

		@classmethod
		def from_config(cls, config):
			"""Create dataset from config with preprocessing."""
			# Extract and process config
			size = config.get('size', 100)
			feature_dim = config.get('feature_dim', 10)
			normalize = config.get('normalize', False)

			# Additional preprocessing
			if normalize:
				print("  Applying normalization preprocessing")

			return cls(size=size, feature_dim=feature_dim, normalize=normalize)

		def __init__(self, size: int, feature_dim: int, normalize: bool = False):
			self.size = size
			self.feature_dim = feature_dim
			self.normalize = normalize
			self.data = torch.randn(size, feature_dim)

			if self.normalize:
				self.data = (self.data - self.data.mean()) / self.data.std()

		def __len__(self):
			return self.size

		def __getitem__(self, idx):
			return self.data[idx]

	# Use from_config through make_dataset
	config = {'size': 50, 'feature_dim': 20, 'normalize': True}
	dataset = make_dataset('ConfigurableDataset', config)

	print(f"\n✓ Created: {dataset}")
	print(f"✓ Size: {len(dataset)}")
	print(f"✓ Data shape: {dataset.data.shape}")
	print(f"✓ Normalized: {dataset.normalize}")


def test_custom_dataset_without_from_config():
	"""Test custom dataset without from_config (fallback to __init__)."""
	print("\n" + "=" * 60)
	print("Test 4: Custom Dataset WITHOUT from_config (Fallback)")
	print("=" * 60)

	@register_dataset('SimpleDataset')
	class SimpleDataset(Dataset):
		"""Dataset without from_config - uses direct __init__."""

		def __init__(self, count: int = 10):
			self.count = count

		def __len__(self):
			return self.count

		def __getitem__(self, idx):
			return torch.randn(5)

	# Config passed directly to __init__
	config = {'count': 25}
	dataset = make_dataset('SimpleDataset', config)

	print(f"\n✓ Created: {dataset}")
	print(f"✓ Length: {len(dataset)}")
	print("✓ make_dataset fell back to __init__(**config)")


def test_from_config_with_complex_logic():
	"""Test from_config with complex initialization logic."""
	print("\n" + "=" * 60)
	print("Test 5: from_config with Complex Initialization")
	print("=" * 60)

	@register_dataset('ComplexDataset')
	class ComplexDataset(Dataset):
		"""Dataset with complex from_config logic."""

		@classmethod
		def from_config(cls, config):
			"""Complex config processing."""
			# Multiple config sources
			base_config = config.get('base', {})
			augmentation_config = config.get('augmentation', {})
			loader_config = config.get('loader', {})

			# Merge and process
			merged = {
				'data_path': config['data_path'],
				'augment': augmentation_config.get('enabled', False),
				'aug_prob': augmentation_config.get('probability', 0.5),
				'batch_loading': loader_config.get('batch', False),
			}

			print(f"  Processed config: {merged}")
			return cls(**merged)

		def __init__(self, data_path, augment=False, aug_prob=0.5, batch_loading=False):
			self.data_path = data_path
			self.augment = augment
			self.aug_prob = aug_prob
			self.batch_loading = batch_loading
			self.data = torch.randn(20, 10)

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			sample = self.data[idx]
			if self.augment and torch.rand(1).item() < self.aug_prob:
				sample = sample + torch.randn_like(sample) * 0.1
			return sample

	# Nested config
	config = {
		'data_path': '/path/to/data',
		'augmentation': {'enabled': True, 'probability': 0.8},
		'loader': {'batch': True},
	}

	dataset = make_dataset('ComplexDataset', config)
	print(f"\n✓ Created complex dataset")
	print(f"✓ Augmentation: {dataset.augment} (prob={dataset.aug_prob})")
	print(f"✓ Batch loading: {dataset.batch_loading}")


def test_registry_listing():
	"""Test listing all datasets including new ones."""
	print("\n" + "=" * 60)
	print("Test 6: Registry Listing")
	print("=" * 60)

	datasets = list_datasets()
	print(f"\nRegistered datasets ({len(datasets)} total):")
	for name in datasets:
		cls = __import__('trigor.data', fromlist=['DATASETS']).DATASETS[name]
		has_from_config = hasattr(cls, 'from_config')
		marker = "✓ from_config" if has_from_config else "✗ __init__ only"
		print(f"  - {name:30s} {marker}")


def main():
	"""Run all tests."""
	print("\n" + "=" * 60)
	print("from_config Pattern Test Suite")
	print("=" * 60)

	try:
		test_tgn_from_config()
		test_make_dataset_uses_from_config()
		test_custom_dataset_with_from_config()
		test_custom_dataset_without_from_config()
		test_from_config_with_complex_logic()
		test_registry_listing()

		print("\n" + "=" * 60)
		print("Summary")
		print("=" * 60)
		print(
			"""
The from_config Pattern:
  ✓ Datasets handle their own construction logic
  ✓ Clean separation: dataset knows how to build itself
  ✓ make_dataset automatically detects from_config
  ✓ Falls back to __init__(**config) if no from_config
  ✓ Supports complex preprocessing and validation
  ✓ Decorator + from_config = fully self-contained datasets

All tests passed! ✓
"""
		)

	except Exception as e:
		print(f"\n✗ Test failed with error: {e}")
		import traceback

		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
