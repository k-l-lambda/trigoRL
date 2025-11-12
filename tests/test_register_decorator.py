#!/usr/bin/env python3
"""
Test script for register_dataset as a decorator.

Tests both decorator and function call usage patterns.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import DATASETS, list_datasets, register_dataset


def test_decorator_usage():
	"""Test using register_dataset as a decorator."""
	print("=" * 60)
	print("Testing Decorator Usage")
	print("=" * 60)

	# Define a dataset using decorator
	@register_dataset('DecoratorDataset')
	class DecoratorDataset(Dataset):
		"""Dataset registered using decorator syntax."""

		def __init__(self, size: int = 10):
			self.size = size

		def __len__(self):
			return self.size

		def __getitem__(self, idx):
			return {'data': torch.randn(5), 'label': idx}

	print(f"\n✓ DecoratorDataset defined with @register_dataset")

	# Verify it's registered
	assert 'DecoratorDataset' in DATASETS, "DecoratorDataset should be in registry"
	print(f"✓ DecoratorDataset found in registry")

	# Verify we can instantiate it
	dataset = DecoratorDataset(size=5)
	assert len(dataset) == 5, "Dataset should have correct size"
	print(f"✓ DecoratorDataset instantiated: {dataset}")

	# Verify it works correctly
	item = dataset[0]
	assert 'data' in item and 'label' in item, "Item should have correct keys"
	print(f"✓ Dataset item has correct structure: {list(item.keys())}")


def test_function_usage():
	"""Test using register_dataset as a function."""
	print("\n" + "=" * 60)
	print("Testing Function Usage")
	print("=" * 60)

	# Define a dataset class
	class FunctionDataset(Dataset):
		"""Dataset registered using function call syntax."""

		def __init__(self, data_size: int = 20):
			self.data_size = data_size

		def __len__(self):
			return self.data_size

		def __getitem__(self, idx):
			return torch.randn(10)

	# Register it using function call
	register_dataset('FunctionDataset', FunctionDataset)
	print(f"\n✓ FunctionDataset registered with function call")

	# Verify it's registered
	assert 'FunctionDataset' in DATASETS, "FunctionDataset should be in registry"
	print(f"✓ FunctionDataset found in registry")

	# Verify we can instantiate it
	dataset = FunctionDataset(data_size=15)
	assert len(dataset) == 15, "Dataset should have correct size"
	print(f"✓ FunctionDataset instantiated: {dataset}")

	# Verify it works correctly
	item = dataset[0]
	assert item.shape == torch.Size([10]), "Item should have correct shape"
	print(f"✓ Dataset item has correct shape: {item.shape}")


def test_registry_listing():
	"""Test listing all registered datasets."""
	print("\n" + "=" * 60)
	print("Testing Registry Listing")
	print("=" * 60)

	datasets = list_datasets()
	print(f"\nRegistered datasets: {datasets}")

	# Should have at least TGNDataset, DecoratorDataset, FunctionDataset
	assert 'TGNDataset' in datasets, "TGNDataset should be registered"
	assert 'DecoratorDataset' in datasets, "DecoratorDataset should be registered"
	assert 'FunctionDataset' in datasets, "FunctionDataset should be registered"

	print(f"✓ Found {len(datasets)} registered datasets")
	for name in datasets:
		print(f"  - {name}")


def test_error_handling():
	"""Test that invalid datasets are rejected."""
	print("\n" + "=" * 60)
	print("Testing Error Handling")
	print("=" * 60)

	# Test registering non-Dataset class with decorator
	try:

		@register_dataset('InvalidDataset')
		class InvalidDataset:
			"""Not a Dataset subclass."""

			pass

		print("✗ Should have raised ValueError for non-Dataset class")
		assert False, "Should have raised ValueError"
	except ValueError as e:
		print(f"✓ Correctly rejected non-Dataset class: {e}")

	# Test registering non-Dataset class with function
	try:

		class AnotherInvalid:
			pass

		register_dataset('AnotherInvalid', AnotherInvalid)
		print("✗ Should have raised ValueError for non-Dataset class")
		assert False, "Should have raised ValueError"
	except ValueError as e:
		print(f"✓ Correctly rejected non-Dataset class: {e}")


def test_decorator_returns_class():
	"""Test that decorator returns the class unchanged."""
	print("\n" + "=" * 60)
	print("Testing Decorator Returns Class")
	print("=" * 60)

	@register_dataset('ReturnTestDataset')
	class ReturnTestDataset(Dataset):
		"""Test that we can use the decorated class."""

		def __init__(self):
			self.value = 42

		def __len__(self):
			return 1

		def __getitem__(self, idx):
			return self.value

	# Should be able to use the class directly
	instance = ReturnTestDataset()
	assert instance.value == 42, "Should be able to instantiate decorated class"
	print(f"✓ Decorated class can be instantiated directly")

	# Should also be in registry
	assert 'ReturnTestDataset' in DATASETS, "Should be in registry"
	registry_class = DATASETS['ReturnTestDataset']
	registry_instance = registry_class()
	assert registry_instance.value == 42, "Registry class should work"
	print(f"✓ Registry class works correctly")


def main():
	"""Run all tests."""
	print("\n" + "=" * 60)
	print("Register Dataset Decorator Test Suite")
	print("=" * 60)

	try:
		# Test 1: Decorator usage
		test_decorator_usage()

		# Test 2: Function usage
		test_function_usage()

		# Test 3: List all datasets
		test_registry_listing()

		# Test 4: Error handling
		test_error_handling()

		# Test 5: Decorator returns class
		test_decorator_returns_class()

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
