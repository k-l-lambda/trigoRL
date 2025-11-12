#!/usr/bin/env python3
"""
Verification script to demonstrate DictConfig support.

This script verifies that dataset classes can directly accept OmegaConf
DictConfig objects without requiring conversion to dict via to_container().
"""

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import make_dataset


def main():
	print("=" * 80)
	print("DictConfig Support Verification")
	print("=" * 80)

	# Create a DictConfig directly (not from YAML file)
	config_dict = {
		'type': 'TGNDataset',
		'data_dir': f'{project_root}/third_party/trigo/trigo-web/tools/output',
		'max_length': 512,
		'min_length': 10,
		'max_file_size': 5000,
		'tokenizer_config': {},
	}

	# Convert to DictConfig
	cfg = OmegaConf.create(config_dict)

	print(f"\nConfig type: {type(cfg)}")
	print(f"Is DictConfig: {isinstance(cfg, DictConfig)}")

	# Verify we can pass DictConfig directly without OmegaConf.to_container()
	print("\nCreating dataset with DictConfig (no conversion)...")
	dataset = make_dataset(cfg.type, cfg)

	print(f"\n✓ Success! Dataset created: {dataset}")
	print(f"✓ Dataset length: {len(dataset)}")
	print(f"✓ Max length: {dataset.max_length}")

	# Verify configuration was applied correctly
	assert dataset.max_length == 512, "Config max_length not applied"
	assert len(dataset) > 0, "Dataset should have files"

	print("\n" + "=" * 80)
	print("✓ DictConfig Support Verified!")
	print("=" * 80)
	print("\nKey Achievement:")
	print("  - No OmegaConf.to_container() conversion needed")
	print("  - Dataset classes directly support DictConfig objects")
	print("  - Both Dict and DictConfig work seamlessly")


if __name__ == "__main__":
	main()
