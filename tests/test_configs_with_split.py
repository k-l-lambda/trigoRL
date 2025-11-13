#!/usr/bin/env python
"""
Test that all training configs work with the new split feature.

Verifies that train and validation datasets can be created from
all training configuration files.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNDataset


def test_config_with_split(config_path: Path):
	"""Test a single config file with split."""
	print(f"\nTesting: {config_path.name}")
	print("-" * 60)

	# Load config
	cfg = OmegaConf.load(config_path)
	OmegaConf.update(cfg, "paths.root", str(project_root))
	OmegaConf.resolve(cfg)

	# Check if split config exists
	if not hasattr(cfg.data, 'train_split') or not hasattr(cfg.data, 'val_split'):
		print(f"⚠ Config missing split specifications")
		return False

	print(f"Train split: {cfg.data.train_split}")
	print(f"Val split: {cfg.data.val_split}")

	try:
		# Create training dataset
		train_config = OmegaConf.create({**cfg.data, 'split': cfg.data.train_split})
		train_dataset = TGNDataset.from_config(train_config)
		print(f"✓ Training dataset: {len(train_dataset)} files")

		# Create validation dataset
		val_config = OmegaConf.create({**cfg.data, 'split': cfg.data.val_split})
		val_dataset = TGNDataset.from_config(val_config)
		print(f"✓ Validation dataset: {len(val_dataset)} files")

		# Verify no overlap
		train_files = {f.name for f in train_dataset.files}
		val_files = {f.name for f in val_dataset.files}
		overlap = train_files & val_files

		if len(overlap) > 0:
			print(f"❌ Found {len(overlap)} overlapping files!")
			return False

		print(f"✓ No overlap between train and val")

		# Test data loading
		train_sample = train_dataset[0]
		val_sample = val_dataset[0]

		assert 'input_ids' in train_sample
		assert 'labels' in train_sample
		assert 'attention_mask' in train_sample

		assert 'input_ids' in val_sample
		assert 'labels' in val_sample
		assert 'attention_mask' in val_sample

		print(f"✓ Data loading works")
		print(f"✓ Config {config_path.name} passed all tests")

		return True

	except Exception as e:
		print(f"❌ Error: {e}")
		import traceback
		traceback.print_exc()
		return False


def main():
	"""Test all training configs."""
	print("="*80)
	print("Testing Training Configs with Split Feature")
	print("="*80)

	configs_dir = project_root / "configs/training"
	config_files = sorted(configs_dir.glob("trigo-*.yaml"))

	if not config_files:
		print(f"❌ No config files found in {configs_dir}")
		return 1

	print(f"\nFound {len(config_files)} config files:")
	for cf in config_files:
		print(f"  - {cf.name}")

	results = {}
	for config_path in config_files:
		success = test_config_with_split(config_path)
		results[config_path.name] = success

	# Summary
	print("\n" + "="*80)
	print("SUMMARY")
	print("="*80)

	passed = sum(1 for v in results.values() if v)
	total = len(results)

	for name, success in sorted(results.items()):
		status = "✓ PASS" if success else "❌ FAIL"
		print(f"{status}: {name}")

	print(f"\nPassed: {passed}/{total}")

	if passed == total:
		print("\n✓ ALL CONFIGS PASSED!")
		return 0
	else:
		print(f"\n❌ {total - passed} config(s) failed")
		return 1


if __name__ == "__main__":
	sys.exit(main())
