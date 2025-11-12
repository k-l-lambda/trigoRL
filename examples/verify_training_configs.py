#!/usr/bin/env python3
"""
Verify all training configuration files can be loaded correctly.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.models import list_models, make_model


def verify_config(config_path: Path):
	"""Verify a single config file."""
	print(f"\n{'='*80}")
	print(f"Verifying: {config_path.name}")
	print('='*80)

	# Load config
	cfg = OmegaConf.load(config_path)
	OmegaConf.update(cfg, "paths.root", str(project_root))
	OmegaConf.resolve(cfg)

	print(f"\nModel type: {cfg.model.type}")
	print(f"Model config:")
	print(f"  vocab_size: {cfg.model.vocab_size}")
	print(f"  hidden_size: {cfg.model.hidden_size}")
	print(f"  num_layers: {cfg.model.num_layers}")
	if hasattr(cfg.model, 'num_heads'):
		print(f"  num_heads: {cfg.model.num_heads}")
	if hasattr(cfg.model, 'num_key_value_heads'):
		print(f"  num_key_value_heads: {cfg.model.num_key_value_heads}")

	print(f"\nTraining config:")
	print(f"  epochs: {cfg.training.epochs}")
	print(f"  learning_rate: {cfg.training.learning_rate}")
	print(f"  batch_size: {cfg.data.loader.batch_size}")

	print(f"\nData config:")
	print(f"  type: {cfg.data.type}")
	print(f"  max_length: {cfg.data.max_length}")

	# Try to create model from config
	print(f"\nCreating model...")
	try:
		model = make_model(cfg.model.type, cfg.model)
		info = model.get_model_info()
		print(f"✓ Model created successfully!")
		print(f"  Type: {info['model_type']}")
		print(f"  Parameters: {info['total_parameters']:,}")
		if 'attention_type' in info:
			print(f"  Attention: {info['attention_type']}")
		return True
	except Exception as e:
		print(f"✗ Model creation failed: {e}")
		return False


def main():
	print("="*80)
	print("Training Configuration Verification")
	print("="*80)

	# Find all trigo-*.yaml configs
	config_dir = project_root / "configs" / "training"
	configs = sorted(config_dir.glob("trigo-*.yaml"))

	if not configs:
		print("\n✗ No training configs found!")
		sys.exit(1)

	print(f"\nFound {len(configs)} training configurations:")
	for cfg in configs:
		print(f"  - {cfg.name}")

	# Verify each config
	results = {}
	for config_path in configs:
		try:
			results[config_path.name] = verify_config(config_path)
		except Exception as e:
			print(f"\n✗ Error verifying {config_path.name}: {e}")
			import traceback
			traceback.print_exc()
			results[config_path.name] = False

	# Summary
	print("\n" + "="*80)
	print("Verification Summary")
	print("="*80)
	for name, success in results.items():
		status = "✓" if success else "✗"
		print(f"{status} {name}")

	total = len(results)
	passed = sum(results.values())
	print(f"\nTotal: {passed}/{total} configs verified successfully")

	if passed == total:
		print("\n✓ All training configurations are valid!")
		return 0
	else:
		print(f"\n✗ {total - passed} configuration(s) failed verification")
		return 1


if __name__ == "__main__":
	sys.exit(main())
