#!/usr/bin/env python3
"""
Quick verification that models can be loaded from trigo_test.yaml config.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.models import list_models, make_model


def main():
	print("=" * 80)
	print("Model Configuration Verification")
	print("=" * 80)

	# Load config
	config_path = project_root / "configs" / "trigo_test.yaml"
	cfg = OmegaConf.load(config_path)
	OmegaConf.update(cfg, "paths.root", str(project_root))
	OmegaConf.resolve(cfg)

	print(f"\nLoaded config from: {config_path}")
	print(f"Model type: {cfg.model.type}")
	print(f"Model config:")
	print(OmegaConf.to_yaml(cfg.model))

	# Create model from config
	print("\nCreating model from config...")
	model = make_model(cfg.model.type, cfg.model)

	print(f"\n✓ Model created successfully!")
	print(f"{model}")

	# Show model info
	info = model.get_model_info()
	print(f"\nModel details:")
	print(f"  Type: {info['model_type']}")
	print(f"  Parameters: {info['total_parameters']:,}")
	print(f"  Vocab size: {info['vocab_size']}")
	print(f"  Hidden size: {info['hidden_size']}")
	print(f"  Layers: {info['num_layers']}")

	print("\n" + "=" * 80)
	print("✓ Configuration verification successful!")
	print("=" * 80)

	print("\nAvailable model types:")
	for model_type in list_models():
		print(f"  - {model_type}")

	print("\nTo switch models, edit configs/trigo_test.yaml:")
	print("  model.type: gpt2    # or llama, rwkv, xlstm")


if __name__ == "__main__":
	main()
