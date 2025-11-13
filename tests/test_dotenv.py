#!/usr/bin/env python
"""
Test script to verify dotenv configuration loading.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from omegaconf import OmegaConf

# Create a minimal config for testing
config = OmegaConf.create({
	'training': {
		'wandb': {
			'enabled': False,
			'entity': None,
			'project': 'trigor',
		}
	}
})

print("Initial config:")
print(OmegaConf.to_yaml(config))

# Load test env file
print("\nLoading .env.local.test...")
load_dotenv('.env.local.test')

# Check what was loaded
print("\nEnvironment variables loaded:")
print(f"  WANDB_API_KEY: {'Yes' if os.getenv('WANDB_API_KEY') else 'No'}")
print(f"  WANDB_ENABLED: {os.getenv('WANDB_ENABLED')}")
print(f"  WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
print(f"  WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")

# Apply overrides (import the function)
sys.path.insert(0, str(Path(__file__).parent.parent))

def override_wandb_config_from_env(config):
	"""Override wandb configuration with environment variables."""
	env_mappings = {
		'WANDB_API_KEY': None,
		'WANDB_ENTITY': 'training.wandb.entity',
		'WANDB_PROJECT': 'training.wandb.project',
		'WANDB_ENABLED': 'training.wandb.enabled',
	}

	print("\nApplying environment variable overrides:")
	for env_var, config_path in env_mappings.items():
		value = os.getenv(env_var)
		if value is not None:
			if env_var == 'WANDB_API_KEY':
				os.environ['WANDB_API_KEY'] = value
				print(f"  {env_var}: ****** (masked)")
			elif env_var == 'WANDB_ENABLED':
				enabled = value.lower() in ('true', '1', 'yes', 'on')
				OmegaConf.update(config, config_path, enabled)
				print(f"  {env_var}: {enabled}")
			else:
				OmegaConf.update(config, config_path, value)
				print(f"  {env_var}: {value}")

override_wandb_config_from_env(config)

print("\nFinal config:")
print(OmegaConf.to_yaml(config))

print("\n✓ Test completed successfully!")
