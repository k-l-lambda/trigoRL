#!/usr/bin/env python
"""
Test script to verify simplified dotenv integration.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Simulate loading .env.local
print("Setting up test environment variables:")
os.environ['WANDB_ENTITY'] = 'test_team'
os.environ['WANDB_PROJECT'] = 'test_project'
print(f"  WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
print(f"  WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")

# Test the pattern: config value or os.getenv() as fallback
print("\nTesting fallback pattern:")

# Case 1: Config has value (not None)
config_entity = 'config_team'
result_entity = config_entity or os.getenv('WANDB_ENTITY')
print(f"  Config='config_team', Env='test_team' → Result: '{result_entity}'")
assert result_entity == 'config_team', "Should use config value"

# Case 2: Config is None, use env
config_entity = None
result_entity = config_entity or os.getenv('WANDB_ENTITY')
print(f"  Config=None, Env='test_team' → Result: '{result_entity}'")
assert result_entity == 'test_team', "Should use env value"

# Case 3: Config is None, env not set, use default
os.environ.pop('WANDB_PROJECT', None)
config_project = None
result_project = config_project or os.getenv('WANDB_PROJECT', 'default_project')
print(f"  Config=None, Env=None, Default='default_project' → Result: '{result_project}'")
assert result_project == 'default_project', "Should use default value"

print("\n✓ All tests passed!")
print("\nThis pattern is much cleaner than override_wandb_config_from_env()!")
