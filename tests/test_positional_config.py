#!/usr/bin/env python
"""
Test script for train_lm.py positional config argument feature.

Tests different ways of specifying the config file.
"""

import subprocess
import sys
from pathlib import Path


def run_test(description, args, expect_config_name):
	"""Run a test case."""
	print(f"\n{'='*80}")
	print(f"Test: {description}")
	print(f"Command: python train_lm.py {' '.join(args)}")
	print(f"Expected config: {expect_config_name}")
	print('='*80)

	# Run train_lm.py with the given args (just show config, don't train)
	cmd = [sys.executable, 'train_lm.py'] + args
	result = subprocess.run(
		cmd,
		capture_output=True,
		text=True,
		timeout=10
	)

	# Check if it loaded the expected config
	if expect_config_name:
		# Look for config name in hydra output dir path
		if f'config_name: {expect_config_name}' in result.stdout or \
		   f'/{expect_config_name}/' in result.stdout or \
		   expect_config_name in result.stdout:
			print(f"✓ SUCCESS: Loaded config '{expect_config_name}'")
			return True
		else:
			print(f"✗ FAILED: Did not find config '{expect_config_name}'")
			print("\nStdout:")
			print(result.stdout[:500])
			print("\nStderr:")
			print(result.stderr[:500])
			return False
	else:
		# Just check if it ran without error
		if result.returncode == 0 or 'Configuration:' in result.stdout:
			print("✓ SUCCESS: Script ran")
			return True
		else:
			print("✗ FAILED: Script error")
			print("\nStderr:")
			print(result.stderr[:500])
			return False


def main():
	"""Run all tests."""
	print("="*80)
	print("Testing train_lm.py Positional Config Argument Feature")
	print("="*80)

	project_root = Path(__file__).parent.parent
	print(f"\nProject root: {project_root}")
	print(f"Working directory: {Path.cwd()}")

	# Change to project root
	import os
	os.chdir(project_root)
	print(f"Changed to: {Path.cwd()}")

	tests = [
		# (description, args, expected_config_name)
		("Default config (no args)", [], "trigo-gpt2"),
		("Short config name", ["trigo-llama"], "trigo-llama"),
		("Short config name (invsqrt)", ["trigo-gpt2-invsqrt"], "trigo-gpt2-invsqrt"),
		("Full config path", ["configs/training/trigo-llama.yaml"], "trigo-llama"),
		("Hydra syntax (backward compat)", ["--config-name=trigo-gpt2"], "trigo-gpt2"),
	]

	passed = 0
	failed = 0

	for description, args, expected_config in tests:
		try:
			if run_test(description, args, expected_config):
				passed += 1
			else:
				failed += 1
		except subprocess.TimeoutExpired:
			print("✗ FAILED: Timeout")
			failed += 1
		except Exception as e:
			print(f"✗ FAILED: {e}")
			failed += 1

	# Summary
	print("\n" + "="*80)
	print("Test Summary")
	print("="*80)
	print(f"Passed: {passed}/{passed+failed}")
	print(f"Failed: {failed}/{passed+failed}")

	if failed == 0:
		print("\n✓ All tests passed!")
		return 0
	else:
		print(f"\n✗ {failed} test(s) failed")
		return 1


if __name__ == "__main__":
	sys.exit(main())
