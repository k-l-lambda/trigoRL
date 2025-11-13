#!/usr/bin/env python
"""
Demonstrate positional config argument parsing without running training.
"""

import sys
from pathlib import Path


def parse_positional_config():
	"""Parse positional argument as config name/path."""
	if len(sys.argv) > 1:
		first_arg = sys.argv[1]

		if first_arg.startswith('-') or '=' in first_arg:
			return None

		config_path = Path(first_arg)

		if config_path.suffix in ['.yaml', '.yml']:
			config_name = config_path.stem
			return config_name, 'path'
		else:
			config_name = first_arg
			return config_name, 'short'

	return None


def main():
	"""Test the parsing logic."""
	print("="*80)
	print("train_lm.py Positional Config Argument Demo")
	print("="*80)
	print()

	test_cases = [
		['train_lm.py', 'trigo-gpt2'],
		['train_lm.py', 'trigo-llama'],
		['train_lm.py', 'trigo-gpt2-invsqrt'],
		['train_lm.py', 'configs/training/trigo-gpt2.yaml'],
		['train_lm.py', 'configs/training/trigo-llama.yaml'],
		['train_lm.py', 'trigo-gpt2', 'training.epochs=50'],
		['train_lm.py', '--config-name=trigo-gpt2'],
		['train_lm.py', 'training.epochs=50'],
		['train_lm.py'],
	]

	for test in test_cases:
		sys.argv = test.copy()
		original = ' '.join(test[1:]) if len(test) > 1 else '(no args)'

		result = parse_positional_config()

		print(f"Input:  {original}")
		if result:
			config_name, parse_type = result
			print(f"Parsed: {config_name} (from {parse_type} name)")
			print(f"Will load: configs/training/{config_name}.yaml")
		elif len(test) > 1 and (test[1].startswith('-') or '=' in test[1]):
			print(f"Parsed: (Hydra syntax - no change)")
		else:
			print(f"Parsed: (default config)")
		print()


if __name__ == '__main__':
	main()
