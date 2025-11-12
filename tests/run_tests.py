#!/usr/bin/env python3
"""
Run all tests in the tests directory.
"""

import subprocess
import sys
from pathlib import Path

# Get project root
tests_dir = Path(__file__).parent
project_root = tests_dir.parent
python_exe = project_root / "env" / "bin" / "python"

# Test files to run
test_files = [
	"test_tgn_dataset.py",
	"test_dataset_factory.py",
	"test_register_decorator.py",
	"test_from_config.py",
]

print("=" * 60)
print("Running All Tests")
print("=" * 60)
print()

passed = 0
failed = 0
failed_tests = []

for test_file in test_files:
	test_path = tests_dir / test_file
	print(f"Running: {test_file}")
	print("-" * 60)

	try:
		result = subprocess.run(
			[str(python_exe), str(test_path)],
			capture_output=True,
			text=True,
			check=True,
		)
		print(f"✓ {test_file} PASSED")
		passed += 1
	except subprocess.CalledProcessError as e:
		print(f"✗ {test_file} FAILED")
		print(f"Error output:\n{e.stderr}")
		failed += 1
		failed_tests.append(test_file)

	print()

# Summary
print("=" * 60)
print("Test Summary")
print("=" * 60)
print(f"Total tests: {passed + failed}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed > 0:
	print("\nFailed tests:")
	for test in failed_tests:
		print(f"  - {test}")
	sys.exit(1)
else:
	print("\n✓ All tests passed!")
	sys.exit(0)
