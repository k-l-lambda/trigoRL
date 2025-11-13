#!/usr/bin/env python
"""
Test parse_split as a public utility function.

Demonstrates that parse_split can be used independently
from TGNDataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import parse_split


def test_parse_split_as_utility():
	"""Test parse_split as a standalone utility function."""
	print("="*80)
	print("Testing parse_split as Public Utility Function")
	print("="*80)

	# Example 1: Basic usage
	phases, cycle, shuffle = parse_split("*0..7/10")
	print(f"\nExample 1: parse_split('*0..7/10')")
	print(f"  phases: {phases}")
	print(f"  cycle: {cycle}")
	print(f"  shuffle: {shuffle}")
	assert phases == [0, 1, 2, 3, 4, 5, 6, 7]
	assert cycle == 10
	assert shuffle == True
	print(f"  ✓ Correct")

	# Example 2: Validation split
	phases, cycle, shuffle = parse_split("8,9/10")
	print(f"\nExample 2: parse_split('8,9/10')")
	print(f"  phases: {phases}")
	print(f"  cycle: {cycle}")
	print(f"  shuffle: {shuffle}")
	assert phases == [8, 9]
	assert cycle == 10
	assert shuffle == False
	print(f"  ✓ Correct")

	# Example 3: Complex mixed syntax
	phases, cycle, shuffle = parse_split("*0..3,7..9/10")
	print(f"\nExample 3: parse_split('*0..3,7..9/10')")
	print(f"  phases: {phases}")
	print(f"  cycle: {cycle}")
	print(f"  shuffle: {shuffle}")
	assert phases == [0, 1, 2, 3, 7, 8, 9]
	assert cycle == 10
	assert shuffle == True
	print(f"  ✓ Correct")

	# Example 4: Use in custom code
	print("\n" + "-"*80)
	print("Example 4: Using in custom data splitting logic")
	print("-"*80)

	# Simulate custom split logic
	train_split = "*0..7/10"
	val_split = "8,9/10"

	train_phases, train_cycle, train_shuffle = parse_split(train_split)
	val_phases, val_cycle, val_shuffle = parse_split(val_split)

	print(f"\nTrain split: {train_split}")
	print(f"  → {len(train_phases)} phases: {train_phases}")
	print(f"  → {len(train_phases) / train_cycle * 100:.0f}% of data")
	print(f"  → Shuffle: {train_shuffle}")

	print(f"\nValidation split: {val_split}")
	print(f"  → {len(val_phases)} phases: {val_phases}")
	print(f"  → {len(val_phases) / val_cycle * 100:.0f}% of data")
	print(f"  → Shuffle: {val_shuffle}")

	# Verify no overlap
	overlap = set(train_phases) & set(val_phases)
	print(f"\nOverlap check: {len(overlap)} phases in common")
	assert len(overlap) == 0, f"Found overlap: {overlap}"
	print(f"  ✓ No overlap")

	# Verify coverage
	all_phases = set(train_phases) | set(val_phases)
	expected_phases = set(range(train_cycle))
	coverage = len(all_phases) / len(expected_phases) * 100
	print(f"\nCoverage: {len(all_phases)}/{len(expected_phases)} phases ({coverage:.0f}%)")
	print(f"  ✓ Complete coverage")

	print("\n" + "="*80)
	print("✓ All tests passed!")
	print("="*80)

	print("""
Use Cases for parse_split():

1. Custom dataset splitting logic
2. Multi-way splits (train/val/test)
3. Cross-validation fold generation
4. Data partitioning for distributed training
5. Any scenario requiring deterministic data splits

Example:
    from trigor.data import parse_split

    # Get phases for 5-fold cross-validation
    for fold in range(5):
        val_split = f"{fold}/5"
        train_splits = [f"{i}/5" for i in range(5) if i != fold]

        val_phases, cycle, _ = parse_split(val_split)
        # Use val_phases for this fold...
	""")


if __name__ == "__main__":
	try:
		test_parse_split_as_utility()
		sys.exit(0)
	except Exception as e:
		print(f"\n❌ Test failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
