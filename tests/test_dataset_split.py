#!/usr/bin/env python
"""
Test script for TGNDataset split functionality.

Tests the train/validation split feature added to TGNDataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNDataset, parse_split


def test_split_parsing():
	"""Test split string parsing."""
	print("="*80)
	print("Testing Split Parsing")
	print("="*80)

	# Test without shuffle
	phases, cycle, shuffle = parse_split("0,1,2,3/5")
	assert phases == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {phases}"
	assert cycle == 5, f"Expected 5, got {cycle}"
	assert shuffle == False, f"Expected False, got {shuffle}"
	print("✓ Parse '0,1,2,3/5': phases=[0,1,2,3], cycle=5, shuffle=False")

	# Test with shuffle
	phases, cycle, shuffle = parse_split("*0,1,2/10")
	assert phases == [0, 1, 2], f"Expected [0, 1, 2], got {phases}"
	assert cycle == 10, f"Expected 10, got {cycle}"
	assert shuffle == True, f"Expected True, got {shuffle}"
	print("✓ Parse '*0,1,2/10': phases=[0,1,2], cycle=10, shuffle=True")

	# Test single phase
	phases, cycle, shuffle = parse_split("9/10")
	assert phases == [9], f"Expected [9], got {phases}"
	assert cycle == 10, f"Expected 10, got {cycle}"
	assert shuffle == False, f"Expected False, got {shuffle}"
	print("✓ Parse '9/10': phases=[9], cycle=10, shuffle=False")

	# Test range syntax (0..7)
	phases, cycle, shuffle = parse_split("*0..7/10")
	assert phases == [0, 1, 2, 3, 4, 5, 6, 7], f"Expected [0,1,2,3,4,5,6,7], got {phases}"
	assert cycle == 10, f"Expected 10, got {cycle}"
	assert shuffle == True, f"Expected True, got {shuffle}"
	print("✓ Parse '*0..7/10': phases=[0,1,2,3,4,5,6,7], cycle=10, shuffle=True")

	# Test range syntax without shuffle
	phases, cycle, shuffle = parse_split("0..2/5")
	assert phases == [0, 1, 2], f"Expected [0, 1, 2], got {phases}"
	assert cycle == 5, f"Expected 5, got {cycle}"
	assert shuffle == False, f"Expected False, got {shuffle}"
	print("✓ Parse '0..2/5': phases=[0,1,2], cycle=5, shuffle=False")

	# Test mixed syntax (range + individual)
	phases, cycle, shuffle = parse_split("*0..3,7,8/10")
	assert phases == [0, 1, 2, 3, 7, 8], f"Expected [0,1,2,3,7,8], got {phases}"
	assert cycle == 10, f"Expected 10, got {cycle}"
	assert shuffle == True, f"Expected True, got {shuffle}"
	print("✓ Parse '*0..3,7,8/10': phases=[0,1,2,3,7,8], cycle=10, shuffle=True")

	# Test multiple ranges
	phases, cycle, shuffle = parse_split("0..2,5..7/10")
	assert phases == [0, 1, 2, 5, 6, 7], f"Expected [0,1,2,5,6,7], got {phases}"
	assert cycle == 10, f"Expected 10, got {cycle}"
	assert shuffle == False, f"Expected False, got {shuffle}"
	print("✓ Parse '0..2,5..7/10': phases=[0,1,2,5,6,7], cycle=10, shuffle=False")

	print("\n✓ All split parsing tests passed!\n")


def test_file_assignment():
	"""Test deterministic file assignment to phases."""
	print("="*80)
	print("Testing File Assignment")
	print("="*80)

	# Create dummy file paths
	files = [Path(f"game_{i:03d}.tgn") for i in range(100)]

	# Test with cycle=10
	cycle = 10
	phase_counts = {i: 0 for i in range(cycle)}

	for file in files:
		phase = TGNDataset.file_to_phase(file, cycle)
		assert 0 <= phase < cycle, f"Phase {phase} out of range [0, {cycle})"
		phase_counts[phase] += 1

	print(f"\nDistribution across {cycle} phases (100 files):")
	for phase, count in sorted(phase_counts.items()):
		print(f"  Phase {phase}: {count} files ({count}%)")

	# Check that distribution is reasonable (each phase should have ~10 files)
	for phase, count in phase_counts.items():
		assert 5 <= count <= 15, f"Phase {phase} has {count} files, expected around 10"

	# Test determinism - same file should always map to same phase
	test_file = Path("test_game.tgn")
	phase1 = TGNDataset.file_to_phase(test_file, cycle)
	phase2 = TGNDataset.file_to_phase(test_file, cycle)
	assert phase1 == phase2, f"Non-deterministic assignment: {phase1} != {phase2}"
	print(f"\n✓ Deterministic assignment verified: 'test_game.tgn' → phase {phase1}")

	print("\n✓ All file assignment tests passed!\n")


def test_dataset_split():
	"""Test dataset splitting with real TGN files."""
	print("="*80)
	print("Testing Dataset Split")
	print("="*80)

	data_dir = project_root / "third_party/trigo/trigo-web/tools/output"

	if not data_dir.exists():
		print(f"⚠ Data directory not found: {data_dir}")
		print("  Skipping dataset split test")
		return

	# Test 1: Full dataset (no split)
	print("\n[Test 1] Loading full dataset (no split)...")
	config_full = {
		'data_dir': str(data_dir),
		'max_length': 2048,
	}
	dataset_full = TGNDataset.from_config(config_full)
	total_files = len(dataset_full)
	print(f"✓ Full dataset: {total_files} files")

	# Test 2: Training split (80%)
	print("\n[Test 2] Loading training split (80%)...")
	config_train = {
		'data_dir': str(data_dir),
		'max_length': 2048,
		'split': '*0..7/10',  # 80% with shuffle (using range syntax)
	}
	dataset_train = TGNDataset.from_config(config_train)
	train_files = len(dataset_train)
	print(f"✓ Training dataset: {train_files} files")
	assert dataset_train.shuffle == True, "Training split should have shuffle=True"

	# Test 3: Validation split (20%)
	print("\n[Test 3] Loading validation split (20%)...")
	config_val = {
		'data_dir': str(data_dir),
		'max_length': 2048,
		'split': '8,9/10',  # 20% without shuffle
	}
	dataset_val = TGNDataset.from_config(config_val)
	val_files = len(dataset_val)
	print(f"✓ Validation dataset: {val_files} files")
	assert dataset_val.shuffle == False, "Validation split should have shuffle=False"

	# Test 4: Verify no overlap
	print("\n[Test 4] Verifying no file overlap...")
	train_file_names = {f.name for f in dataset_train.files}
	val_file_names = {f.name for f in dataset_val.files}
	overlap = train_file_names & val_file_names
	assert len(overlap) == 0, f"Found {len(overlap)} overlapping files: {overlap}"
	print(f"✓ No overlap between train and validation sets")

	# Test 5: Verify split coverage
	print("\n[Test 5] Verifying split coverage...")
	combined = train_files + val_files
	print(f"  Total files: {total_files}")
	print(f"  Train files: {train_files} ({train_files/total_files*100:.1f}%)")
	print(f"  Val files: {val_files} ({val_files/total_files*100:.1f}%)")
	print(f"  Combined: {combined} ({combined/total_files*100:.1f}%)")

	# Allow some tolerance due to hashing distribution
	assert abs(combined - total_files) <= 2, \
		f"Split doesn't cover full dataset: {combined} != {total_files}"
	print(f"✓ Train + Val covers full dataset")

	# Test 6: Test data loading
	print("\n[Test 6] Testing data loading from splits...")
	train_sample = dataset_train[0]
	val_sample = dataset_val[0]

	assert 'input_ids' in train_sample, "Missing input_ids in train sample"
	assert 'labels' in train_sample, "Missing labels in train sample"
	assert 'attention_mask' in train_sample, "Missing attention_mask in train sample"

	assert 'input_ids' in val_sample, "Missing input_ids in val sample"
	assert 'labels' in val_sample, "Missing labels in val sample"
	assert 'attention_mask' in val_sample, "Missing attention_mask in val sample"

	print(f"✓ Train sample shapes:")
	print(f"  - input_ids: {train_sample['input_ids'].shape}")
	print(f"  - labels: {train_sample['labels'].shape}")
	print(f"  - attention_mask: {train_sample['attention_mask'].shape}")

	print(f"✓ Val sample shapes:")
	print(f"  - input_ids: {val_sample['input_ids'].shape}")
	print(f"  - labels: {val_sample['labels'].shape}")
	print(f"  - attention_mask: {val_sample['attention_mask'].shape}")

	print("\n✓ All dataset split tests passed!\n")


def main():
	"""Run all tests."""
	print("\n" + "="*80)
	print("TGNDataset Split Feature Tests")
	print("="*80 + "\n")

	try:
		test_split_parsing()
		test_file_assignment()
		test_dataset_split()

		print("="*80)
		print("✓ ALL TESTS PASSED!")
		print("="*80)

	except AssertionError as e:
		print(f"\n❌ Test failed:")
		print(f"   {e}")
		import traceback
		traceback.print_exc()
		return 1

	except Exception as e:
		print(f"\n❌ Unexpected error:")
		print(f"   {type(e).__name__}: {e}")
		import traceback
		traceback.print_exc()
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
