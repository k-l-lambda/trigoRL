# TGNDataset Split Feature - Implementation Summary

## Overview

Added train/validation split functionality to TGNDataset, inspired by deep-starry's phase-based splitting mechanism. This allows deterministic, reproducible data splitting without physically moving files.

## Changes Made

### 1. Core Implementation (`trigor/data/tgn_dataset.py`)

#### Added Imports
```python
import hashlib
from typing import Tuple
```

#### New Methods

**`parse_split(split: str) -> Tuple[List[int], int, bool]`**
- Static method to parse split string format
- Returns: (phases, cycle, shuffle)
- Example: `"*0,1,2/5"` → `([0, 1, 2], 5, True)`

**`file_to_phase(file_path: Path, cycle: int) -> int`**
- Static method for deterministic file-to-phase assignment
- Uses MD5 hash of filename for determinism
- Returns phase index (0 to cycle-1)

#### Modified Methods

**`__init__(..., split: Optional[str] = None, ...)`**
- Added `split` parameter
- Parses split specification
- Filters files based on phase assignment
- Applies deterministic shuffling if requested
- Enhanced logging to show split information

**`from_config(...)`**
- Added support for `split` parameter in config dict

### 2. Configuration Files

Updated all training configs with split specifications:

**`configs/training/trigo-gpt2.yaml`**
**`configs/training/trigo-llama.yaml`**
**`configs/training/trigo-rwkv.yaml`**

Added:
```yaml
data:
  # ... existing config ...

  # Train/validation split
  train_split: "*0,1,2,3,4,5,6,7/10"  # 80% training set with shuffle
  val_split: "8,9/10"                  # 20% validation set without shuffle
```

### 3. Test Suite

**`tests/test_dataset_split.py`**
- Tests split string parsing
- Tests file-to-phase assignment
- Tests determinism and distribution
- Tests train/val dataset creation
- Verifies no overlap between splits
- Tests data loading from splits

Results: ✓ All tests passed

### 4. Examples

**`examples/dataset_split_example.py`**
- Example 1: Basic 80/20 split
- Example 2: Load from YAML config
- Example 3: Custom split ratios (60/20/20)
- Example 4: No split (full dataset)

### 5. Documentation

**`docs/dataset_split.md`**
- Comprehensive split feature documentation
- Format specification and examples
- Usage patterns and best practices
- API reference
- Troubleshooting guide

## Split Format

```
[*]phase1,phase2,phase3,.../total_phases
```

- **phases**: Comma-separated list of phase indices to include
- **total_phases**: Total number of phases (e.g., 10 for 10% increments)
- **\***: Optional prefix to enable shuffling

### Examples

| Split | Meaning |
|-------|---------|
| `*0,1,2,3,4,5,6,7/10` | 80% training, shuffled |
| `8,9/10` | 20% validation, not shuffled |
| `*0,1,2/5` | 60% training, shuffled |
| `3/5` | 20% validation |
| `4/5` | 20% test |

## Key Features

### 1. Deterministic
- Same files always map to same phases
- Based on MD5 hash of filename
- Reproducible across runs and machines

### 2. No Overlap
- Each file belongs to exactly one phase
- Train and validation sets are completely separate
- Phase assignments don't change over time

### 3. Balanced Distribution
- Files distributed approximately evenly across phases
- Natural variation due to hashing (±20% per phase is normal)

### 4. Configurable Shuffling
- `*` prefix enables deterministic shuffling
- Shuffle seed derived from split string
- Reproducible shuffle order

### 5. Flexible Ratios
- Any ratio possible by choosing phases
- Common: 80/20, 70/30, 60/20/20
- Adjust granularity with cycle size

## Usage Examples

### Basic Usage

```python
from trigor.data import TGNDataset

# Training set (80%)
train_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '*0,1,2,3,4,5,6,7/10',
})

# Validation set (20%)
val_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '8,9/10',
})
```

### From Config

```python
from omegaconf import OmegaConf
from trigor.data import TGNDataset

cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')

# Training dataset
train_config = OmegaConf.create({**cfg.data, 'split': cfg.data.train_split})
train_dataset = TGNDataset.from_config(train_config)

# Validation dataset
val_config = OmegaConf.create({**cfg.data, 'split': cfg.data.val_split})
val_dataset = TGNDataset.from_config(val_config)
```

## Test Results

Running `tests/test_dataset_split.py` with 100 TGN files:

```
✓ Split parsing: 3/3 tests passed
✓ File assignment: Deterministic, balanced distribution
✓ Dataset split:
  - Full dataset: 100 files
  - Training (80%): 76 files
  - Validation (20%): 24 files
  - No overlap: 0 files in common
  - Coverage: 100% of files included
  - Data loading: All samples load correctly
```

## Comparison with deep-starry

### Similarities
- Phase-based deterministic splitting
- `*` prefix for shuffle control
- `phases/cycle` string format
- Hash-based file assignment

### Differences

| Aspect | trigoRL (TGNDataset) | deep-starry |
|--------|----------------------|-------------|
| API | Single dataset class with `split` parameter | Separate `Dataset` classes per split |
| Config | `split` in dataset config | `splits` passed at load time |
| Hashing | MD5 of filename | May use group-based hashing |
| Shuffle | Deterministic shuffle with fixed seed | Shuffle controlled by `*` prefix |
| Usage | More straightforward for single datasets | More flexible for complex scenarios |

## Integration

### Backward Compatibility

The `split` parameter is optional. Existing code without splits continues to work:

```python
# Old code (no split) - still works
dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
})
# Uses all files
```

### Migration Path

To add splits to existing code:

1. Add split specifications to config:
```yaml
data:
  # ... existing config ...
  train_split: "*0,1,2,3,4,5,6,7/10"
  val_split: "8,9/10"
```

2. Create separate datasets for train/val:
```python
train_dataset = TGNDataset.from_config({**cfg.data, 'split': cfg.data.train_split})
val_dataset = TGNDataset.from_config({**cfg.data, 'split': cfg.data.val_split})
```

## Files Changed

```
Modified:
  trigor/data/tgn_dataset.py           (Added split functionality)
  configs/training/trigo-gpt2.yaml     (Added split config)
  configs/training/trigo-llama.yaml    (Added split config)
  configs/training/trigo-rwkv.yaml     (Added split config)

Added:
  tests/test_dataset_split.py          (Test suite)
  examples/dataset_split_example.py    (Usage examples)
  docs/dataset_split.md                (Documentation)
  docs/dataset_split_summary.md        (This file)
```

## Next Steps

To use splits in training:

1. **Update training script** to create separate train/val datasets
2. **Create separate dataloaders** for train and validation
3. **Add validation loop** to training script
4. **Log split information** for reproducibility
5. **Consider adding test split** for final evaluation (e.g., `"9/10"` for 10% test)

Example training loop structure:
```python
# Create datasets
train_dataset = TGNDataset.from_config({**cfg.data, 'split': cfg.data.train_split})
val_dataset = TGNDataset.from_config({**cfg.data, 'split': cfg.data.val_split})

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=TGNDataset.collate_batch)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=TGNDataset.collate_batch)

# Training loop
for epoch in range(num_epochs):
    # Train
    model.train()
    for batch in train_loader:
        # ... training step ...

    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # ... validation step ...
```

## Verification

Run all tests:
```bash
# Test split functionality
python tests/test_dataset_split.py

# Run examples
python examples/dataset_split_example.py

# Test with configs
python tests/test_updated_configs.py
```

All tests should pass with expected output showing proper train/val splits.
