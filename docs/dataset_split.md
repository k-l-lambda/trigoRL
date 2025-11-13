# TGNDataset Train/Validation Split

## Overview

TGNDataset now supports deterministic train/validation splitting using a phase-based approach similar to the deep-starry project. This allows you to split your data without physically moving files.

## Split Format

The split parameter uses a simple string format:

```
[*]phase1,phase2,phase3,.../total_phases
```

- **phases**: Comma-separated list of phase indices to include
- **total_phases**: Total number of phases (determines granularity)
- **\***: Optional prefix to enable shuffling

### Examples

```python
# 80% training set (shuffled)
split = "*0,1,2,3,4,5,6,7/10"

# 20% validation set (not shuffled)
split = "8,9/10"

# 60/20/20 split for train/val/test
train_split = "*0,1,2/5"  # 60%
val_split = "3/5"          # 20%
test_split = "4/5"         # 20%
```

## How It Works

1. **Deterministic Hashing**: Each file is assigned to a phase based on MD5 hash of its filename
2. **Phase Assignment**: Files are distributed evenly across phases (0 to cycle-1)
3. **Phase Selection**: Only files in specified phases are included in the dataset
4. **No Overlap**: Different phases always contain different files

### Phase Distribution

With 100 files and cycle=10, each phase gets approximately 10 files:

```
Phase 0: ~10 files (0-9%)
Phase 1: ~10 files (10-19%)
Phase 2: ~10 files (20-29%)
...
Phase 9: ~10 files (90-99%)
```

## Usage

### Method 1: Direct Configuration

```python
from trigor.data import TGNDataset

# Training set
train_config = {
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '*0,1,2,3,4,5,6,7/10',  # 80% with shuffle
}
train_dataset = TGNDataset.from_config(train_config)

# Validation set
val_config = {
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '8,9/10',  # 20% without shuffle
}
val_dataset = TGNDataset.from_config(val_config)
```

### Method 2: From YAML Config

**Config file (`configs/training/trigo-gpt2.yaml`):**

```yaml
data:
  type: TGNDataset
  data_dir: ${paths.root}/data/tgn_games
  max_length: 2048

  # Split specifications
  train_split: "*0,1,2,3,4,5,6,7/10"  # 80% training
  val_split: "8,9/10"                  # 20% validation
```

**Python code:**

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

### Method 3: No Split (Use All Files)

```python
# Omit 'split' parameter to use all files
config = {
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    # No split = use all files
}
full_dataset = TGNDataset.from_config(config)
```

## Properties

### Deterministic
- Same files always map to same phases
- Reproducible across runs
- Based on filename hash (independent of file order)

### No Overlap
- Each file belongs to exactly one phase
- Train and validation sets are completely separate

### Balanced Distribution
- Files distributed approximately evenly across phases
- Small variations due to hashing (±20% is normal)

## Shuffling

The `*` prefix controls shuffling behavior:

- **With `*`**: Files are shuffled using a deterministic seed (based on split string)
- **Without `*`**: Files maintain sorted order

```python
# Training: shuffled for better convergence
train_split = "*0,1,2,3,4,5,6,7/10"

# Validation: not shuffled for consistent evaluation
val_split = "8,9/10"
```

**Note**: Even with shuffling enabled via `*`, the shuffle is deterministic and reproducible.

## Split Ratios

### Common Ratios

| Split | Train | Val | Test | Format |
|-------|-------|-----|------|--------|
| 80/20 | 80% | 20% | - | `*0-7/10`, `8,9/10` |
| 70/30 | 70% | 30% | - | `*0-6/10`, `7,8,9/10` |
| 60/20/20 | 60% | 20% | 20% | `*0,1,2/5`, `3/5`, `4/5` |
| 90/10 | 90% | 10% | - | `*0-8/10`, `9/10` |

### Custom Ratios

For more granular control, use larger cycle values:

```python
# 85/15 split using cycle=20
train_split = "*0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16/20"  # 85%
val_split = "17,18,19/20"                                      # 15%
```

## API Reference

### `TGNDataset.__init__`

```python
def __init__(
    self,
    data_dir: str,
    tokenizer: TGNByteTokenizer,
    max_length: int = 8192,
    min_length: int = 10,
    max_file_size: int = 10000,
    split: Optional[str] = None,  # NEW: Split specification
    filter_fn: Optional[Callable[[Path], bool]] = None,
):
```

**New Parameter:**
- `split` (Optional[str]): Phase-based split specification
  - Format: `"[*]phases/cycle"`
  - Example: `"*0,1,2/5"` for 60% training with shuffle
  - Default: `None` (use all files)

### `TGNDataset.parse_split`

```python
@staticmethod
def parse_split(split: str) -> Tuple[List[int], int, bool]:
    """
    Parse split string into components.

    Args:
        split: Split specification (e.g., "*0,1,2/5")

    Returns:
        Tuple of (phases, cycle, shuffle)
        - phases: List of phase indices
        - cycle: Total number of phases
        - shuffle: Whether to shuffle
    """
```

### `TGNDataset.file_to_phase`

```python
@staticmethod
def file_to_phase(file_path: Path, cycle: int) -> int:
    """
    Deterministically assign file to phase.

    Args:
        file_path: Path to the file
        cycle: Total number of phases

    Returns:
        Phase index (0 to cycle-1)
    """
```

## Examples

### Example 1: Basic 80/20 Split

```python
from trigor.data import TGNDataset
from torch.utils.data import DataLoader

# Training dataset (80%)
train_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '*0,1,2,3,4,5,6,7/10',
})

# Validation dataset (20%)
val_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '8,9/10',
})

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=TGNDataset.collate_batch)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=TGNDataset.collate_batch)
```

### Example 2: Three-Way Split (60/20/20)

```python
datasets = {}
for name, split in [
    ('train', '*0,1,2/5'),
    ('val', '3/5'),
    ('test', '4/5'),
]:
    datasets[name] = TGNDataset.from_config({
        'data_dir': 'data/tgn_games',
        'max_length': 2048,
        'split': split,
    })

print(f"Train: {len(datasets['train'])} files")
print(f"Val: {len(datasets['val'])} files")
print(f"Test: {len(datasets['test'])} files")
```

## Testing

Run the split functionality tests:

```bash
python tests/test_dataset_split.py
```

Run example demonstrations:

```bash
python examples/dataset_split_example.py
```

## Migration from No-Split

If you have existing code without splits:

**Before:**
```python
dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
})
```

**After:**
```python
# Training set
train_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '*0,1,2,3,4,5,6,7/10',  # Add this
})

# Validation set
val_dataset = TGNDataset.from_config({
    'data_dir': 'data/tgn_games',
    'max_length': 2048,
    'split': '8,9/10',  # Add this
})
```

## Comparison with deep-starry

This implementation is inspired by deep-starry's split mechanism with some differences:

### Similarities
- Phase-based deterministic splitting
- `*` prefix for shuffle control
- `phases/cycle` string format
- Hash-based file assignment

### Differences
- **Single dataset class**: TGNDataset handles splits via parameter (deep-starry uses separate Dataset classes)
- **Simpler API**: Split specified as string parameter instead of at load time
- **File-level hashing**: Uses filename hash (deep-starry may use group-based hashing)

## Best Practices

1. **Use larger cycles for precise ratios**: `cycle=20` for 5% increments, `cycle=100` for 1% increments

2. **Shuffle training, not validation**: Use `*` for training split only
   ```python
   train_split = "*0-7/10"  # Shuffled
   val_split = "8,9/10"     # Not shuffled
   ```

3. **Verify split coverage**: Check that train + val files sum to total
   ```python
   total = len(full_dataset)
   combined = len(train_dataset) + len(val_dataset)
   assert abs(combined - total) <= cycle * 0.1  # Allow 10% tolerance per phase
   ```

4. **Use consistent cycle across splits**: Don't mix different cycles for same dataset
   ```python
   # Good: Same cycle (10) for both
   train_split = "*0-7/10"
   val_split = "8,9/10"

   # Bad: Different cycles
   train_split = "*0-7/10"
   val_split = "4/5"  # Don't do this
   ```

## Troubleshooting

### Issue: Uneven split sizes

**Cause**: Hash distribution may not be perfectly uniform

**Solution**: This is expected. With 100 files and cycle=10, expect 8-12 files per phase.

### Issue: Split doesn't cover all files

**Cause**: Missing phases in specification

**Solution**: Verify phases sum to full cycle:
```python
# Bad: Only uses phases 0-8 (missing phase 9)
split = "0,1,2,3,4,5,6,7,8/10"

# Good: Uses all phases 0-9
train_split = "0,1,2,3,4,5,6,7,8/10"
val_split = "9/10"
```

### Issue: Overlapping files in train/val

**Cause**: Using same phases in multiple splits

**Solution**: Use disjoint phase sets:
```python
# Good: Disjoint phases
train = "0-7/10"
val = "8,9/10"

# Bad: Overlapping phases
train = "0-8/10"
val = "7,8,9/10"  # Phase 7,8 overlap!
```
