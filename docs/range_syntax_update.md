# Range Syntax Support for Split Parameter

## Update Summary

Added `..` range syntax support to TGNDataset split parameter for more concise phase specification.

## Changes

### Before
```yaml
train_split: "*0,1,2,3,4,5,6,7/10"  # 80% training
```

### After (More Concise)
```yaml
train_split: "*0..7/10"  # 80% training (using range syntax)
```

## Syntax

The split parameter now supports three formats:

1. **Comma-separated**: `"0,1,2,3/10"`
2. **Range with ..**: `"0..3/10"` (equivalent to `"0,1,2,3/10"`)
3. **Mixed**: `"0..3,7,8/10"` or `"0..2,5..7/10"`

## Examples

| Format | Expands to | Meaning |
|--------|-----------|---------|
| `*0..7/10` | `*0,1,2,3,4,5,6,7/10` | 80% training, shuffled |
| `8,9/10` | `8,9/10` | 20% validation |
| `*0..2/5` | `*0,1,2/5` | 60% training, shuffled |
| `*0..3,7..9/10` | `*0,1,2,3,7,8,9/10` | Phases 0-3 and 7-9 |
| `0..2,5..7/10` | `0,1,2,5,6,7/10` | Multiple ranges |

## Implementation

Updated `TGNDataset.parse_split()` method to:
1. Split by commas
2. For each part, check if it's a range (`\d+\.\.\d+`)
3. If range, expand to list of integers
4. If not, treat as single integer

## Tests

Added comprehensive tests in `tests/test_dataset_split.py`:

```python
✓ Parse '*0..7/10': phases=[0,1,2,3,4,5,6,7], cycle=10, shuffle=True
✓ Parse '0..2/5': phases=[0,1,2], cycle=5, shuffle=False
✓ Parse '*0..3,7,8/10': phases=[0,1,2,3,7,8], cycle=10, shuffle=True
✓ Parse '0..2,5..7/10': phases=[0,1,2,5,6,7], cycle=10, shuffle=False
```

All tests pass (including 4 new range syntax tests).

## Updated Files

- **trigor/data/tgn_dataset.py**: Added range parsing logic
- **configs/training/*.yaml**: Updated to use `*0..7/10` syntax
- **tests/test_dataset_split.py**: Added 4 new test cases
- **examples/dataset_split_example.py**: Updated examples

## Backward Compatibility

✅ Fully backward compatible - comma-separated syntax still works:
- `"0,1,2,3/5"` → Works as before
- `"0..3/5"` → New range syntax (same result)

## Usage

```python
from trigor.data import TGNDataset

# Old way (still works)
train_dataset = TGNDataset.from_config({
    'data_dir': 'data',
    'split': '*0,1,2,3,4,5,6,7/10',
})

# New way (more concise)
train_dataset = TGNDataset.from_config({
    'data_dir': 'data',
    'split': '*0..7/10',  # Equivalent to above
})
```

## Benefits

1. **More concise**: `*0..7/10` vs `*0,1,2,3,4,5,6,7/10`
2. **Easier to read**: Clear intent for contiguous ranges
3. **Less error-prone**: No risk of typos in long sequences
4. **Flexible**: Can mix ranges and individual phases

## Verification

All tests pass:
- ✓ `test_dataset_split.py`: 7 parsing tests (including 4 new range tests)
- ✓ `test_configs_with_split.py`: All 3 configs load correctly
- ✓ Examples run successfully with new syntax
