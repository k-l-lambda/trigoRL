# train_lm.py Positional Config Argument - Implementation Summary

## Date
2025-11-13

## Overview
Modified `train_lm.py` to accept a positional argument for specifying the configuration file, making it more convenient and intuitive to use.

## Changes Made

### 1. train_lm.py

**Added `parse_positional_config()` function** (lines 44-82):
```python
def parse_positional_config():
    """
    Parse positional argument as config name/path.

    Supports:
      - Short name: trigo-gpt2
      - Relative path: configs/training/trigo-gpt2.yaml
      - Absolute path: /path/to/config.yaml

    Converts to Hydra's --config-name format.
    """
    # Check if first argument is a positional config (not a Hydra override)
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]

        # Skip if it's already a Hydra parameter
        if first_arg.startswith('-') or '=' in first_arg:
            return

        # Parse the positional argument
        config_path = Path(first_arg)

        # Case 1: Full path to config file
        if config_path.suffix in ['.yaml', '.yml']:
            config_name = config_path.stem
            sys.argv[1] = f'--config-name={config_name}'

        # Case 2: Short name (e.g., trigo-gpt2)
        else:
            config_name = first_arg
            sys.argv[1] = f'--config-name={config_name}'
```

**Updated `__main__` block** (line 246-247):
```python
if __name__ == "__main__":
    # Parse positional config argument before Hydra processes sys.argv
    parse_positional_config()

    try:
        main()
    # ...
```

**Updated docstring** (lines 1-11):
```python
"""
Training script for attention-based language models.

Usage:
    python train_lm.py                                   # Use default config (trigo-gpt2)
    python train_lm.py trigo-llama                      # Use specific config (short name)
    python train_lm.py configs/training/trigo-rwkv.yaml # Use config file path
    python train_lm.py trigo-gpt2 training.epochs=50    # Config + overrides
    python train_lm.py --config-name=trigo-rwkv          # Alternative syntax
"""
```

### 2. commands.local.sh

Updated example command from:
```bash
python -m train_lm configs/training/trigo-gpt2.yaml
```

To:
```bash
python train_lm.py trigo-gpt2

# Or with full path:
# python train_lm.py configs/training/trigo-gpt2.yaml

# Or with overrides:
# python train_lm.py trigo-gpt2 training.epochs=10 data.loader.batch_size=4

# Other configs:
# python train_lm.py trigo-llama
# python train_lm.py trigo-gpt2-invsqrt
```

### 3. Documentation

**Created:**
- `docs/train_lm_positional_arg.md` - Complete feature documentation
- `tests/demo_positional_config.py` - Demonstration script
- `docs/train_lm_positional_arg_summary.md` - This summary

## Usage Examples

### Before (Hydra standard syntax)
```bash
python train_lm.py --config-name=trigo-gpt2
python train_lm.py --config-name=trigo-llama training.learning_rate=1e-4
```

### After (with positional argument)
```bash
python train_lm.py trigo-gpt2
python train_lm.py trigo-llama training.learning_rate=1e-4
```

### All Supported Formats

1. **Short name** (recommended):
   ```bash
   python train_lm.py trigo-gpt2
   ```

2. **Full path**:
   ```bash
   python train_lm.py configs/training/trigo-gpt2.yaml
   ```

3. **With overrides**:
   ```bash
   python train_lm.py trigo-gpt2 training.epochs=10 data.loader.batch_size=4
   ```

4. **Default config**:
   ```bash
   python train_lm.py
   ```

5. **Hydra syntax** (still works):
   ```bash
   python train_lm.py --config-name=trigo-gpt2
   ```

## How It Works

1. **Before Hydra runs**: The script checks `sys.argv[1]`
2. **Identifies positional config**: If it's not a Hydra parameter (doesn't start with `-` and doesn't contain `=`)
3. **Parses the argument**:
   - If it has `.yaml` or `.yml` extension → Extract config name from filename
   - Otherwise → Use as-is (short name)
4. **Converts to Hydra format**: Replaces `sys.argv[1]` with `--config-name=<name>`
5. **Hydra processes**: Hydra then handles the modified arguments normally

## Argument Parsing Table

| Input | Detected As | Converted To | Config Loaded |
|-------|-------------|--------------|---------------|
| `trigo-gpt2` | Short name | `--config-name=trigo-gpt2` | `trigo-gpt2.yaml` |
| `configs/training/trigo-llama.yaml` | Path | `--config-name=trigo-llama` | `trigo-llama.yaml` |
| `trigo-gpt2 training.epochs=10` | Short name + override | `--config-name=trigo-gpt2 training.epochs=10` | `trigo-gpt2.yaml` + override |
| `--config-name=trigo-gpt2` | Hydra syntax | (no change) | `trigo-gpt2.yaml` |
| `training.epochs=10` | Override only | (no change) | default + override |
| (empty) | Default | (no change) | default |

## Verification

Tested with:
```bash
python tests/demo_positional_config.py
```

Output:
```
Input:  trigo-gpt2
Parsed: trigo-gpt2 (from short name)
Will load: configs/training/trigo-gpt2.yaml

Input:  configs/training/trigo-gpt2.yaml
Parsed: trigo-gpt2 (from path name)
Will load: configs/training/trigo-gpt2.yaml

Input:  trigo-gpt2 training.epochs=50
Parsed: trigo-gpt2 (from short name)
Will load: configs/training/trigo-gpt2.yaml

Input:  --config-name=trigo-gpt2
Parsed: (Hydra syntax - no change)

✓ All parsing tests passed
```

Manual verification:
```bash
$ python train_lm.py trigo-gpt2-invsqrt 2>&1 | head -20
# ✓ Loads trigo-gpt2-invsqrt.yaml correctly

$ python train_lm.py configs/training/trigo-llama.yaml 2>&1 | head -20
# ✓ Loads trigo-llama.yaml correctly
```

## Benefits

1. **More concise**:
   - Before: `python train_lm.py --config-name=trigo-gpt2`
   - After: `python train_lm.py trigo-gpt2`

2. **More intuitive**: Positional argument is natural for the primary input

3. **Backward compatible**: All existing Hydra syntax still works

4. **Flexible**: Supports both short names and full paths

5. **Shell-friendly**: Easy to use in scripts and aliases

## Shell Integration

Easy to create convenience functions:
```bash
# In commands.local.sh or ~/.bashrc
train_gpt2() {
    python train_lm.py trigo-gpt2 "$@"
}

train_llama() {
    python train_lm.py trigo-llama "$@"
}

# Usage:
# train_gpt2 training.epochs=50
# train_llama training.learning_rate=1e-4
```

## Implementation Notes

### Why Modify sys.argv?

This approach was chosen because:
1. **Minimal changes**: Only one function added, no changes to Hydra decorator
2. **Transparent**: Hydra sees standard arguments, no special handling needed
3. **Compatible**: Works with all Hydra features (overrides, multirun, etc.)
4. **Early**: Runs before Hydra processes arguments

### Alternative Approaches Considered

1. **Using Hydra Compose API**: Too invasive, requires rewriting main()
2. **Custom argument parser**: Would conflict with Hydra's parser
3. **Wrapper script**: Extra file, less convenient

### Edge Cases Handled

- ✓ Positional arg with overrides: `trigo-gpt2 training.epochs=50`
- ✓ Hydra flags unchanged: `--config-name=trigo-gpt2`
- ✓ Override-only unchanged: `training.epochs=50`
- ✓ Empty args use default
- ✓ Path extraction works for any valid path

### Known Limitations

1. **Config search path**: Full paths are converted to just the config name, so the file must be in `configs/training/`
2. **No validation**: Config existence is checked by Hydra later, not immediately
3. **No auto-complete**: Shell won't auto-complete config names (could add in future)

## Files Modified

1. `train_lm.py` - Added positional config parsing
2. `commands.local.sh` - Updated example commands
3. `docs/train_lm_positional_arg.md` - Feature documentation
4. `tests/demo_positional_config.py` - Demonstration script
5. `docs/train_lm_positional_arg_summary.md` - This summary

## Testing

Tested scenarios:
- ✓ Short config name
- ✓ Full config path (relative)
- ✓ Positional arg + overrides
- ✓ Hydra syntax (backward compat)
- ✓ Override-only syntax
- ✓ No args (default config)

All parsing logic verified with demo script.

## Future Enhancements

Potential improvements:
1. Support for config directories (not just single files)
2. Shell auto-completion script
3. `--list-configs` flag to show available configs
4. Fuzzy matching for typos in config names
5. Early validation (check config exists before Hydra processes)
6. Support for multiple config paths (not just default)

## User Request

Original request: "改造train_lm，使得它接受一个无命名参数，即配置文件路径"

Translation: "Modify train_lm to accept an unnamed parameter, i.e., config file path"

✓ Request satisfied:
- Accepts positional (unnamed) parameter
- Supports both config names and paths
- Backward compatible with existing syntax
- More convenient to use

## Status

✓ Implementation complete
✓ Parsing logic tested
✓ Manual verification done
✓ Documentation complete
✓ Commands updated
✓ Ready for use

## Example Session

```bash
# Train with default config
$ python train_lm.py
# Uses trigo-gpt2.yaml

# Train with specific config
$ python train_lm.py trigo-llama
# Uses trigo-llama.yaml

# With overrides
$ python train_lm.py trigo-gpt2 training.epochs=10
# Uses trigo-gpt2.yaml, overrides epochs

# Full path
$ python train_lm.py configs/training/trigo-gpt2-invsqrt.yaml
# Uses trigo-gpt2-invsqrt.yaml

# Old syntax still works
$ python train_lm.py --config-name=trigo-gpt2
# Uses trigo-gpt2.yaml
```
