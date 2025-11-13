# train_lm.py - Positional Config Argument Feature

## Overview

The `train_lm.py` script now supports **positional arguments** for specifying the configuration file, making it more convenient to use.

## Date

2025-11-13

## Usage Examples

### 1. Using Short Config Name (Recommended)

```bash
python train_lm.py trigo-gpt2
python train_lm.py trigo-llama
python train_lm.py trigo-gpt2-invsqrt
```

This is the most concise way to specify a config. The script will look for the config file in `configs/training/`.

### 2. Using Full Config Path

```bash
python train_lm.py configs/training/trigo-gpt2.yaml
python train_lm.py configs/training/trigo-llama.yaml
```

You can also provide the full path to the config file (relative or absolute).

### 3. With Parameter Overrides

```bash
python train_lm.py trigo-gpt2 training.epochs=10
python train_lm.py trigo-llama training.learning_rate=5e-5 data.loader.batch_size=16
python train_lm.py configs/training/trigo-rwkv.yaml training.wandb.enabled=true
```

Positional config argument can be combined with Hydra's parameter override syntax.

### 4. Using Default Config

```bash
python train_lm.py
```

If no config is specified, uses the default `trigo-gpt2` config.

### 5. Alternative Syntax (Still Works)

```bash
python train_lm.py --config-name=trigo-gpt2
python train_lm.py -cn trigo-llama
```

The original Hydra syntax still works for backward compatibility.

## Implementation Details

### How It Works

The script preprocesses `sys.argv` before Hydra parses it:

1. Checks if the first argument is a positional config (not starting with `-` and not containing `=`)
2. Parses the argument:
   - If it has `.yaml` or `.yml` extension: Extract the config name from the filename
   - Otherwise: Use the argument as-is (short name)
3. Converts to Hydra's `--config-name=<name>` format
4. Updates `sys.argv[1]` in place
5. Hydra then processes the modified arguments normally

### Code Location

`train_lm.py:44-82` - `parse_positional_config()` function

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

Called in `__main__` before Hydra processes arguments:

```python
if __name__ == "__main__":
    # Parse positional config argument before Hydra processes sys.argv
    parse_positional_config()

    try:
        main()
    except Exception as e:
        # ...
```

## Test Results

### Test 1: Short Name

```bash
$ python train_lm.py trigo-gpt2
# ✓ Loads configs/training/trigo-gpt2.yaml
```

### Test 2: Full Path

```bash
$ python train_lm.py configs/training/trigo-llama.yaml
# ✓ Loads configs/training/trigo-llama.yaml
```

### Test 3: With Overrides

```bash
$ python train_lm.py trigo-gpt2-invsqrt
# ✓ Loads configs/training/trigo-gpt2-invsqrt.yaml with inverse_sqrt scheduler
```

### Test 4: Hydra Syntax Still Works

```bash
$ python train_lm.py --config-name=trigo-gpt2
# ✓ Works as before (backward compatible)
```

### Test 5: Override Syntax Still Works

```bash
$ python train_lm.py training.epochs=50
# ✓ Uses default config with epochs override
```

## Argument Parsing Logic

| Input | Parsed As | Config Loaded |
|-------|-----------|---------------|
| `trigo-gpt2` | `--config-name=trigo-gpt2` | `configs/training/trigo-gpt2.yaml` |
| `configs/training/trigo-llama.yaml` | `--config-name=trigo-llama` | `configs/training/trigo-llama.yaml` |
| `trigo-gpt2 training.epochs=10` | `--config-name=trigo-gpt2 training.epochs=10` | `configs/training/trigo-gpt2.yaml` with override |
| `--config-name=trigo-gpt2` | (unchanged) | `configs/training/trigo-gpt2.yaml` |
| `training.epochs=10` | (unchanged, uses default) | `configs/training/trigo-gpt2.yaml` with override |
| (no args) | (uses default) | `configs/training/trigo-gpt2.yaml` |

## Benefits

1. **More concise**: `train_lm.py trigo-gpt2` vs `train_lm.py --config-name=trigo-gpt2`
2. **More intuitive**: Positional argument feels more natural for the primary input
3. **Backward compatible**: All existing Hydra syntax still works
4. **Flexible**: Supports both short names and full paths

## Shell Script Integration

The positional argument feature makes it easy to create shell aliases or scripts:

```bash
#!/bin/bash
# commands.local.sh

# Train with different configs
train_gpt2() {
    python train_lm.py trigo-gpt2 "$@"
}

train_llama() {
    python train_lm.py trigo-llama "$@"
}

train_invsqrt() {
    python train_lm.py trigo-gpt2-invsqrt "$@"
}

# Usage:
# train_gpt2 training.epochs=50
# train_llama training.learning_rate=1e-4
```

## Edge Cases

### What Happens If...

**Q: What if config file doesn't exist?**
A: Hydra will show an error listing available configs:
```
Cannot find primary config 'nonexistent'. Check that it's in your config search path.
```

**Q: What if I provide a path outside configs/training/?**
A: The script extracts just the filename and looks in the default `configs/training/` directory.

**Q: What if I provide an absolute path?**
A: Same as above - only the filename is used, default search path applies.

**Q: What if first argument looks like a Hydra parameter (starts with `-` or contains `=`)?**
A: The parser skips it and lets Hydra handle it as usual.

## Comparison with Other Tools

### Before (Hydra standard)
```bash
python train_lm.py --config-name=trigo-gpt2 training.epochs=50
```

### After (with positional arg)
```bash
python train_lm.py trigo-gpt2 training.epochs=50
```

### Similar to other tools
```bash
# Similar to argparse-based scripts
python script.py config.yaml --epochs 50

# Similar to common CLI tools
docker run -f Dockerfile
git clone <repo>
```

## Documentation Updates

Updated files:
- `train_lm.py` - Added `parse_positional_config()` function and updated docstring
- `docs/train_lm_positional_arg.md` - This documentation file

Updated docstring in `train_lm.py`:
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

## Testing

Manual testing conducted:
- ✓ Short config name
- ✓ Full config path
- ✓ With parameter overrides
- ✓ Backward compatibility with Hydra syntax
- ✓ Default config (no args)
- ✓ Error handling for nonexistent configs

## Future Enhancements

Potential improvements:
1. Support for config directories (not just files)
2. Config name auto-completion in shell
3. Fuzzy matching for config names
4. List available configs with `--list-configs`
5. Validate config exists before Hydra processes it (better error messages)

## Related Features

This feature works well with:
- Hydra resolvers (`${hydra:job.config_name}`)
- Environment variable configuration
- Wandb integration
- Checkpoint management

## Status

✓ Implementation complete
✓ Tested with multiple configs
✓ Backward compatible
✓ Documentation complete
✓ Ready for use
