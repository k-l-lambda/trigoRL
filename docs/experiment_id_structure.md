# Experiment ID and Output Directory Structure

## Date
2025-11-13

## Overview

Refactored output directory structure to use a global `id` field (like deep-starry), which determines where all experiment outputs (checkpoints, logs, config) are saved.

## Changes Made

### 1. Config File Structure

**Before:**
```yaml
paths:
  output: ./outputs

training:
  save_dir: ${paths.output}/checkpoints/${date:}-${hydra:job.config_name}
```

**After:**
```yaml
# Global experiment ID
id: trigor/${date:}-${hydra:job.config_name}

paths:
  output: ./outputs

training:
  # No save_dir field - derived from id automatically
  save_frequency: 5
  keep_n_checkpoints: 5
  save_mode: best
```

### 2. Output Directory Structure

**New structure:**
```
outputs/
└── trigor/
    └── 20251113-trigo-gpt2/          # Based on id field
        ├── config.yaml               # Saved config (NEW)
        ├── train.log                 # Console log file (NEW)
        └── checkpoints/              # Checkpoints directory
            ├── latest.chkpt
            └── best_ep0010_val_loss_0.1234.chkpt
```

**Benefits:**
- All experiment outputs in one directory
- Easy to identify experiments by date and config name
- Config and logs are preserved with checkpoints
- Follows deep-starry convention

### 3. train_lm.py Changes

**Added at start of main():**

```python
# Setup output directory based on id
output_dir = Path(config.paths.output) / config.id
output_dir.mkdir(parents=True, exist_ok=True)

# Setup file logging
log_file = output_dir / "train.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logging.getLogger().addHandler(file_handler)

# Save config to output directory
config_file = output_dir / "config.yaml"
with open(config_file, 'w') as f:
    f.write(OmegaConf.to_yaml(config))
```

**Logs:**
```
Experiment ID: trigor/20251113-trigo-gpt2
Output directory: ./outputs/trigor/20251113-trigo-gpt2
Config saved to: ./outputs/trigor/20251113-trigo-gpt2/config.yaml
Log file: ./outputs/trigor/20251113-trigo-gpt2/train.log
```

### 4. LMTrainer Changes

**Before:**
```python
self.checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path(config.training.save_dir),
    # ...
)
```

**After:**
```python
checkpoint_dir = Path(config.paths.output) / config.id / "checkpoints"
self.checkpoint_mgr = CheckpointManager(
    checkpoint_dir=checkpoint_dir,
    # ...
)
```

Checkpoint directory is now derived from the global `id` field.

## Features Added

### 1. Config File Preservation

The resolved configuration is saved to `{output_dir}/config.yaml`:
- All variables resolved (${date:}, ${hydra:...}, etc.)
- Complete snapshot of training configuration
- Easy to reproduce experiments

### 2. Console Log File

All console output is written to `{output_dir}/train.log`:
- Same format as console output
- Includes timestamps
- Preserved after training
- Easy to review training progress

### 3. Structured Output

All experiment artifacts in one place:
```
outputs/trigor/20251113-trigo-gpt2/
├── config.yaml          # Configuration snapshot
├── train.log            # Console log
└── checkpoints/         # Model checkpoints
    ├── latest.chkpt
    └── best_ep0010_val_loss_0.1234.chkpt
```

## ID Format

The `id` field uses resolvers to create meaningful directory names:

```yaml
# Format
id: {project}/{date}-{config_name}

# Example
id: trigor/${date:}-${hydra:job.config_name}

# Resolves to
trigor/20251113-trigo-gpt2
```

**Components:**
- `trigor`: Project name
- `${date:}`: Current date in yyyymmdd format (e.g., 20251113)
- `${hydra:job.config_name}`: Config file name without .yaml

**Benefits:**
- Chronologically sortable (date first)
- Self-documenting (includes config name)
- Unique per day and config

## Example Usage

### Training with ID

```bash
# Standard training
python train_lm.py trigo-gpt2

# Output structure created automatically:
# outputs/trigor/20251113-trigo-gpt2/
#   ├── config.yaml
#   ├── train.log
#   └── checkpoints/...
```

### Custom ID

```yaml
# In config file
id: my-experiment/${date:}-custom-run

# Results in:
# outputs/my-experiment/20251113-custom-run/...
```

### Finding Experiments

```bash
# List all experiments
ls outputs/trigor/

# Find today's experiments
ls outputs/trigor/ | grep $(date +%Y%m%d)

# Find specific config experiments
ls outputs/trigor/ | grep trigo-gpt2

# View experiment config
cat outputs/trigor/20251113-trigo-gpt2/config.yaml

# View experiment log
cat outputs/trigor/20251113-trigo-gpt2/train.log
```

## Compatibility with Other Tools

### Wandb Integration

The id can be used with wandb:

```yaml
training:
  wandb:
    name: ${id}  # Use id as run name
    tags:
      - ${date:}
      - ${hydra:job.config_name}
```

### Tensorboard

```python
# Can use id for tensorboard logs too
tensorboard_dir = Path(config.paths.output) / config.id / "tensorboard"
```

## Migration Guide

### For Existing Configs

1. Add `id` field at the top:
   ```yaml
   id: trigor/${date:}-${hydra:job.config_name}
   ```

2. Remove `training.save_dir` field

3. Update any hardcoded paths to use `${paths.output}/${id}`

### For Running Experiments

Old checkpoints remain in their original locations. New experiments will use the new structure.

## Files Modified

1. `configs/training/trigo-gpt2.yaml`:
   - Added `id` field
   - Removed `training.save_dir` field

2. `train_lm.py`:
   - Added output directory setup
   - Added file logging
   - Added config saving

3. `trigor/training/lm_trainer.py`:
   - Changed checkpoint dir to use `config.id`
   - Added checkpoint dir logging

4. `docs/experiment_id_structure.md`:
   - This documentation

## Benefits Summary

1. **Organization**: All experiment files in one place
2. **Reproducibility**: Config saved with results
3. **Debugging**: Console logs preserved
4. **Discovery**: Easy to find experiments by date/name
5. **Consistency**: Follows deep-starry convention
6. **Simplicity**: One `id` field controls all output paths

## Status

✓ Implemented
✓ Tested
✓ Documented
✓ Ready to use
