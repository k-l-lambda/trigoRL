# Resume Training from Experiment Directory

## Overview

The training script `train_lm.py` supports resuming training from an experiment directory. This allows you to continue training from where it left off, preserving all training state including model weights, optimizer state, learning rate schedule, and epoch progress.

## Configuration Resolution

The saved `config.yaml` in the experiment directory has all variable references resolved to actual values:

**Original config (with variables):**
```yaml
id: trigor/${date:}-${hydra:job.config_name}
paths:
  data: ${paths.root}/data
  output: ${paths.root}/outputs
data:
  data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output
wandb:
  name: ${hydra:job.config_name}
```

**Saved config (resolved):**
```yaml
id: trigor/20251113-trigo-gpt2
paths:
  data: ./data
  output: ./outputs
data:
  data_dir: ./third_party/trigo/trigo-web/tools/output
wandb:
  name: trigo-gpt2
```

This makes the config file:
- **Self-contained** - No external variables needed
- **Reproducible** - Same values regardless of when loaded
- **Readable** - Clear what values were actually used

## Usage

### Resume from Experiment Directory

Simply pass the experiment directory path as the first argument:

```bash
python train_lm.py outputs/trigor/20251113-trigo-gpt2/
```

The script will automatically:
1. Detect that the path is an experiment directory
2. Load the saved configuration from `config.yaml`
3. Load the latest checkpoint from `checkpoints/latest.chkpt`
4. Continue training from the next epoch

### Resume with Configuration Overrides

You can override configuration parameters when resuming:

```bash
# Resume and change number of epochs
python train_lm.py outputs/trigor/20251113-trigo-gpt2/ training.epochs=100

# Resume and enable wandb
python train_lm.py outputs/trigor/20251113-trigo-gpt2/ training.wandb.enabled=true

# Resume with multiple overrides
python train_lm.py outputs/trigor/20251113-trigo-gpt2/ \
    training.epochs=100 \
    data.loader.batch_size=8 \
    training.wandb.enabled=true
```

**Note:** Configuration overrides have higher priority than the saved config. This allows you to adjust hyperparameters when resuming.

## Requirements

For resume to work, the experiment directory must contain:

1. **config.yaml** - Saved configuration file
2. **checkpoints/latest.chkpt** - Latest checkpoint file

If either file is missing, the script will report an error and exit.

## How It Works

### Detection

When you pass a directory path as the first argument, the script checks:
```python
if arg_path.is_dir():
    config_file = arg_path / "config.yaml"
    checkpoint_file = arg_path / "checkpoints" / "latest.chkpt"

    if config_file.exists() and checkpoint_file.exists():
        # This is a valid experiment directory
```

### Loading Process

1. **Load saved configuration** from `config.yaml`
2. **Merge with CLI overrides** (overrides have priority)
3. **Use same output directory** (no new directory created)
4. **Open log file in append mode** (adds to existing log)
5. **Load checkpoint** including:
   - Model state (weights)
   - Optimizer state (momentum, etc.)
   - Scheduler state (learning rate schedule)
   - Training progress (epoch, global_step)
   - Best validation metric

### Training Continuation

The trainer will:
- Start from the next epoch after the saved checkpoint
- Continue with the same global step count
- Maintain learning rate schedule position
- Track best validation metric from before

## Examples

### Example 1: Basic Resume

Train for 5 epochs, then resume to train for 10 more:

```bash
# Initial training (5 epochs)
python train_lm.py trigo-gpt2 training.epochs=5

# Resume and train for 10 more epochs
python train_lm.py outputs/trigor/20251113-trigo-gpt2/ training.epochs=15
```

The model will train from epoch 5 to epoch 15.

### Example 2: Resume with Wandb

Start without wandb, then enable it when resuming:

```bash
# Initial training without wandb
python train_lm.py trigo-gpt2 training.epochs=10 training.wandb.enabled=false

# Resume with wandb enabled
python train_lm.py outputs/trigor/20251113-trigo-gpt2/ \
    training.epochs=100 \
    training.wandb.enabled=true
```

### Example 3: Resume After Crash

If training crashes or is interrupted:

```bash
# Training crashes at epoch 37
python train_lm.py trigo-gpt2 training.epochs=100
# ... (Ctrl+C or crash)

# Simply resume from the same directory
python train_lm.py outputs/trigor/20251113-trigo-gpt2/
```

The checkpoint manager saves both `latest.chkpt` (most recent) and `best_*.chkpt` (best validation), so you can always resume from the latest state.

## Log File Behavior

### Append Mode

When resuming, the log file is opened in append mode:
```
outputs/trigor/20251113-trigo-gpt2/train.log
```

**Example log structure:**
```
# Initial training
2025-11-13 18:36:27 - INFO - Attention Language Model Training
2025-11-13 18:36:27 - INFO - Experiment ID: trigor/20251113-trigo-gpt2
...
2025-11-13 18:40:33 - INFO - Training Complete

# After resume
2025-11-13 18:51:24 - INFO - Resuming Training from Experiment Directory
2025-11-13 18:51:25 - INFO - Resumed from epoch 1, step 19
...
2025-11-13 18:55:12 - INFO - Training Complete
```

This preserves the complete training history in one file.

## Checkpoint Information

The checkpoint contains all training state:

```python
{
    'epoch': current_epoch,           # Resume from next epoch
    'global_step': global_step,       # Continue step count
    'model_state_dict': ...,          # Model weights
    'optimizer_state_dict': ...,      # Optimizer state (momentum, etc.)
    'scheduler_state_dict': ...,      # LR schedule position
    'best_val_metric': ...,           # Best validation metric so far
    'config': ...,                    # Original configuration
}
```

## Comparison with resume_from Parameter

There are two ways to resume training:

### 1. From Experiment Directory (Recommended)
```bash
python train_lm.py outputs/trigor/20251113-trigo-gpt2/
```

**Advantages:**
- Automatically loads saved config
- Uses same output directory
- Appends to log file
- More convenient

### 2. Using resume_from Parameter
```bash
python train_lm.py trigo-gpt2 resume_from=outputs/trigor/20251113-trigo-gpt2/checkpoints/latest.chkpt
```

**Advantages:**
- Can resume into a different configuration
- Can create a new experiment directory
- More flexible but requires manual config management

## Error Handling

### Missing config.yaml
```
ERROR - Invalid experiment directory: outputs/trigor/20251113-trigo-gpt2
ERROR -   Missing config file: outputs/trigor/20251113-trigo-gpt2/config.yaml
```

### Missing checkpoint
```
ERROR - Invalid experiment directory: outputs/trigor/20251113-trigo-gpt2
ERROR -   Missing checkpoint: outputs/trigor/20251113-trigo-gpt2/checkpoints/latest.chkpt
```

If you see these errors, the directory is not a valid experiment directory or the checkpoint was deleted.

## Best Practices

1. **Always keep checkpoints safe** - Don't delete checkpoint files if you might resume
2. **Monitor disk space** - Checkpoints are ~80MB each
3. **Use meaningful experiment IDs** - The directory name should be descriptive
4. **Check logs before resuming** - Verify the last epoch that completed
5. **Test resume on small runs** - Make sure resume works before long training

## Troubleshooting

### Resume doesn't start from expected epoch

Check the checkpoint file:
```bash
python -c "import torch; ckpt = torch.load('outputs/trigor/20251113-trigo-gpt2/checkpoints/latest.chkpt'); print(f'Epoch: {ckpt[\"epoch\"]}, Step: {ckpt[\"global_step\"]}')"
```

### Learning rate seems wrong after resume

The scheduler state is restored, so LR will continue from where it left off. This is correct behavior.

### Different results after resume

If `deterministic=true` and seeds are set, results should be reproducible. However, if you change batch size or other data parameters, results may differ.

## Implementation Details

The resume logic is in `train_lm.py`:

**Detection phase** (`parse_positional_config()`):
- Checks if first argument is a directory
- Validates presence of config.yaml and latest.chkpt
- Returns experiment directory path

**Main function**:
- Loads saved config from directory
- Merges with CLI overrides
- Sets up logging in append mode
- Passes checkpoint path to trainer

**Trainer** (`lm_trainer.py`):
- Loads checkpoint with `load_checkpoint()`
- Restores model, optimizer, scheduler states
- Sets `current_epoch` and `global_step`
- Continues training from next epoch
