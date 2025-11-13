# Logging Improvements

## Summary

Refactored the training pipeline to use Python's `logging` module instead of print statements for more professional and configurable logging.

## Changes Made

### 1. Added Logging Setup

**train_lm.py:**
```python
import logging

# Setup logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
```

**trigor/training/lm_trainer.py:**
```python
import logging

logger = logging.getLogger(__name__)
```

### 2. Replaced Print Statements

**Before:**
```python
print("\n" + "="*80)
print("Creating Model")
print("="*80)
print(f"\nModel created:")
print(f"  Type: {config.model.type}")
```

**After:**
```python
logger.info("=" * 80)
logger.info("Creating Model")
logger.info("=" * 80)
logger.info("")
logger.info("Model created:")
logger.info(f"  Type: {config.model.type}")
```

### 3. Log Levels Used

- **`logger.info()`** - Standard information messages (most output)
- **`logger.warning()`** - Warnings (e.g., CUDA not available, no checkpoint found)
- **`logger.error()`** - Error messages (training failures)

## Benefits

### 1. Professional Output Format

**Before (plain print):**
```
Creating Model
================================================================================

Model created:
  Type: AttentionCausalLoss
```

**After (with logging):**
```
[2025-11-13 14:48:34][INFO] - ================================================================================
[2025-11-13 14:48:34][INFO] - Creating Model
[2025-11-13 14:48:34][INFO] - ================================================================================
[2025-11-13 14:48:34][INFO] -
[2025-11-13 14:48:34][INFO] - Model created:
[2025-11-13 14:48:34][INFO] -   Type: AttentionCausalLoss
```

### 2. Timestamps

Every log message includes a timestamp, making it easy to:
- Track training duration
- Debug timing issues
- Measure epoch/batch times
- Correlate logs with system events

### 3. Log Levels

Different message types are clearly distinguished:
```python
logger.info("Training started")          # [INFO]
logger.warning("CUDA not available")     # [WARNING]
logger.error("Training failed")          # [ERROR]
```

### 4. Configurable Verbosity

Easy to adjust log level without code changes:
```python
# Show only warnings and errors
logging.basicConfig(level=logging.WARNING)

# Show debug messages too
logging.basicConfig(level=logging.DEBUG)
```

### 5. Structured Logging

Logs can be easily:
- Redirected to files: `python train_lm.py > training.log 2>&1`
- Parsed by log aggregation tools
- Filtered by level or module
- Integrated with monitoring systems

### 6. Module-Level Loggers

Each module has its own logger:
```python
# train_lm.py
logger = logging.getLogger(__name__)  # __main__

# lm_trainer.py
logger = logging.getLogger(__name__)  # trigor.training.lm_trainer
```

This allows filtering by module:
```python
# Only show trainer logs
logging.getLogger('trigor.training').setLevel(logging.INFO)
logging.getLogger('__main__').setLevel(logging.WARNING)
```

## Example Output

### Training Start
```
[2025-11-13 14:48:34][INFO] - ================================================================================
[2025-11-13 14:48:34][INFO] - Attention Language Model Training
[2025-11-13 14:48:34][INFO] - ================================================================================
[2025-11-13 14:48:34][INFO] - Setting Random Seeds
[2025-11-13 14:48:34][INFO] -   Seed: 42
[2025-11-13 14:48:34][INFO] -   Deterministic: True
[2025-11-13 14:48:34][INFO] - Device: cuda
[2025-11-13 14:48:34][INFO] -   GPU: NVIDIA GeForce RTX 3090
[2025-11-13 14:48:34][INFO] -   Memory: 25.43 GB
```

### Model Creation
```
[2025-11-13 14:48:35][INFO] - ================================================================================
[2025-11-13 14:48:35][INFO] - Creating Model
[2025-11-13 14:48:35][INFO] - ================================================================================
[2025-11-13 14:48:36][INFO] -
[2025-11-13 14:48:36][INFO] - Model created:
[2025-11-13 14:48:36][INFO] -   Type: AttentionCausalLoss
[2025-11-13 14:48:36][INFO] -   Base model: GPT2CausalLM
[2025-11-13 14:48:36][INFO] -   Total parameters: 6,902,528
[2025-11-13 14:48:36][INFO] -   Trainable parameters: 6,902,528
```

### Epoch Summary
```
[2025-11-13 14:52:15][INFO] -
[2025-11-13 14:52:15][INFO] - Epoch 1/100 Summary:
[2025-11-13 14:52:15][INFO] -   Train Loss: 5.1936
[2025-11-13 14:52:15][INFO] -   Train Accuracy: 0.1053
[2025-11-13 14:52:15][INFO] -   Train Perplexity: 185.20
[2025-11-13 14:52:15][INFO] -   Val Loss: 4.6997
[2025-11-13 14:52:15][INFO] -   Val Accuracy: 0.1739
[2025-11-13 14:52:15][INFO] -   Val Perplexity: 109.99
[2025-11-13 14:52:15][INFO] - Checkpoint saved: outputs/checkpoints/gpt2/best_ep0000_val_loss_4.6997.chkpt
[2025-11-13 14:52:15][INFO] - New best val_loss: 4.6997
```

### Warning Example
```
[2025-11-13 14:48:34][WARNING] - CUDA requested but not available, using CPU
[2025-11-13 14:52:20][WARNING] - Training interrupted by user
[2025-11-13 14:52:20][INFO] - Saving checkpoint before exit...
```

### Error Example
```
[2025-11-13 14:48:35][ERROR] - Training failed with error: CUDA out of memory
Traceback (most recent call last):
  File "train_lm.py", line 173, in <module>
    main()
  ...
```

## Advanced Usage

### Save Logs to File

```bash
# Save all logs to file
python train_lm.py > training.log 2>&1

# Save with timestamps in filename
python train_lm.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1
```

### Filter by Level

```bash
# Only show warnings and errors
python train_lm.py 2>&1 | grep -E "WARNING|ERROR"

# Show only info messages (no warnings)
python train_lm.py 2>&1 | grep INFO
```

### Custom Log Format

Modify `train_lm.py` to customize format:

```python
# Add more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Simpler format
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Add file and line number
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
```

## Files Modified

- `trigor/training/lm_trainer.py`: Added logging, replaced all print statements
- `train_lm.py`: Added logging setup, replaced all print statements

## Backward Compatibility

The change is fully backward compatible:
- Training behavior unchanged
- Output content unchanged (only format improved)
- All existing configs work without modification
- Progress bars (tqdm) still work as before
