# LambdaLR Scheduler Implementation - Summary

## Implementation Date

2025-11-13

## Overview

Added support for PyTorch's `LambdaLR` scheduler to LMTrainer, including:
1. **Inverse square root scheduler** (Transformer-style from "Attention Is All You Need")
2. **Custom lambda scheduler** (user-defined learning rate functions)

## Changes Made

### 1. trigor/training/lm_trainer.py

**Import added:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
```

**_setup_scheduler() method updated** (lines 177-264):

Added two new scheduler types before existing scheduler logic:

**Inverse sqrt scheduler** (type: `inverse_sqrt`):
- Uses LambdaLR with formula: `lr = lr_mul * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)`
- Parameters:
  - `d_model`: Model hidden dimension (default: 512)
  - `lr_mul`: Learning rate multiplier (default: 1.0)
- Built-in warmup (no separate warmup phase needed)
- Recommended for Transformer models

**Custom lambda scheduler** (type: `lambda`):
- Uses LambdaLR with user-provided lambda function
- Parameter: `lambda_fn` (string that will be evaluated)
- Warning: Uses `eval()` - only for trusted input
- Flexible for custom schedules

### 2. Config Files

**Created: configs/training/trigo-gpt2-invsqrt.yaml**
- Complete example using inverse_sqrt scheduler
- Shows recommended parameters:
  - `learning_rate: 1.0` (scaled by scheduler)
  - `warmup_steps: 4000`
  - `d_model: 256` (matches model hidden_size)
  - `lr_mul: 1.0`

**Updated: configs/training/trigo-gpt2.yaml**
- Updated scheduler type comment to include new options:
  ```yaml
  type: cosine  # cosine, linear, constant, inverse_sqrt, lambda
  ```

### 3. Testing

**Created: tests/test_schedulers.py**
- Comprehensive test for both scheduler types
- Generates learning rate curves
- Validates warmup and decay behavior
- Creates visualization: `outputs/scheduler_comparison.png`

**Test results:**
- Inverse sqrt scheduler: ✓ Passed
- Custom lambda scheduler: ✓ Passed
- Learning rate curves generated successfully

### 4. Documentation

**Created: docs/schedulers.md**
- Complete guide for all supported schedulers
- Configuration examples for each type
- Use case recommendations
- Formula explanations
- Visualization instructions
- Implementation details

## Supported Scheduler Types

Now supporting 5 scheduler types:

1. **cosine** - Cosine annealing (existing)
2. **linear** - Linear decay (existing)
3. **constant** - Constant LR after warmup (existing)
4. **inverse_sqrt** - Transformer-style (NEW)
5. **lambda** - Custom function (NEW)

## Configuration Examples

### Inverse Sqrt (Transformer-style)

```yaml
training:
  learning_rate: 1.0
  warmup_steps: 4000
  scheduler:
    type: inverse_sqrt
    d_model: 256      # Match model hidden_size
    lr_mul: 1.0       # Optional multiplier
```

### Custom Lambda

```yaml
training:
  learning_rate: 1e-4
  scheduler:
    type: lambda
    lambda_fn: "lambda step: 0.95 ** (step / 1000)"
```

## Learning Rate Behavior

### Inverse Sqrt Schedule (d_model=256, warmup=4000)

```
Step     1: lr = 0.000000
Step   100: lr = 0.000025
Step  1000: lr = 0.000247
Step  4000: lr = 0.000988 (peak after warmup)
Step  8000: lr = 0.000699
Step 16000: lr = 0.000494
Step 32000: lr = 0.000349
```

**Characteristics:**
- Gradual warmup proportional to step
- Decays as `step^-0.5` after warmup
- Peak LR at end of warmup
- Slower decay than cosine/linear

### Comparison with Existing Schedulers

**Cosine:**
- Fast decay to min_lr
- Good for fixed-length training
- Most common choice

**Linear:**
- Predictable decay
- Good for fine-tuning
- Simple and stable

**Inverse Sqrt:**
- Slower decay
- Better for long training
- Scales with model size
- Recommended for Transformers

## Use Cases

**When to use inverse_sqrt:**
- Training Transformer models from scratch
- Long training runs (100k+ steps)
- Want LR to scale automatically with model size
- Following "Attention Is All You Need" paper

**When to use custom lambda:**
- Reproducing specific papers
- Experimenting with novel schedules
- Need fine-grained control

## Implementation Details

### Inverse Sqrt Formula

Original Transformer paper formula:
```
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

Our implementation:
```python
def lr_lambda(current_step):
    if current_step == 0:
        current_step = 1

    scale = d_model ** -0.5
    step_scale = min(current_step ** (-0.5),
                    current_step * warmup_steps ** (-1.5))

    return lr_mul * scale * step_scale
```

### Why d_model Scaling?

The `d_model^-0.5` factor scales learning rate with model size:
- Larger models (d_model=512) → smaller base LR (0.044)
- Smaller models (d_model=256) → larger base LR (0.062)

This automatic scaling helps with training stability across different model sizes.

## Testing

Run scheduler tests:
```bash
python tests/test_schedulers.py
```

Output:
- Console: Learning rate values at key steps
- File: `outputs/scheduler_comparison.png` (visualization)

## Backward Compatibility

✓ No breaking changes
✓ Existing configs continue to work
✓ New scheduler types are opt-in

## Performance

- LambdaLR has negligible overhead
- Lambda function called once per training step
- No impact on training speed

## Safety Considerations

**Custom lambda scheduler uses `eval()`:**
- Only use with trusted config files
- Warning logged when lambda scheduler is used
- Consider restricting in production environments

**Recommendation:**
- Use built-in schedulers when possible
- Only use custom lambda for experimentation
- Never use lambda with user-provided configs

## Related Work

### Deep-starry InvSqrtScheduler

Compared to deep-starry's implementation:
- **Similar formula**: Both use Transformer paper formula
- **Integration**: deep-starry wraps optimizer, we use PyTorch's LambdaLR
- **Compatibility**: Our version integrates with PyTorch ecosystem
- **Features**: We support additional schedulers (cosine, linear, etc.)

### PyTorch Built-in Schedulers

PyTorch provides:
- `LambdaLR` (what we use)
- `CosineAnnealingLR` (we already use)
- `LinearLR` (we already use)
- `StepLR`, `MultiStepLR`, `ExponentialLR`, etc.

We chose to implement inverse_sqrt using `LambdaLR` because:
1. No built-in inverse sqrt scheduler in PyTorch
2. LambdaLR provides maximum flexibility
3. Consistent with PyTorch's design patterns

## Future Enhancements

Potential additions:
1. **Polynomial decay** (easy with LambdaLR)
2. **OneCycle** (already in PyTorch)
3. **Warm restarts** (CosineAnnealingWarmRestarts)
4. **ReduceLROnPlateau** (validation-based)
5. **Preset schedules** (e.g., "bert", "gpt3", "t5")

## Files Modified

1. `trigor/training/lm_trainer.py` - Added LambdaLR support
2. `configs/training/trigo-gpt2.yaml` - Updated comments
3. `configs/training/trigo-gpt2-invsqrt.yaml` - Created example

## Files Created

1. `tests/test_schedulers.py` - Test suite
2. `docs/schedulers.md` - Complete documentation
3. `docs/lambda_lr_implementation.md` - This summary

## Verification

✓ Unit tests pass
✓ Scheduler creates correctly
✓ Learning rates match expected values
✓ Integration with LMTrainer works
✓ Documentation complete

## User Request

Original request (Chinese): "补充LambdaLR即可"

Translation: "Just add LambdaLR support"

✓ Request satisfied:
- Added inverse_sqrt scheduler using LambdaLR
- Added custom lambda scheduler
- Provided documentation and examples
- Tested and verified

## Status

✓ Implementation complete
✓ Tests passing
✓ Documentation complete
✓ Config examples provided
✓ Ready for use
