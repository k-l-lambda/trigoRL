# Learning Rate Schedulers Guide

## Overview

LMTrainer supports multiple learning rate scheduling strategies through the config file. All schedulers can optionally include a warmup phase.

## Supported Schedulers

### 1. Cosine Annealing (`cosine`)

Cosine decay from initial learning rate to minimum learning rate.

**Config:**
```yaml
training:
  learning_rate: 1e-4
  warmup_steps: 1000
  scheduler:
    type: cosine
    min_lr: 1e-6
```

**Formula:**
- Warmup (steps 1 to warmup_steps): Linear increase from `0.01 * lr` to `lr`
- Main phase: Cosine decay from `lr` to `min_lr`

**Use cases:**
- General purpose training
- Standard transformer training
- Works well for most scenarios

### 2. Linear Decay (`linear`)

Linear decay from initial learning rate to minimum learning rate.

**Config:**
```yaml
training:
  learning_rate: 1e-4
  warmup_steps: 1000
  scheduler:
    type: linear
    min_lr: 1e-6
```

**Formula:**
- Warmup (steps 1 to warmup_steps): Linear increase from `0.01 * lr` to `lr`
- Main phase: Linear decay from `lr` to `min_lr`

**Use cases:**
- Fine-tuning pre-trained models
- Short training runs
- When you know exact number of training steps

### 3. Constant Learning Rate (`constant`)

Constant learning rate after warmup (or no scheduler at all).

**Config:**
```yaml
training:
  learning_rate: 1e-4
  warmup_steps: 1000
  scheduler:
    type: constant
```

**Formula:**
- Warmup (if warmup_steps > 0): Linear increase from `0.01 * lr` to `lr`
- Main phase: Constant at `lr`

**Use cases:**
- Debugging
- Exploratory training
- When learning rate is already well-tuned

### 4. Inverse Square Root (`inverse_sqrt`)

Transformer-style learning rate schedule from "Attention Is All You Need" paper.

**Config:**
```yaml
training:
  learning_rate: 1.0  # Will be scaled by scheduler
  warmup_steps: 4000
  scheduler:
    type: inverse_sqrt
    d_model: 512      # Model hidden dimension
    lr_mul: 1.0       # Learning rate multiplier
```

**Formula:**
```python
lr = lr_mul * (d_model^-0.5) * min(step^-0.5, step * warmup_steps^-1.5)
```

- Warmup (steps 1 to warmup_steps): Learning rate increases proportionally to step
- Main phase: Learning rate decreases proportionally to `step^-0.5`

**Parameters:**
- `d_model`: Model hidden dimension (default: 512)
  - Should match your model's `hidden_size`
  - Scales the learning rate: larger models → smaller base LR
- `lr_mul`: Learning rate multiplier (default: 1.0)
  - Additional scaling factor
  - Useful for adjusting learning rate without changing d_model

**Use cases:**
- Transformer models (recommended in original paper)
- Training from scratch
- Long training runs
- When you want automatic LR scaling based on model size

**Characteristics:**
- No need for `min_lr` parameter
- Learning rate determined by warmup_steps and d_model
- Decays more slowly than cosine/linear schedules
- Works well for very long training

**Example learning rates** (d_model=512, lr_mul=1.0, warmup=4000):
```
Step     1: lr = 0.000000
Step   100: lr = 0.000017
Step  1000: lr = 0.000175
Step  4000: lr = 0.000699 (peak)
Step  8000: lr = 0.000494
Step 16000: lr = 0.000349
Step 32000: lr = 0.000247
```

### 5. Custom Lambda (`lambda`)

Arbitrary learning rate schedule using custom lambda function.

**Config:**
```yaml
training:
  learning_rate: 1e-4
  scheduler:
    type: lambda
    lambda_fn: "lambda step: 0.5 ** (step / 10000)"  # Exponential decay
```

**Warning:** Uses `eval()` internally - only use trusted lambda functions!

**Use cases:**
- Experimentation
- Custom schedules not covered by built-in options
- Reproducing specific training recipes

**Examples:**

Exponential decay:
```yaml
lambda_fn: "lambda step: 0.95 ** (step / 1000)"
```

Polynomial decay:
```yaml
lambda_fn: "lambda step: (1 - step / 100000) ** 0.9"
```

Step decay:
```yaml
lambda_fn: "lambda step: 0.1 ** (step // 30000)"
```

Cyclic with warmup:
```yaml
lambda_fn: "lambda step: min(step / 1000, 1.0) * (0.9 + 0.1 * abs(step % 2000 - 1000) / 1000)"
```

## Configuration Examples

### Example 1: GPT-2 with Cosine Annealing

```yaml
training:
  epochs: 100
  learning_rate: 1e-4
  warmup_steps: 1000
  scheduler:
    type: cosine
    min_lr: 1e-6
```

### Example 2: Transformer with Inverse Sqrt

```yaml
training:
  epochs: 100
  learning_rate: 1.0  # Scaled by scheduler
  warmup_steps: 4000
  scheduler:
    type: inverse_sqrt
    d_model: 256  # Match model hidden_size
    lr_mul: 1.0
```

### Example 3: Fine-tuning with Linear Decay

```yaml
training:
  epochs: 10
  learning_rate: 5e-5
  warmup_steps: 100
  scheduler:
    type: linear
    min_lr: 1e-6
```

### Example 4: No Scheduler (Constant LR)

```yaml
training:
  epochs: 50
  learning_rate: 1e-4
  warmup_steps: 0
  scheduler:
    type: constant
```

## Warmup Behavior

All schedulers except `inverse_sqrt` use **separate warmup phase**:
- Linear increase from `0.01 * learning_rate` to `learning_rate`
- Duration: `warmup_steps` steps
- Applied before main scheduler

The `inverse_sqrt` scheduler has **built-in warmup**:
- No separate warmup phase needed
- Warmup behavior controlled by formula
- More gradual than linear warmup

## Choosing a Scheduler

**Use `cosine`** when:
- Training from scratch
- You have a fixed budget of training steps
- General-purpose training

**Use `linear`** when:
- Fine-tuning pre-trained models
- Short training runs
- You need predictable decay

**Use `inverse_sqrt`** when:
- Training Transformer models
- Training from scratch with long runs
- You want LR to scale with model size
- Following Transformer paper recommendations

**Use `constant`** when:
- Debugging
- Learning rate is already well-tuned
- Very short training runs

**Use `lambda`** when:
- You need custom behavior
- Reproducing specific papers
- Experimentation

## Visualization

Run the scheduler test to see learning rate curves:

```bash
python tests/test_schedulers.py
```

Output: `outputs/scheduler_comparison.png`

## Implementation Details

**File:** `trigor/training/lm_trainer.py:177-264`

**Schedulers:**
- `inverse_sqrt`: Uses `LambdaLR` with inverse sqrt formula
- `lambda`: Uses `LambdaLR` with custom function
- `cosine`, `linear`: Use `SequentialLR` with warmup + main phase
- `constant`: Uses `LinearLR` for warmup only (or no scheduler)

**Scheduler step timing:**
- Called after each optimizer step
- Frequency: Every training batch (after gradient accumulation)
- Not called during validation

## References

- Vaswani et al. (2017). "Attention Is All You Need" - Inverse sqrt scheduler
- PyTorch documentation: torch.optim.lr_scheduler
- Smith & Topin (2018). "Super-Convergence" - OneCycle scheduler (not yet implemented)

## Future Enhancements

Potential schedulers to add:
- Polynomial decay
- Reduce on plateau
- OneCycle
- Warm restarts
- Custom schedules from papers
