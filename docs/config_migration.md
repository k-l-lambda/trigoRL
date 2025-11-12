# Training Configuration Migration Guide

## Overview

The training configurations have been updated to use `AttentionCausalLoss` as a wrapper around the base models. This change simplifies the training pipeline by automatically computing loss and metrics in a single forward pass.

## What Changed

### Before (Old Configuration)

```yaml
model:
  type: GPT2CausalLM
  vocab_size: 259
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  max_seq_len: 2048
  # ... other model parameters
```

### After (New Configuration) - Nested Structure

The new configuration uses a **nested structure** with `type` + `config` pattern at each hierarchy level:

```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: GPT2CausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        num_heads: 8
        max_seq_len: 2048
        # ... other model parameters
    ignore_index: 256  # PAD token to ignore in loss
    label_smoothing: 0.1  # Regularization
```

**Key Structure Changes:**
1. Top level: `model.type` = `AttentionCausalLoss` (wrapper)
2. Second level: `model.config` contains wrapper configuration
3. Third level: `model.config.model_config.type` = base model type (e.g., `GPT2CausalLM`)
4. Fourth level: `model.config.model_config.config` = base model parameters

This consistent `type` + `config` pattern makes the hierarchy clearer and more maintainable.

## Updated Configurations

All training configurations have been migrated:

1. **`configs/training/trigo-gpt2.yaml`** - GPT-2 with MHA
2. **`configs/training/trigo-llama.yaml`** - LLaMA with GQA
3. **`configs/training/trigo-rwkv.yaml`** - RWKV with linear attention
4. **`configs/training/trigo-xlstm.yaml`** - xLSTM with chunk parallelization

## Benefits of the New Structure

### 1. Unified Loss Computation

The model now automatically computes loss and metrics in forward pass:

```python
# Old way (manual loss computation)
model = make_model('GPT2CausalLM', config)
outputs = model(input_ids, attention_mask)
logits = outputs.logits
loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

# New way (automatic loss and metrics)
# Note: Pass cfg.model.config to get the wrapper configuration
model = make_model('AttentionCausalLoss', cfg.model.config)
outputs = model(input_ids, labels, attention_mask)
# Automatically returns: loss, accuracy, top5_accuracy, perplexity, num_tokens
```

### 2. Comprehensive Metrics

Every forward pass provides:
- **Loss**: Cross-entropy with label smoothing
- **Accuracy**: Token-level accuracy (excluding padding)
- **Top-5 Accuracy**: Correct token in top-5 predictions
- **Perplexity**: exp(loss)
- **Num Tokens**: Valid token count

### 3. Simplified Training Loop

```python
from omegaconf import OmegaConf
from trigor.models import make_model

# Load config
cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')

# Create model (pass cfg.model.config for nested structure)
model = make_model(cfg.model.type, cfg.model.config)

# Training loop
for batch in dataloader:
    outputs = model(batch['input_ids'], batch['labels'], batch['attention_mask'])
    loss = outputs['loss']

    # Optionally log metrics
    print(f"Loss: {loss.item():.4f}, Accuracy: {outputs['accuracy'].item():.4f}")

    # Backward pass
    loss.backward()
    optimizer.step()
```

### 4. Padding Handling

Padding tokens (ID=256) are automatically ignored in:
- Loss computation (via `ignore_index`)
- Accuracy calculation
- All metrics

### 5. Label Smoothing

Built-in label smoothing (`label_smoothing=0.1`) for regularization:
- Prevents overfitting by softening hard labels
- Improves generalization

## Configuration Parameters

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `type` | str | Must be `AttentionCausalLoss` | Required |
| `config.model_config.type` | str | Underlying model (e.g., `GPT2CausalLM`) | Required |
| `config.model_config.config` | dict | Configuration for the underlying model | Required |
| `config.ignore_index` | int | Token ID to ignore in loss (PAD token) | 256 |
| `config.label_smoothing` | float | Label smoothing factor (0.0-1.0) | 0.1 |

### Model-Specific Parameters

Each model type has its own parameters in `config.model_config.config`:

**GPT-2**:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: GPT2CausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        num_heads: 8
        max_seq_len: 2048
        dropout: 0.1
        activation: gelu_new
        intermediate_size: 1024
    ignore_index: 256
    label_smoothing: 0.1
```

**LLaMA**:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: LlamaCausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        num_heads: 8
        num_key_value_heads: 2  # GQA
        max_seq_len: 2048
        activation: silu
    ignore_index: 256
    label_smoothing: 0.1
```

**RWKV**:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: RwkvCausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        max_seq_len: 2048
        intermediate_size: 1024
    ignore_index: 256
    label_smoothing: 0.1
```

**xLSTM**:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: xLSTMCausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        num_heads: 8
        chunk_size: 64
        max_seq_len: 2048
    ignore_index: 256
    label_smoothing: 0.1
```

## Backward Compatibility

### For Existing Code

If you have existing training code that expects the old configuration:

**Option 1**: Update to use new configs (recommended)
```python
# Simply change config loading, rest of code stays the same
cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')  # Uses new format
# Pass cfg.model.config to get the wrapper configuration
model = make_model(cfg.model.type, cfg.model.config)  # Creates AttentionCausalLoss
```

**Option 2**: Extract inner model if needed
```python
# Load new config
cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')

# Create wrapped model (pass cfg.model.config)
loss_module = make_model(cfg.model.type, cfg.model.config)

# Access inner model directly
inner_model = loss_module.model  # The actual GPT2CausalLM instance
```

### For Direct Model Creation

You can still create models directly without the wrapper:

```python
from trigor.models import GPT2CausalLM

# Direct model creation (old way)
model = GPT2CausalLM.from_config(model_config)

# Or with factory
from trigor.models import make_model
model = make_model('GPT2CausalLM', model_config)
```

## Verification

Test that all configurations work:

```bash
python tests/test_updated_configs.py
```

**Expected output:**
```
✓ ALL TESTS PASSED!

Verification:
  ✓ All configs load correctly
  ✓ AttentionCausalLoss wrapper created for all models
  ✓ Forward pass works with loss and metrics
  ✓ Label smoothing and ignore_index configured
```

## Training Script Example

Complete training script using new configurations:

```python
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from trigor.data import TGNDataset
from trigor.models import make_model

# Load configuration
cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')

# Setup paths
OmegaConf.update(cfg, "paths.root", ".")
OmegaConf.resolve(cfg)

# Create dataset
dataset = TGNDataset.from_config(cfg.data)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.data.loader.batch_size,
    shuffle=cfg.data.loader.shuffle,
    collate_fn=TGNDataset.collate_batch,
)

# Create model (automatically includes loss)
model = make_model(cfg.model.type, cfg.model.config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.training.learning_rate,
    weight_decay=cfg.training.weight_decay,
)

# Training loop
model.train()
for epoch in range(cfg.training.epochs):
    for batch in dataloader:
        # Forward pass (computes loss and metrics automatically)
        outputs = model(
            batch['input_ids'],
            batch['labels'],
            batch['attention_mask'],
        )

        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.training.max_grad_norm,
        )
        optimizer.step()

        # Logging
        if step % cfg.training.log_frequency == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {outputs['accuracy'].item():.4f}, "
                  f"Perplexity: {outputs['perplexity'].item():.2f}")
```

## FAQ

### Q: Why wrap models in AttentionCausalLoss?

**A**: Simplifies training by automatically computing loss and metrics. No need to manually implement loss calculation and accuracy tracking.

### Q: Can I still access the underlying model?

**A**: Yes, via `model.model` attribute:
```python
loss_module = make_model('AttentionCausalLoss', cfg.model.config)
inner_model = loss_module.model  # GPT2CausalLM, LlamaCausalLM, etc.
```

### Q: Does this change affect inference?

**A**: No. For inference, you can:
1. Use the wrapper (returns logits if you pass `return_logits=True`)
2. Access the inner model directly
3. Use the `generate()` method for autoregressive generation

### Q: What if I don't want label smoothing?

**A**: Set `label_smoothing: 0.0` in the config:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: GPT2CausalLM
      config: {...}
    label_smoothing: 0.0  # No smoothing
```

### Q: Can I use custom loss functions?

**A**: Yes, subclass `AttentionCausalLoss` and override `__init__`:
```python
class CustomLoss(AttentionCausalLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use custom loss function
        self.loss_fn = MyCustomLoss(...)
```

## Summary

The migration to `AttentionCausalLoss` wrapper provides:

✓ **Automatic loss and metrics computation**
✓ **Cleaner training code** (less boilerplate)
✓ **Consistent interface** across all models
✓ **Built-in label smoothing** for regularization
✓ **Proper padding handling** (ignore PAD tokens)
✓ **Comprehensive metrics** (loss, accuracy, top-5, perplexity)
✓ **Backward compatible** (inner model still accessible)

All existing functionality is preserved while making the training pipeline simpler and more maintainable.
