# AttentionCausalLoss Module

## Overview

The `AttentionCausalLoss` module provides a unified interface for causal language modeling with automatic loss computation and accuracy metrics. It internally uses the model factory to construct attention-based models (GPT-2, LLaMA, RWKV, xLSTM) and provides comprehensive training metrics.

## Features

- **Model Factory Integration**: Automatically constructs models using the model registry
- **Cross-Entropy Loss**: Computes loss with optional label smoothing
- **Comprehensive Metrics**:
  - Token-level accuracy
  - Top-5 accuracy
  - Perplexity
  - Valid token counting
- **Padding Handling**: Properly ignores padding tokens in loss and metrics
- **Text Generation**: Supports autoregressive generation with temperature, top-k, and top-p sampling
- **OmegaConf Integration**: Load configurations from YAML/dict seamlessly

## Architecture

```
AttentionCausalLoss
├── Model (constructed via factory)
│   ├── GPT2CausalLM
│   ├── LlamaCausalLM
│   ├── RwkvCausalLM
│   └── xLSTMCausalLM
├── Loss Function (CrossEntropyLoss)
└── Metrics Computation
    ├── Accuracy
    ├── Top-5 Accuracy
    ├── Perplexity
    └── Token Counting
```

## Basic Usage

### Creating from Configuration

The `from_config()` method supports two configuration formats:

#### Format 1: Nested Structure (Recommended)

```python
from trigor.models.attentionCausalLoss import AttentionCausalLoss

# Configuration dictionary with nested structure
config = {
    'model_config': {
        'type': 'GPT2CausalLM',
        'config': {
            'vocab_size': 259,
            'hidden_size': 256,
            'num_layers': 6,
            'num_heads': 8,
            'max_seq_len': 2048,
        }
    },
    'ignore_index': 256,  # PAD token
    'label_smoothing': 0.1,
}

# Create loss module
loss_module = AttentionCausalLoss.from_config(config)

print(loss_module)
# Output:
# AttentionCausalLoss(
#   model_type=GPT2CausalLM,
#   parameters=5,329,664,
#   ignore_index=256,
#   label_smoothing=0.1
# )
```

#### Format 2: Flat Structure (Backward Compatible)

```python
# Flat configuration format (still supported)
config = {
    'model_type': 'GPT2CausalLM',
    'model_config': {
        'vocab_size': 259,
        'hidden_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
    },
    'ignore_index': 256,
    'label_smoothing': 0.1,
}

loss_module = AttentionCausalLoss.from_config(config)
```

### Forward Pass with Loss and Metrics

```python
import torch

# Prepare data
batch_size = 4
seq_len = 512
input_ids = torch.randint(0, 259, (batch_size, seq_len))
labels = torch.randint(0, 259, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

# Forward pass
outputs = loss_module(input_ids, labels, attention_mask)

# Access metrics
loss = outputs['loss']              # Cross-entropy loss
accuracy = outputs['accuracy']      # Token-level accuracy
top5_acc = outputs['top5_accuracy'] # Top-5 accuracy
perplexity = outputs['perplexity']  # Perplexity metric
num_tokens = outputs['num_tokens']  # Number of valid tokens

print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")
print(f"Perplexity: {perplexity.item():.2f}")
```

### Text Generation

```python
# Start with START token + '['
start_tokens = torch.tensor([[257, 91]])

# Generate with temperature sampling
generated = loss_module.generate(
    start_tokens,
    max_length=50,
    temperature=0.8,
    top_k=50,
)

print(f"Generated: {generated.shape}")
# Output: Generated: torch.Size([1, 51])
```

## Configuration Options

### Model Types

Supported model types (must be registered in model registry):
- `GPT2CausalLM`: Standard GPT-2 with multi-head attention
- `LlamaCausalLM`: LLaMA with grouped query attention
- `RwkvCausalLM`: RWKV with linear attention
- `xLSTMCausalLM`: xLSTM with matrix-valued cell states

### Loss Configuration

```python
config = {
    'model_type': 'GPT2CausalLM',
    'model_config': {...},

    # Loss settings
    'ignore_index': 256,       # Token ID to ignore (PAD token)
    'label_smoothing': 0.1,    # Label smoothing (0.0 = no smoothing)
}
```

**Label Smoothing**: Reduces overfitting by softening hard labels. Typical values: 0.0-0.2.

## Training Example

```python
from torch.utils.data import DataLoader
from trigor.data import TGNDataset

# Load dataset
dataset = TGNDataset.from_config(data_config)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=TGNDataset.collate_batch)

# Create loss module
loss_module = AttentionCausalLoss.from_config(model_config)
optimizer = torch.optim.AdamW(loss_module.parameters(), lr=1e-4)

# Training loop
loss_module.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        # Forward pass
        outputs = loss_module(input_ids, labels, attention_mask)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"Acc: {outputs['accuracy'].item():.4f}")
```

## Metrics Explanation

### Token-Level Accuracy

Percentage of correctly predicted tokens (excluding padding):

```
accuracy = (correct_predictions & valid_tokens).sum() / valid_tokens.sum()
```

### Top-5 Accuracy

Percentage of times the correct token appears in top-5 predictions:

```python
top5_predictions = torch.topk(logits, k=5).indices
top5_accuracy = (top5_predictions == labels).any(dim=-1).mean()
```

### Perplexity

Exponential of the cross-entropy loss:

```
perplexity = exp(loss)
```

Lower perplexity indicates better model performance. Random baseline: ~259 (vocab size).

## Generation Methods

### Temperature Sampling

```python
generated = loss_module.generate(
    start_tokens,
    max_length=100,
    temperature=0.8,  # Lower = more conservative, higher = more random
)
```

### Top-K Sampling

```python
generated = loss_module.generate(
    start_tokens,
    max_length=100,
    temperature=0.9,
    top_k=50,  # Consider only top-50 most likely tokens
)
```

### Top-P (Nucleus) Sampling

```python
generated = loss_module.generate(
    start_tokens,
    max_length=100,
    temperature=0.9,
    top_p=0.9,  # Consider tokens with cumulative probability < 0.9
)
```

## Working with Different Models

### GPT-2

```python
config = {
    'model_type': 'GPT2CausalLM',
    'model_config': {
        'vocab_size': 259,
        'hidden_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1,
        'activation': 'gelu_new',
    },
}
loss_module = AttentionCausalLoss.from_config(config)
```

### LLaMA with Grouped Query Attention

```python
config = {
    'model_type': 'LlamaCausalLM',
    'model_config': {
        'vocab_size': 259,
        'hidden_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'num_key_value_heads': 2,  # GQA with 4 groups
        'max_seq_len': 2048,
    },
}
loss_module = AttentionCausalLoss.from_config(config)
```

### RWKV with Linear Attention

```python
config = {
    'model_type': 'RwkvCausalLM',
    'model_config': {
        'vocab_size': 259,
        'hidden_size': 256,
        'num_layers': 6,
        'max_seq_len': 2048,
    },
}
loss_module = AttentionCausalLoss.from_config(config)
```

### xLSTM

```python
config = {
    'model_type': 'xLSTMCausalLM',
    'model_config': {
        'vocab_size': 259,
        'hidden_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'chunk_size': 64,
        'max_seq_len': 2048,
    },
}
loss_module = AttentionCausalLoss.from_config(config)
```

## OmegaConf Integration

### From YAML File

#### Nested Format (Recommended)

```python
from omegaconf import OmegaConf

# Load config from YAML
config = OmegaConf.load('configs/loss_config.yaml')
loss_module = AttentionCausalLoss.from_config(config)
```

Example YAML with nested structure (`configs/loss_config.yaml`):

```yaml
model_config:
  type: GPT2CausalLM
  config:
    vocab_size: 259
    hidden_size: 256
    num_layers: 6
    num_heads: 8
    max_seq_len: 2048
ignore_index: 256
label_smoothing: 0.1
```

#### Flat Format (Backward Compatible)

```yaml
model_type: GPT2CausalLM
model_config:
  vocab_size: 259
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  max_seq_len: 2048
ignore_index: 256
label_smoothing: 0.1
```

### From Training Config

The training configs use nested structure with an additional `config` wrapper:

```python
# Load full training config
cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')

# Create model directly from config (nested structure is automatically detected)
loss_module = AttentionCausalLoss.from_config(cfg.model.config)

# Or use the model factory
from trigor.models import make_model
loss_module = make_model(cfg.model.type, cfg.model.config)
```

Training config format:
```yaml
model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: GPT2CausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        # ...
    ignore_index: 256
    label_smoothing: 0.1
```

## Utility Methods

### Get Model Information

```python
info = loss_module.get_model_info()
print(info)
# Output:
# {
#   'model_type': 'GPT2CausalLM',
#   'ignore_index': 256,
#   'label_smoothing': 0.1,
#   'model_info': {
#     'model_type': 'gpt2',
#     'total_parameters': 5329664,
#     ...
#   }
# }
```

### Count Parameters

```python
params = loss_module.count_parameters()
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")
# Output:
# Total: 5,329,664
# Trainable: 5,329,664
```

## Advanced Usage

### Custom Loss Function

To use a different loss function:

```python
class CustomAttentionCausalLoss(AttentionCausalLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction='sum',  # Sum instead of mean
        )
```

### Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = loss_module(batch['input_ids'], batch['labels'], batch['attention_mask'])
    loss = outputs['loss'] / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = loss_module(batch['input_ids'], batch['labels'], batch['attention_mask'])
        loss = outputs['loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Testing

Run the comprehensive test suite:

```bash
python tests/test_attention_causal_loss.py
```

Tests cover:
- Loss module creation for all model types
- Forward pass with loss and metrics
- Different model architectures
- Label smoothing functionality
- Text generation
- OmegaConf integration

## Performance Considerations

### Memory Usage

Model parameters by architecture (hidden_size=256, num_layers=6):
- GPT-2: ~5.3M parameters
- LLaMA (GQA): ~4.3M parameters
- RWKV: ~5.3M parameters
- xLSTM: ~5.0M parameters

### Computational Cost

Forward pass (batch_size=8, seq_len=2048):
- GPT-2: ~92 MB memory, ~0.5s per batch (CPU)
- LLaMA: ~85 MB memory, ~0.4s per batch (CPU)
- RWKV: ~90 MB memory, ~0.3s per batch (CPU)

## Troubleshooting

### Issue: "Unknown model type"

**Error**: `ValueError: Unknown model type 'GPT2'. Available: GPT2CausalLM, ...`

**Solution**: Use the full model class name (e.g., `GPT2CausalLM` not `GPT2`)

### Issue: High perplexity

**Cause**: Untrained model or poor data quality

**Solution**:
- Train for more epochs
- Check data preprocessing
- Verify tokenization is correct

### Issue: Low accuracy

**Cause**: Random predictions from untrained model

**Solution**: Expected for initial training. Accuracy should improve with training.

### Issue: NaN loss

**Cause**: Gradient explosion or invalid inputs

**Solution**:
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Check for NaN in input data
- Reduce learning rate

## Related Documentation

- [Model Registry](../trigor/models/registry.py) - Model factory implementation
- [CausalLM Models](../trigor/models/) - Individual model implementations
- [TGNDataset](../trigor/data/tgn_dataset.py) - Dataset for training
- [Training Configs](../configs/training/) - Example training configurations

## Example Output

```
AttentionCausalLoss(
  model_type=GPT2CausalLM,
  parameters=5,329,664,
  ignore_index=256,
  label_smoothing=0.1
)

Forward pass outputs:
  Loss: 5.5752
  Accuracy: 0.0047
  Top-5 Accuracy: 0.0234
  Perplexity: 263.80
  Num valid tokens: 214
  Logits shape: torch.Size([4, 64, 259])
```

## Summary

The `AttentionCausalLoss` module provides a complete solution for causal language modeling:

✓ **Easy Integration**: Single-line model creation from config
✓ **Multiple Metrics**: Loss, accuracy, top-5 accuracy, perplexity
✓ **Flexible**: Supports 4 different model architectures
✓ **Production Ready**: Label smoothing, generation, padding handling
✓ **Well Tested**: Comprehensive test suite covering all features
