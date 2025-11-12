# Training Configurations

This directory contains training configuration files for different model architectures trained on Trigo game data.

## Configuration Structure

All training configs now use `AttentionCausalLoss` as a wrapper around the base models, which automatically computes loss and metrics during forward pass.

### Configuration Format (Nested Structure)

```yaml
model:
  type: AttentionCausalLoss  # Wrapper that computes loss + metrics
  config:                    # Wrapper configuration
    model_config:            # Inner model specification
      type: GPT2CausalLM     # Underlying model architecture
      config:                # Architecture-specific parameters
        vocab_size: 259
        hidden_size: 256
        # ... other model parameters
    ignore_index: 256        # PAD token to ignore in loss
    label_smoothing: 0.1     # Regularization factor
```

**Benefits:**
- Automatic loss computation with label smoothing
- Built-in metrics: accuracy, top-5 accuracy, perplexity
- Proper padding handling (ignores PAD tokens)
- Simplified training code
- Consistent `type` + `config` pattern at each hierarchy level

See [Configuration Migration Guide](../../docs/config_migration.md) for details.

## Available Configurations

### 1. `trigo-gpt2.yaml` - GPT-2 Baseline
**Architecture:** Standard transformer decoder with multi-head attention (MHA)

**Key Features:**
- GELU activation
- Learned positional embeddings
- Most straightforward implementation
- Good educational baseline

**Model Parameters:**
- Hidden size: 256
- Layers: 6
- Attention heads: 8
- Total parameters: ~5.3M

**Training Settings:**
- Learning rate: 1e-4
- Batch size: 8
- Warmup steps: 1000
- Label smoothing: 0.1

**Usage:**
```bash
python train.py training=trigo-gpt2
```

**Verification:**
```python
from omegaconf import OmegaConf
from trigor.models import make_model

cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')
# Pass cfg.model.config to get the wrapper configuration
model = make_model(cfg.model.type, cfg.model.config)
print(model.count_parameters())  # 5,329,664 parameters
```

---

### 2. `trigo-llama.yaml` - LLaMA with GQA
**Architecture:** LLaMA with Grouped Query Attention for efficient inference

**Key Features:**
- RoPE (Rotary Position Embedding)
- RMSNorm instead of LayerNorm
- SiLU activation
- **Configurable attention:** MHA/GQA/MQA switching
- GQA reduces KV cache for faster inference

**Model Parameters:**
- Hidden size: 256
- Layers: 6
- Query heads: 8
- Key/Value heads: 2 (GQA with 4 groups)
- Total parameters: ~4.3M (with GQA)

**Training Settings:**
- Learning rate: 3e-4 (higher than GPT-2)
- Batch size: 8
- Weight decay: 0.1 (stronger regularization)
- Warmup steps: 1000

**Attention Configuration:**
- `num_key_value_heads: 8` → MHA (Multi-Head Attention)
- `num_key_value_heads: 2` → GQA (4 groups) **[Default]**
- `num_key_value_heads: 1` → MQA (Multi-Query Attention)

**Usage:**
```bash
python train.py training=trigo-llama
# Or override for MHA:
python train.py training=trigo-llama model.num_key_value_heads=8
```

---

### 3. `trigo-rwkv.yaml` - Linear Attention
**Architecture:** RWKV with linear attention mechanism

**Key Features:**
- Linear complexity: O(N·D²) instead of O(N²·D)
- No softmax - uses exponential time decay
- Time-mixing and channel-mixing blocks
- Constant memory during inference
- Excellent for long sequences (10K+ tokens)

**Model Parameters:**
- Hidden size: 256
- Layers: 6
- Total parameters: ~5.3M

**Training Settings:**
- Learning rate: 6e-4 (higher for RWKV)
- Batch size: 8
- Warmup steps: 500 (shorter warmup)

**Advantages:**
- Memory efficient for long sequences
- Linear attention complexity
- Recurrent state for autoregressive generation

**Usage:**
```bash
python train.py training=trigo-rwkv
```

---

### 4. `trigo-xlstm.yaml` - Extended LSTM
**Architecture:** xLSTM with matrix-valued cell states

**Key Features:**
- Matrix-valued cell states (vs scalar in standard LSTM)
- Exponential gating in log-space
- Chunk-wise parallelization
- Multi-head architecture

**Model Parameters:**
- Hidden size: 256
- Layers: 6
- Heads: 8
- Chunk size: 64
- Total parameters: ~5.0M

**Training Settings:**
- Learning rate: 5e-4
- Batch size: 8
- Warmup steps: 1000

**Special Parameters:**
- `chunk_size`: Controls parallelization granularity (64 is default)
- `qk_dim_factor`: Query/Key dimension scaling (0.5 = 128 dims)
- `v_dim_factor`: Value dimension scaling (1.0 = 256 dims)
- `mode`: 'train' or 'inference'

**Note:** Forward pass may have kernel issues in some transformers versions, but model creation and parameter counting work correctly.

**Usage:**
```bash
python train.py training=trigo-xlstm
```

---

## Model Comparison

| Model | Attention Type | Complexity | Parameters | Memory | Best For |
|-------|----------------|------------|------------|--------|----------|
| **GPT-2** | MHA | O(N²·D) | 5.3M | High | Baseline, educational |
| **LLaMA** | GQA | O(N²·D) | 4.3M | Medium | Most flexible, efficient |
| **RWKV** | Linear | O(N·D²) | 5.3M | Low | Long sequences |
| **xLSTM** | Recurrent | O(N·D²) | 5.0M | Low | Recurrent alternative |

## Common Configuration Structure

All training configs share the same structure:

```yaml
# Paths
paths:
  root: .
  data: ${paths.root}/data
  output: ${paths.root}/outputs

# Data configuration
data:
  type: TGNDataset
  data_dir: ...
  max_length: 2048
  loader:
    batch_size: 8
    shuffle: true
    num_workers: 4
    pin_memory: true

# Model configuration
model:
  type: gpt2|llama|rwkv|xlstm
  vocab_size: 259  # TGN byte tokenizer
  hidden_size: 256
  num_layers: 6
  # ... model-specific parameters

# Training configuration
training:
  epochs: 100
  learning_rate: ...
  weight_decay: ...
  warmup_steps: ...
  scheduler:
    type: cosine
    min_lr: ...
  save_frequency: 10
  save_dir: ${paths.output}/checkpoints/{model}
  wandb:
    enabled: false
    project: trigor
    name: trigo-{model}

# Evaluation
eval:
  eval_frequency: 5
  eval_batches: 50

# Device
device: cuda
seed: 42
deterministic: true
```

## CLI Overrides

Hydra allows easy configuration overrides from command line:

```bash
# Change batch size
python train.py training=trigo-gpt2 data.loader.batch_size=16

# Change learning rate
python train.py training=trigo-llama training.learning_rate=5e-4

# Use different dataset
python train.py training=trigo-rwkv data.data_dir=/path/to/data

# Enable wandb logging
python train.py training=trigo-xlstm training.wandb.enabled=true

# Multiple overrides
python train.py training=trigo-llama \
    data.loader.batch_size=16 \
    training.learning_rate=5e-4 \
    model.num_key_value_heads=1  # Switch to MQA
```

## Experiment Tracking

Each config has WandB settings configured but disabled by default:

```yaml
wandb:
  enabled: false  # Set to true to enable
  project: trigor
  entity: null    # Set your WandB entity
  name: trigo-{model}
  tags: [model_type, attention_type, ...]
```

To enable WandB:
```bash
python train.py training=trigo-gpt2 training.wandb.enabled=true
```

## Checkpointing

All configs save checkpoints to separate directories:
- GPT-2: `outputs/checkpoints/gpt2/`
- LLaMA: `outputs/checkpoints/llama/`
- RWKV: `outputs/checkpoints/rwkv/`
- xLSTM: `outputs/checkpoints/xlstm/`

Checkpoint settings:
- `save_frequency: 10` - Save every 10 epochs
- `keep_n_checkpoints: 5` - Keep only 5 best checkpoints
- `save_mode: best` - Save based on best validation loss

## Choosing a Configuration

**For getting started:**
→ Use `trigo-gpt2.yaml` - most straightforward baseline

**For production/efficiency:**
→ Use `trigo-llama.yaml` - best balance of performance and efficiency

**For long sequences:**
→ Use `trigo-rwkv.yaml` - linear complexity, memory efficient

**For recurrent model research:**
→ Use `trigo-xlstm.yaml` - modern LSTM alternative

## Custom Configurations

To create a custom config:

1. Copy an existing config:
```bash
cp trigo-gpt2.yaml trigo-custom.yaml
```

2. Modify parameters as needed

3. Run with your custom config:
```bash
python train.py training=trigo-custom
```

## Notes

- All configs use the same TGN dataset (259 token vocab)
- Vocab size must match tokenizer (259 for TGN byte tokenizer)
- Batch size of 8 fits on most GPUs with these model sizes
- Gradient accumulation can be used to increase effective batch size
- All models support mixed precision training (FP16/BF16)
- Deterministic mode ensures reproducibility (may be slightly slower)
