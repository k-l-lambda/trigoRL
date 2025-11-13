# Language Model Training Guide

This guide explains how to train attention-based language models on TGN (Trigo Game Notation) data.

## Quick Start

### Basic Training

Train with default GPT-2 configuration:

```bash
python train_lm.py
```

### Using Different Models

Train with Llama (GQA attention):

```bash
python train_lm.py training=trigo-llama
```

Train with RWKV (linear attention):

```bash
python train_lm.py training=trigo-rwkv
```

Train with xLSTM:

```bash
python train_lm.py training=trigo-xlstm
```

### Enable Wandb Logging

```bash
python train_lm.py training.wandb.enabled=true
```

### Custom Hyperparameters

```bash
python train_lm.py \
    training.epochs=50 \
    training.learning_rate=5e-5 \
    data.loader.batch_size=16 \
    training.wandb.enabled=true
```

### Resume from Checkpoint

```bash
python train_lm.py resume_from=outputs/checkpoints/gpt2/latest.chkpt
```

## Configuration Structure

Training configurations are located in `configs/training/`:

- `trigo-gpt2.yaml` - GPT-2 with multi-head attention (baseline)
- `trigo-llama.yaml` - Llama with grouped query attention (most efficient)
- `trigo-rwkv.yaml` - RWKV with linear attention (long sequences)
- `trigo-xlstm.yaml` - xLSTM with extended LSTM architecture

### Configuration Sections

#### 1. Paths

```yaml
paths:
  root: .
  data: ${paths.root}/data
  output: ${paths.root}/outputs
```

#### 2. Data Configuration

```yaml
data:
  type: TGNDataset
  data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output
  max_length: 8192
  min_length: 10
  max_file_size: 10000

  # Train/validation split using phase-based deterministic assignment
  train_split: "*0..7/10"  # 80% training (shuffled)
  val_split: "8,9/10"      # 20% validation (no shuffle)

  loader:
    batch_size: 8
    shuffle: true
    num_workers: 4
    pin_memory: true
```

**Split Format:**
- Format: `"[*]phases/cycle"`
- `*` prefix enables shuffling
- Phases use range syntax: `"0..7"` expands to `[0,1,2,3,4,5,6,7]`
- Mixed syntax: `"0..3,7,8/10"` → `[0,1,2,3,7,8]`

#### 3. Model Configuration

```yaml
model:
  type: AttentionCausalLoss  # Wrapper with loss + metrics
  config:
    model_config:
      type: GPT2CausalLM  # Or LlamaCausalLM, RwkvCausalLM, xLSTMCausalLM
      config:
        vocab_size: 259
        hidden_size: 256
        num_layers: 6
        num_heads: 8
        max_seq_len: 8192
        dropout: 0.1
    ignore_index: 256       # PAD token
    label_smoothing: 0.1    # Regularization
```

#### 4. Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1

  scheduler:
    type: cosine  # cosine, linear, constant
    min_lr: 1e-6

  # Checkpointing
  save_frequency: 10
  save_dir: ${paths.output}/checkpoints/gpt2
  keep_n_checkpoints: 5
  save_mode: best  # 'best', 'all', or 'latest'

  # Monitoring
  monitor:
    field: val_loss
    mode: min  # 'min' for loss, 'max' for accuracy

  # Logging
  log_frequency: 100
  wandb:
    enabled: false
    project: trigor
    entity: null
    name: trigo-gpt2
    tags: [gpt2, mha, baseline]
```

#### 5. Evaluation Configuration

```yaml
eval:
  eval_frequency: 5      # Evaluate every N epochs
  eval_batches: 50       # Limit validation batches
```

## Training Features

### Automatic Features

- **Deterministic Seeding**: Reproducible results with fixed seed
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Warmup + cosine annealing
- **Checkpointing**: Saves best and latest checkpoints
- **Progress Bars**: tqdm progress tracking with metrics
- **Automatic Cleanup**: Keeps only best N checkpoints

### Metrics Logged

**Training Metrics:**
- Loss (cross-entropy)
- Accuracy (token-level)
- Perplexity (exp(loss))
- Top-5 Accuracy
- Learning Rate

**Validation Metrics:**
- Validation Loss
- Validation Accuracy
- Validation Perplexity
- Validation Top-5 Accuracy

### Checkpointing

Checkpoints are saved in `outputs/checkpoints/{model_name}/`:

- `latest.chkpt` - Most recent epoch (always saved)
- `best_ep{N}_{metric}_{value}.chkpt` - Best checkpoints based on val_loss

**Checkpoint Contents:**
```python
{
    'epoch': current_epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_metric': best_val_metric,
    'config': config_dict,
}
```

## Model Architectures

### GPT-2 (Multi-Head Attention)

**Parameters:** ~6.9M
**Attention:** Standard MHA
**Best for:** Baseline comparisons

```bash
python train_lm.py training=trigo-gpt2
```

### Llama (Grouped Query Attention)

**Parameters:** ~4.3M
**Attention:** GQA (fewer KV heads)
**Best for:** Memory efficiency, inference speed

```bash
python train_lm.py training=trigo-llama
```

### RWKV (Linear Attention)

**Parameters:** ~5.3M
**Attention:** Linear (O(n) vs O(n²))
**Best for:** Long sequences, fast inference

```bash
python train_lm.py training=trigo-rwkv
```

### xLSTM (Extended LSTM)

**Parameters:** ~5.0M
**Attention:** None (LSTM-based)
**Best for:** Sequential modeling, alternative to attention

```bash
python train_lm.py training=trigo-xlstm
```

## Monitoring Training

### Console Output

Training displays real-time metrics in progress bars:

```
Epoch 1/100 [Train]:  50%|█████     | 19/38 [01:54<01:54, 6.02s/it,
    loss=5.2158, acc=0.0912, ppl=184.15, lr=2.78e-06]
```

### Wandb Dashboard

When enabled, view training in real-time at https://wandb.ai:

```bash
python train_lm.py training.wandb.enabled=true
```

Dashboard includes:
- Loss curves (train/val)
- Accuracy trends
- Perplexity over time
- Learning rate schedule
- Model gradients and parameters
- System metrics (GPU, CPU, memory)

### Epoch Summary

After each epoch:

```
Epoch 1/100 Summary:
  Train Loss: 5.1936
  Train Accuracy: 0.1053
  Train Perplexity: 185.20
  Val Loss: 4.6997
  Val Accuracy: 0.1739
  Val Perplexity: 109.99
```

## Tips and Best Practices

### Finding Optimal Batch Size

Start with smaller batch size and increase until OOM:

```bash
# Test different batch sizes
python train_lm.py data.loader.batch_size=4
python train_lm.py data.loader.batch_size=8
python train_lm.py data.loader.batch_size=16
python train_lm.py data.loader.batch_size=32
```

### Gradient Accumulation

Simulate larger batch sizes with gradient accumulation:

```bash
# Effective batch size = 8 * 4 = 32
python train_lm.py \
    data.loader.batch_size=8 \
    training.gradient_accumulation_steps=4
```

### Learning Rate Tuning

Rule of thumb: scale LR with effective batch size

```bash
# For larger batches, use larger LR
python train_lm.py \
    data.loader.batch_size=32 \
    training.learning_rate=2e-4
```

### Warmup Steps

Typical: 1-10% of total training steps

```bash
# For 100 epochs * 38 batches = 3800 steps
# Use 380 warmup steps (10%)
python train_lm.py training.warmup_steps=380
```

### Early Stopping

Monitor validation loss and stop manually if no improvement:

1. Watch `val_loss` in console output
2. If no improvement for many epochs, stop training (Ctrl+C)
3. Trainer will save checkpoint before exit
4. Best checkpoint is already saved

### CPU Training

For testing without GPU:

```bash
python train_lm.py device=cpu data.loader.batch_size=2
```

## Troubleshooting

### CUDA Out of Memory

**Solutions:**
1. Reduce batch size: `data.loader.batch_size=4`
2. Reduce sequence length: `data.max_length=4096`
3. Use gradient accumulation
4. Reduce model size (num_layers, hidden_size)

### Slow Training

**Solutions:**
1. Increase `num_workers`: `data.loader.num_workers=8`
2. Enable `pin_memory`: `data.loader.pin_memory=true`
3. Reduce validation frequency: `eval.eval_frequency=10`
4. Limit validation batches: `eval.eval_batches=20`

### Loss Not Decreasing

**Solutions:**
1. Check learning rate (too high/low)
2. Increase warmup steps
3. Reduce label smoothing
4. Check data quality (min_length, max_file_size filters)

### NaN Loss

**Solutions:**
1. Reduce learning rate
2. Enable gradient clipping (max_grad_norm)
3. Check for corrupted data files
4. Reduce label smoothing

## Example Workflows

### Quick Experiment

```bash
# Fast iteration: 10 epochs, small batch, frequent validation
python train_lm.py \
    training.epochs=10 \
    data.loader.batch_size=4 \
    eval.eval_frequency=1 \
    training.wandb.enabled=true \
    training.wandb.name=quick-experiment
```

### Full Training Run

```bash
# Production: 100 epochs, optimized settings
python train_lm.py \
    training.epochs=100 \
    data.loader.batch_size=16 \
    training.gradient_accumulation_steps=2 \
    eval.eval_frequency=5 \
    training.save_frequency=10 \
    training.wandb.enabled=true \
    training.wandb.name=gpt2-full-run
```

### Hyperparameter Search

```bash
# Test different LRs
for lr in 5e-5 1e-4 2e-4; do
    python train_lm.py \
        training.learning_rate=$lr \
        training.epochs=20 \
        training.wandb.enabled=true \
        training.wandb.name=gpt2-lr-${lr}
done
```

### Model Comparison

```bash
# Compare all architectures
for model in trigo-gpt2 trigo-llama trigo-rwkv trigo-xlstm; do
    python train_lm.py \
        training=$model \
        training.epochs=50 \
        training.wandb.enabled=true
done
```

## Next Steps

After training:

1. **Evaluate Model**: Load best checkpoint and run evaluation
2. **Export to ONNX**: Convert for deployment
3. **Integrate with Game Engine**: Use for AI player inference
4. **Visualize Attention**: Analyze what model learned
5. **Fine-tune**: Continue training with adjusted hyperparameters

## Files Reference

- **Trainer**: `trigor/training/lm_trainer.py`
- **Entry Point**: `train_lm.py`
- **Configs**: `configs/training/trigo-*.yaml`
- **Models**: `trigor/models/` (GPT2, Llama, RWKV, xLSTM)
- **Dataset**: `trigor/data/tgn_dataset.py`
- **Utilities**: `trigor/utils/` (logger, checkpoint)
