# ONNX Export Guide

This guide explains how to export trained TrigoRL models to ONNX format for deployment.

## Quick Start

```bash
# Export latest checkpoint with default settings
python exportOnnx.py training_output/trigo-gpt2-20250115_120000

# Export best checkpoint
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 --checkpoint best

# Export with dynamic batch and sequence sizes (RECOMMENDED)
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --dynamic-batch --dynamic-seq
```

## Best Practices

### 1. Always Use Dynamic Sequence Length

**CRITICAL**: Always export with `--dynamic-seq` for production use.

Fixed sequence length models require padding all inputs to the same length, which:
- Wastes computation on padding tokens
- Can cause 10-20x performance degradation for short sequences
- May cause errors if input exceeds the fixed length

```bash
# ✓ RECOMMENDED: Dynamic sequence length
python exportOnnx.py training_output/model --dynamic-seq

# ✗ AVOID: Fixed sequence length (unless you have a specific reason)
python exportOnnx.py training_output/model --seq-len 256
```

### 2. Choose the Right Export Mode

TrigoRL supports three export modes for different use cases:

| Mode | Files Generated | Use Case |
|------|-----------------|----------|
| `tree` | Single ONNX file | TypeScript tree-based move generation |
| `evaluation` | Single ONNX file | TypeScript position evaluation |
| `shared` | 3 ONNX files (base + heads) | C++ MCTS with shared backbone |

```bash
# For TypeScript (trigo-web)
python exportOnnx.py training_output/model --mode tree --dynamic-seq
python exportOnnx.py training_output/model --mode evaluation --dynamic-seq

# For C++ (trigo.cpp MCTS)
python exportOnnx.py training_output/model --mode shared --dynamic-seq
```

### 3. Export All Three Modes for Full Compatibility

For a complete deployment, export all modes:

```bash
CHECKPOINT="outputs/trigor/20251215-model/ep0045_val_loss_2.0877.chkpt"

# TypeScript tree agent
python exportOnnx.py $CHECKPOINT --mode tree --dynamic-seq

# TypeScript evaluation
python exportOnnx.py $CHECKPOINT --mode evaluation --dynamic-seq

# C++ AlphaZero MCTS
python exportOnnx.py $CHECKPOINT --mode shared --dynamic-seq
```

### 4. GPU Acceleration

For GPU inference, ensure CUDA provider is available:

```python
import onnxruntime as ort

# Check available providers
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider' for GPU

# Create session with GPU
session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### 5. Performance Considerations

- **Short sequences (< 50 tokens)**: ~2-4ms per inference on GPU
- **Medium sequences (50-200 tokens)**: ~5-20ms per inference on GPU
- **Long sequences (> 200 tokens)**: Transformer O(n²) complexity applies

For MCTS with 50-100 simulations, expect:
- Early game (short TGN): ~100-200ms per move
- Late game (long TGN): ~1-5s per move (depends on sequence length)

## Command Line Options

```
python exportOnnx.py <training_dir> [options]

Required:
  training_dir              Path to training output directory

Optional:
  --checkpoint NAME         Checkpoint to export: "latest", "best", or filename
                           (default: latest)
  --output PATH            Output ONNX file path
                           (default: auto-generated in training_dir)
  --batch-size N           Batch size for dummy input (default: 1)
  --seq-len N              Sequence length for dummy input (default: 256)
  --dynamic-batch          Enable dynamic batch size axis
  --dynamic-seq            Enable dynamic sequence length axis
  --opset-version N        ONNX opset version (default: 14)
```

## Examples

### 1. Basic Export

Export the latest checkpoint with default settings:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000
```

Output: `training_output/trigo-gpt2-20250115_120000/gpt2_ep0050.onnx`

### 2. Export Best Checkpoint

Export the checkpoint with the best validation metric:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 --checkpoint best
```

### 3. Export Specific Checkpoint

Export a specific checkpoint file:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --checkpoint ep0050_loss_0.1234.chkpt
```

### 4. Custom Output Path

Specify a custom output path:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --output models/my_model.onnx
```

### 5. Dynamic Batch and Sequence

Enable dynamic axes for flexible input sizes:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --dynamic-batch --dynamic-seq
```

This allows you to use different batch sizes and sequence lengths at inference time.

### 6. Long Sequence Models

Export for longer sequences:

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --seq-len 512 --dynamic-seq
```

## Using Exported Models

### Python Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx')

# Prepare input (batch_size=1, seq_len=256)
input_ids = np.random.randint(0, 259, (1, 256), dtype=np.int64)

# Run inference
outputs = session.run(['logits'], {'input_ids': input_ids})
logits = outputs[0]  # Shape: (1, 256, 259)

# Get predicted tokens
predicted_tokens = np.argmax(logits, axis=-1)
```

### Batch Inference

With dynamic axes enabled:

```python
# Variable batch sizes
for batch_size in [1, 2, 4, 8]:
    input_ids = np.random.randint(0, 259, (batch_size, 256), dtype=np.int64)
    outputs = session.run(['logits'], {'input_ids': input_ids})
    print(f"Batch {batch_size}: output shape {outputs[0].shape}")

# Variable sequence lengths
for seq_len in [64, 128, 256, 512]:
    input_ids = np.random.randint(0, 259, (1, seq_len), dtype=np.int64)
    outputs = session.run(['logits'], {'input_ids': input_ids})
    print(f"SeqLen {seq_len}: output shape {outputs[0].shape}")
```

## Training Directory Structure

The export script expects the following directory structure:

```
training_output/trigo-gpt2-20250115_120000/
├── config.yaml                   # Model configuration (required)
├── latest.chkpt                  # Latest checkpoint
├── ep0050_loss_0.1234.chkpt     # Best checkpoints
└── gpt2_ep0050.onnx             # Exported ONNX model (created by script)
```

## Checkpoint Types

- **latest**: Most recent checkpoint saved during training
- **best**: Checkpoint with the best validation metric
- **specific**: Any checkpoint file by name (e.g., `ep0050_loss_0.1234.chkpt`)

## Model Input/Output

### Input

- **Name**: `input_ids`
- **Type**: int64
- **Shape**: `(batch_size, sequence_length)`
- **Description**: Token IDs from vocabulary (0-258 for Trigo vocab)

### Output

- **Name**: `logits`
- **Type**: float32 (or configured dtype)
- **Shape**: `(batch_size, sequence_length, vocab_size)`
- **Description**: Raw logits for each token position

## Supported Models

The export script supports all TrigoRL model types:

- GPT-2 (GPT2CausalLM)
- LLaMA (LlamaCausalLM)
- RWKV (RwkvCausalLM)
- xLSTM (xLSTMCausalLM)

## Dependencies

Required packages (installed via requirements.txt):

```bash
pip install onnx>=1.14.0
pip install onnxscript>=0.5.0
pip install onnxruntime>=1.15.0
```

## Troubleshooting

### Model Too Large

If the ONNX model is too large, consider:

1. Using a smaller model architecture
2. Reducing sequence length with `--seq-len`
3. Using quantization (post-export)

### Dynamic Axes Not Working

If you get errors with dynamic axes:

1. Ensure you're using a recent ONNX opset version (`--opset-version 14` or higher)
2. Check that your inference code supports dynamic shapes
3. Try exporting without dynamic axes first

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install individually
pip install onnx onnxscript onnxruntime
```

### Fixed Sequence Length Performance Issues

**Symptom**: Model inference is extremely slow (10-20x expected time)

**Cause**: Using fixed sequence length models that pad all inputs

**Solution**: Re-export with `--dynamic-seq`:

```bash
python exportOnnx.py training_output/model --dynamic-seq
```

### C++ MCTS "vector::reserve" Error

**Symptom**: `std::exception: vector::reserve` error during value inference

**Cause**: The shared model inferencer expects sequences within certain bounds

**Solution**:
1. Ensure using dynamic sequence length models
2. Check that `shared_model_inferencer.cpp` uses dynamic `prefix_len` calculation
3. The default split is `prefix_len = max(1, total_seq_len / 2)`

### CUDA Device Not Found (C++)

**Symptom**: "CUDA failure 100: no CUDA-capable device is detected"

**Cause**: Wrong CUDA device ID or device not visible

**Solution**:
```bash
# Check available GPUs
nvidia-smi

# Set correct device
CUDA_VISIBLE_DEVICES=0 ./self_play_generator --model ...

# Or force CPU mode
TRIGO_FORCE_CPU=1 ./self_play_generator --model ...
```

### Slow Inference with Long Sequences

**Symptom**: Late-game MCTS moves taking 5-10+ seconds

**Cause**: Transformer attention is O(n²) with sequence length

**Understanding**: This is expected behavior, not a bug. For a 500-token sequence:
- Each attention layer computes 500×500 = 250,000 attention scores
- With 6 layers, that's 1.5M attention computations per forward pass

**Mitigation strategies**:
1. Use smaller board sizes for faster games
2. Reduce MCTS simulations for late-game moves
3. Consider implementing prefix caching (advanced)
4. Use linear attention architectures (RWKV, xLSTM)

### Model Input/Output Mismatch

**Symptom**: Shape mismatch errors during inference

**Check the model I/O**:
```python
import onnx
model = onnx.load('model.onnx')
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
```

For shared models, verify:
- `base_model.onnx`: inputs `prefix_ids`, `evaluated_ids`, `evaluated_mask`
- `policy_head.onnx`: input `hidden_states`, output `logits`
- `value_head.onnx`: input `hidden_states`, output `values`

## Advanced Usage

### Custom Opset Version

```bash
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --opset-version 15
```

### Multiple Exports

Export the same model with different configurations:

```bash
# Static export for production
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --output models/model_static.onnx

# Dynamic export for development
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --output models/model_dynamic.onnx \
    --dynamic-batch --dynamic-seq
```

## Integration with Trigo Game Engine

### TypeScript (trigo-web)

For tree-based move generation and evaluation:

```typescript
// tools/selfPlayGames.ts usage
npx ts-node tools/selfPlayGames.ts \
    --black /path/to/model_tree.onnx \
    --white /path/to/model_tree.onnx \
    --evaluator /path/to/model_evaluation.onnx \
    --gameCount 10 \
    --board 5x5x1
```

TypeScript uses:
- `*_tree.onnx` for `trigoTreeAgent.ts` - tree-based move generation with prefix tree
- `*_evaluation.onnx` for `trigoEvaluationAgent.ts` - position value estimation

### C++ (trigo.cpp AlphaZero MCTS)

For AlphaZero-style MCTS with neural network guidance:

```bash
# Build self_play_generator
cd trigo.cpp/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make

# Run with shared model (3-file format)
./self_play_generator \
    --num-games 100 \
    --board 5x5x1 \
    --black-policy alphazero \
    --white-policy alphazero \
    --model /path/to/model_shared \
    --mcts-simulations 50 \
    --output ./selfplay_data
```

The `--model` path should point to the directory containing:
- `base_model.onnx` - Shared transformer backbone
- `policy_head.onnx` - Move prediction head
- `value_head.onnx` - Position evaluation head

### Shared Model Architecture

The shared model architecture enables efficient inference:

```
Input Tokens
     │
     ▼
┌─────────────────┐
│  Base Model     │  (Transformer backbone)
│  base_model.onnx│
└─────────────────┘
     │
     │ Hidden States
     ▼
  ┌──┴──┐
  │     │
  ▼     ▼
┌─────┐ ┌─────┐
│Policy│ │Value│
│ Head │ │ Head│
└─────┘ └─────┘
  │       │
  ▼       ▼
Logits   Value
```

**Advantages**:
- Single forward pass through backbone for both policy and value
- Reduced memory usage compared to separate models
- Efficient batched inference in MCTS

### Python Backend

```python
# Load ONNX model for AI player
session = ort.InferenceSession('trigo_model.onnx')

def ai_move(board_state):
    # Convert board to token sequence
    input_ids = tokenize_board(board_state)

    # Run inference
    outputs = session.run(['logits'], {'input_ids': input_ids})

    # Decode move
    move = decode_move(outputs[0])
    return move
```

### Frontend (JavaScript)

```javascript
// Use onnxruntime-web for client-side AI
const session = await ort.InferenceSession.create('trigo_model.onnx');

async function aiMove(boardState) {
  const inputIds = tokenizeBoard(boardState);
  const outputs = await session.run({input_ids: inputIds});
  return decodeMove(outputs.logits);
}
```

## See Also

- `trigor/exportOnnx.py` - Export script with all options
- `tests/test_onnx_export.py` - Test suite for export functionality
- `third_party/trigo/trigo-web/tools/selfPlayGames.ts` - TypeScript self-play with ONNX
- `third_party/trigo.cpp/src/self_play_generator.cpp` - C++ AlphaZero MCTS
- `third_party/trigo.cpp/src/shared_model_inferencer.cpp` - C++ shared model inference
- ONNX Runtime documentation: https://onnxruntime.ai/
