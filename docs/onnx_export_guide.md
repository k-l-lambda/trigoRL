# ONNX Export Guide

This guide explains how to export trained TrigoRL models to ONNX format for deployment.

## Quick Start

```bash
# Export latest checkpoint with default settings
python exportOnnx.py training_output/trigo-gpt2-20250115_120000

# Export best checkpoint
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 --checkpoint best

# Export with dynamic batch and sequence sizes
python exportOnnx.py training_output/trigo-gpt2-20250115_120000 \
    --dynamic-batch --dynamic-seq
```

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

The exported ONNX models can be integrated with the Trigo game engine:

### Backend (Python)

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

- `tests/test_onnx_export.py` - Test suite for export functionality
- `examples/example_onnx_export.py` - Complete usage examples
- ONNX Runtime documentation: https://onnxruntime.ai/
