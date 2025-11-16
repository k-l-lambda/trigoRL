# Known Issues and Solutions

## Issue 1: Dynamic Sequence Length Not Supported

**Error:**
```
Non-zero status code returned while running Reshape node. Name:'/model/transformer/Reshape_1'
The input tensor cannot be reshaped to the requested shape.
Input shape:{64}, requested shape:{256,1}
```

**Cause:**
GPT-2's position embedding implementation contains Reshape operations that get hardcoded to the export-time sequence length (256) during ONNX conversion. This is a known limitation of PyTorch's ONNX exporter with transformer models, even when using `--dynamic-seq` flag.

**Impact:**
- ✅ Fixed-length inference (seq_len=256): Works perfectly
- ❌ Variable sequence lengths (64, 128, etc.): Not supported
- ❌ Autoregressive generation: Not supported (requires seq_len starting from 1)

**Workaround:**
1. **Use fixed sequence length of 256** (recommended for current tests)
2. **Re-export for different lengths**: Export model with desired sequence length
3. **Pad inputs**: Pad shorter sequences to 256 tokens

**Future Solutions:**
- Use torch.export-based ONNX exporter (PyTorch 2.9+) with `dynamo=True`
- Implement custom ONNX graph transformations to make Reshape nodes dynamic
- Switch to models with learnable position embeddings (e.g., RoPE in LLaMA)

**Current Test Configuration:**
```javascript
const CONFIG = {
    modelPath: 'GPT2CausalLM_ep0015_int8.onnx',
    tests: {
        batchSizes: [1],     // Fixed batch size
        seqLengths: [256],   // Fixed sequence length
        generation: false,    // Disabled (requires variable seq length)
    }
};
```

## Issue 2: bfloat16 Not Supported

**Error:**
```
Type Error: Type 'tensor(bfloat16)' of input parameter ... is invalid
```

**Cause:**
The model was exported with `dtype: bfloat16` in the training config, but onnxruntime-node doesn't support bfloat16 tensors well.

**Solution:**

### Option 1: Re-export with float32 (Recommended)

Modify the checkpoint config temporarily or export with float32:

```bash
# Method A: Edit config before export
# In outputs/trigor/.../config.yaml, change:
#   dtype: bfloat16
# to:
#   dtype: float32

# Then export
python exportOnnx.py outputs/trigor/YOUR_TRAINING_DIR/

# Method B: Use a different checkpoint that was trained with float32
python exportOnnx.py outputs/trigor/FLOAT32_TRAINING_DIR/
```

### Option 2: Use Python onnxruntime

Python's onnxruntime has better bfloat16 support:

```bash
cd tests
python test_onnx_export.py
```

### Option 3: Convert Model After Export

Use ONNX tools to convert bfloat16 to float32:

```python
import onnx
from onnx import numpy_helper

model = onnx.load("model_bf16.onnx")

# Convert bfloat16 to float32
# ... conversion code ...

onnx.save(model, "model_fp32.onnx")
```

## Recommended Workflow

For Node.js inference:
1. **Train with float32 or mixed precision that ends in float32**
2. **Export with `dtype: float32` in config**
3. **Use fixed sequence length (256)**
4. **Test with Node.js onnxruntime**

For maximum compatibility:
- Use float32 for production models
- Use bfloat16/float16 for training efficiency
- Convert to float32 for deployment
- Export with fixed dimensions that match your use case

## Testing Different Model Types

```javascript
// In test_inference.js
const CONFIG = {
	// Update to your model path
	modelPath: '../../outputs/trigor/YOUR_MODEL/model_float32.onnx',
	vocabSize: 259,
	tests: {
		batchSizes: [1],    // Fixed batch
		seqLengths: [256],  // Fixed seq length
	}
};
```

## Platform Support Matrix

| Platform | float32 | float16 | bfloat16 | Dynamic Seq |
|----------|---------|---------|----------|-------------|
| Python onnxruntime | ✓ | ✓ | ✓ | ✗ (GPT-2 limitation) |
| Node.js onnxruntime | ✓ | ✓ | ✗ | ✗ (GPT-2 limitation) |
| Browser onnxruntime-web | ✓ | Partial | ✗ | ✗ (GPT-2 limitation) |

## Future Work

- Add automatic dtype conversion in export script
- Provide conversion utility for existing models
- Document dtype selection best practices
- Investigate alternative export methods for dynamic sequence support
- Test with LLaMA/RWKV models (RoPE may handle dynamic seq better)
