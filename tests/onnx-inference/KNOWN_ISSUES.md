# Known Issues and Solutions

## Issue: bfloat16 Not Supported

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
3. **Test with Node.js onnxruntime**

For maximum compatibility:
- Use float32 for production models
- Use bfloat16/float16 for training efficiency
- Convert to float32 for deployment

## Testing Different Model Types

```javascript
// In test_inference.js
const CONFIG = {
	// Update to your model path
	modelPath: '../../outputs/trigor/YOUR_MODEL/model_float32.onnx',
	vocabSize: 259,
	// ...
};
```

## Platform Support Matrix

| Platform | float32 | float16 | bfloat16 |
|----------|---------|---------|----------|
| Python onnxruntime | ✓ | ✓ | ✓ |
| Node.js onnxruntime | ✓ | ✓ | ✗ |
| Browser onnxruntime-web | ✓ | Partial | ✗ |

## Future Work

- Add automatic dtype conversion in export script
- Provide conversion utility for existing models
- Document dtype selection best practices
