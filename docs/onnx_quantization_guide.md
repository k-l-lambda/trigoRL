# ONNX Model Quantization Guide

This guide explains how to quantize TrigoRL models for faster inference and smaller file sizes.

## Supported Quantization Types

ONNX Runtime supports multiple quantization formats:

### 8-bit Quantization (Recommended)
- **int8**: Signed 8-bit integers (-128 to 127)
- **uint8**: Unsigned 8-bit integers (0 to 255)
- **Combinations**: U8U8, U8S8, S8U8, S8S8 (default)

**Benefits:**
- ~4x smaller model size
- ~2-4x faster inference on CPU
- Works with all ONNX Runtime backends (Python, Node.js, Web)
- Minimal accuracy loss (<1% with proper calibration)

### 4-bit Quantization (For very large models)
- **int4/uint4**: 4-bit integers
- Weight-only quantization
- ~8x smaller than float32
- Currently limited to MatMul and Gather operators

### Other Types
- **float16**: 2x smaller, GPU optimized
- **bfloat16**: 2x smaller, training-friendly (limited Node.js support)
- **float32**: Full precision (baseline)

## Quantization Methods

### 1. Dynamic Quantization (Easiest)

**Use when:** You don't have representative calibration data

**Pros:**
- No calibration data needed
- Simple one-line command
- Good for weight-heavy models

**Cons:**
- Only quantizes weights, not activations
- Slightly less speedup than static quantization

**Example:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input='GPT2CausalLM_ep0015.onnx',
    model_output='GPT2CausalLM_ep0015_int8.onnx',
    weight_type=QuantType.QInt8,
    optimize_model=True,
)
```

### 2. Static Quantization (Best accuracy)

**Use when:** You have representative input data for calibration

**Pros:**
- Quantizes both weights AND activations
- Maximum inference speedup
- Better accuracy with proper calibration

**Cons:**
- Requires calibration dataset
- More complex setup

**Example:**
```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np

class TrigoCalibrationDataReader(CalibrationDataReader):
    def __init__(self, vocab_size=259, seq_len=256, num_samples=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.current_idx = 0

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None

        # Generate random calibration data
        input_ids = np.random.randint(
            0, self.vocab_size,
            (1, self.seq_len),
            dtype=np.int64
        )

        self.current_idx += 1
        return {'input_ids': input_ids}

    def rewind(self):
        self.current_idx = 0

# Quantize
calibration_reader = TrigoCalibrationDataReader()

quantize_static(
    model_input='GPT2CausalLM_ep0015.onnx',
    model_output='GPT2CausalLM_ep0015_int8_static.onnx',
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,  # or QuantFormat.QOperator
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    optimize_model=True,
)
```

### 3. QAT (Quantization-Aware Training)

**Use when:** Maximum accuracy is critical

Train with quantization simulation to learn quantization-friendly weights.

**Note:** Requires modifying training code - not covered in this guide.

## Complete Quantization Script

```python
#!/usr/bin/env python3
"""
Quantize TrigoRL ONNX models to INT8.

Usage:
    python quantize_model.py model.onnx --method dynamic
    python quantize_model.py model.onnx --method static --samples 100
"""

import argparse
import numpy as np
from pathlib import Path
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)


class TrigoCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for Trigo models."""

    def __init__(self, vocab_size=259, seq_len=256, num_samples=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.current_idx = 0

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None

        input_ids = np.random.randint(
            0, self.vocab_size,
            (1, self.seq_len),
            dtype=np.int64
        )

        self.current_idx += 1
        return {'input_ids': input_ids}

    def rewind(self):
        self.current_idx = 0


def quantize_model_dynamic(input_path, output_path):
    """Dynamic quantization (weights only)."""
    print(f"Dynamic quantization: {input_path} -> {output_path}")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    print("✓ Dynamic quantization complete")


def quantize_model_static(input_path, output_path, num_samples=100):
    """Static quantization (weights + activations)."""
    print(f"Static quantization: {input_path} -> {output_path}")
    print(f"Calibration samples: {num_samples}")

    calibration_reader = TrigoCalibrationDataReader(num_samples=num_samples)

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        optimize_model=True,
    )

    print("✓ Static quantization complete")


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX model to INT8')
    parser.add_argument('model', type=str, help='Input ONNX model path')
    parser.add_argument('--method', choices=['dynamic', 'static'], default='dynamic',
                        help='Quantization method')
    parser.add_argument('--output', type=str, help='Output model path (default: auto)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of calibration samples for static quantization')

    args = parser.parse_args()

    input_path = Path(args.model)

    if not input_path.exists():
        print(f"Error: Model file not found: {input_path}")
        return 1

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f'_int8_{args.method}'
        output_path = input_path.parent / f"{input_path.stem}{suffix}.onnx"

    # Print info
    input_size = input_path.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print("ONNX Model Quantization")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Method: {args.method}")
    print(f"Input size: {input_size:.2f} MB")
    print()

    # Quantize
    if args.method == 'dynamic':
        quantize_model_dynamic(input_path, output_path)
    else:
        quantize_model_static(input_path, output_path, args.samples)

    # Print result
    output_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = input_size / output_size

    print()
    print("=" * 80)
    print("Quantization Complete")
    print("=" * 80)
    print(f"Output size: {output_size:.2f} MB")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Saved: {input_size - output_size:.2f} MB")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
```

## Usage Examples

### Basic Quantization
```bash
# Dynamic quantization (easiest)
python quantize_model.py outputs/trigor/.../GPT2CausalLM_ep0015.onnx

# Static quantization (better accuracy)
python quantize_model.py outputs/trigor/.../GPT2CausalLM_ep0015.onnx --method static
```

### Custom Options
```bash
# Custom output path
python quantize_model.py model.onnx --output model_quantized.onnx

# More calibration samples
python quantize_model.py model.onnx --method static --samples 500
```

## Testing Quantized Models

### Python Test
```python
import onnxruntime as ort
import numpy as np

# Load quantized model
session = ort.InferenceSession('model_int8.onnx')

# Run inference
input_ids = np.random.randint(0, 259, (1, 256), dtype=np.int64)
outputs = session.run(['logits'], {'input_ids': input_ids})

print(f"Output shape: {outputs[0].shape}")
print(f"Output dtype: {outputs[0].dtype}")
```

### Node.js Test
```javascript
// Quantized models work seamlessly with Node.js
const ort = require('onnxruntime-node');

const session = await ort.InferenceSession.create('model_int8.onnx');
// Use normally - int8 is fully supported
```

## Performance Comparison

Typical results for GPT-2 style models:

| Format | Size | Inference Time | Accuracy |
|--------|------|---------------|----------|
| float32 | 100 MB | 50ms | Baseline |
| float16 | 50 MB | 35ms | ~99.9% |
| int8 (dynamic) | 25 MB | 20ms | ~99.5% |
| int8 (static) | 25 MB | 15ms | ~99.8% |
| int4 | 13 MB | 25ms | ~98% |

*Times are approximate for batch=1, seq=256 on CPU*

## Best Practices

1. **Always test accuracy** after quantization
2. **Use static quantization** when you have calibration data
3. **Start with dynamic** for quick results
4. **Use representative data** for calibration
5. **Validate on Node.js** if targeting browser/edge deployment

## Troubleshooting

### Accuracy Drop Too Large

```python
# Try per-channel quantization
per_channel=True  # Default

# Use more calibration samples
num_samples=500  # Default is 100

# Try QOperator format instead of QDQ
quant_format=QuantFormat.QOperator
```

### Unsupported Operators

Some operations may not have quantized versions. Use:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='model.onnx',
    model_output='model_int8.onnx',
    weight_type=QuantType.QInt8,
    nodes_to_exclude=['problematic_node_name']  # Skip specific nodes
)
```

## See Also

- [ONNX Runtime Quantization Docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Quantization Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization)
- Export guide: `docs/onnx_export_guide.md`
- Node.js test: `tests/onnx-inference/`
