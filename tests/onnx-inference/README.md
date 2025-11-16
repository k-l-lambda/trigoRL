# ONNX Inference Test (Node.js)

This directory contains Node.js tests for validating exported ONNX models from TrigoRL.

## Overview

The test suite validates:
- Basic model inference
- Variable batch sizes
- Variable sequence lengths
- Autoregressive token generation
- Output shape and dtype validation

## Prerequisites

- Node.js >= 18.0.0
- npm or yarn

## Setup

```bash
cd tests/onnx-inference
npm install
```

## Running Tests

### Basic Test

```bash
npm test
```

### Verbose Output

```bash
npm run test:verbose
```

### Direct Execution

```bash
node test_inference.js
```

## Test Configuration

Edit `test_inference.js` to configure:

```javascript
const CONFIG = {
	modelPath: '../../outputs/trigor/.../model.onnx',  // Model path
	vocabSize: 259,                                      // Vocabulary size
	tests: {
		basicInference: true,                          // Enable basic test
		batchSizes: [1, 2, 4],                         // Batch sizes to test
		seqLengths: [64, 128, 256],                    // Sequence lengths to test
		generation: true,                              // Enable generation test
		generationTokens: 10,                          // Tokens to generate
	}
};
```

## Test Suite

### Test 1: Basic Inference
- Single forward pass with batch=1, seq_len=256
- Validates output shape and dtype
- Measures inference time

### Test 2: Variable Batch Sizes
- Tests with different batch sizes (1, 2, 4)
- Fixed sequence length of 256
- Validates shape consistency

### Test 3: Variable Sequence Lengths
- Tests with different sequence lengths (64, 128, 256)
- Fixed batch size of 1
- Validates shape consistency

### Test 4: Autoregressive Generation
- Simulates token-by-token generation
- Generates 10 tokens autoregressively
- Measures tokens/second throughput

## Expected Output

```
================================================================================
ONNX Model Inference Test Suite (Node.js)
================================================================================

Model: GPT2CausalLM_ep0015.onnx
Size: 2.33 MB
Path: /path/to/model.onnx

--------------------------------------------------------------------------------
Creating inference session...
✓ Session created

--------------------------------------------------------------------------------
Model Information
--------------------------------------------------------------------------------

Inputs:
  [0] input_ids

Outputs:
  [0] logits

================================================================================
TEST 1: Basic Inference
================================================================================

Running: batch=1, seq_len=256
  Input shape: [1, 256]
  Output shape: [1, 256, 259]
  Output dtype: float32
  Inference time: 45ms
  Sample predictions: [123, 45, 67, 89, 12, 34, 56, 78, 90, 11]
  Logits range: [-8.234, 7.891]
  ✓ Test passed

...

================================================================================
Test Summary
================================================================================
Total tests: 10
Passed: 10
Failed: 0

✓ All tests passed!
================================================================================
```

## Model Requirements

The test expects an ONNX model with:

**Input:**
- Name: `input_ids`
- Type: `int64`
- Shape: `[batch_size, seq_len]`

**Output:**
- Name: `logits`
- Type: `float32`
- Shape: `[batch_size, seq_len, vocab_size]`

**Important:** The model must use `float32` dtype. Models exported with `bfloat16` are not supported by onnxruntime-node. See [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) for details and solutions.

## Troubleshooting

### Model Not Found

```
Error: Model file not found: /path/to/model.onnx
```

Solution: Update `CONFIG.modelPath` to point to your ONNX model file.

### Installation Errors

If `npm install` fails to download onnxruntime binaries:

```bash
# Set proxy if needed
npm config set proxy http://localhost:1091
npm config set https-proxy http://localhost:1091

# Retry installation
npm install
```

### Shape Mismatch

```
Error: Shape mismatch! Expected [1, 256, 259], got [1, 256, 260]
```

Solution: Update `CONFIG.vocabSize` to match your model's vocabulary size.

## Performance Benchmarks

Typical performance on CPU (Intel i7):

- Basic inference (batch=1, seq=256): ~40-60ms
- Batch=4, seq=256: ~150-200ms
- Generation (10 tokens): ~15-25 tokens/second

Performance varies based on:
- CPU model and cores
- Model architecture size
- Sequence length
- Batch size

## Integration

This test can be integrated into CI/CD:

```yaml
# .github/workflows/test.yml
- name: Test ONNX Inference
  run: |
    cd tests/onnx-inference
    npm install
    npm test
```

## See Also

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime Node.js API](https://onnxruntime.ai/docs/api/js/index.html)
- TrigoRL ONNX Export: `../../exportOnnx.py`
- ONNX Export Guide: `../../docs/onnx_export_guide.md`
