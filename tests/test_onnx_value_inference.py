#!/usr/bin/env python3
"""
Test ONNX model value inference to compare with C++.
Uses the same inputs and model files that C++ uses.
"""

import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_onnx_value_inference():
    """Test ONNX value inference matching C++ implementation."""

    model_dir = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached")

    print(f"Model directory: {model_dir}")

    # Load ONNX models
    prefix_model_path = model_dir / "base_model_prefix.onnx"
    eval_cached_model_path = model_dir / "base_model_eval_cached.onnx"
    value_head_path = model_dir / "value_head.onnx"

    print(f"\nLoading ONNX models...")
    print(f"  Prefix: {prefix_model_path}")
    print(f"  Eval cached: {eval_cached_model_path}")
    print(f"  Value head: {value_head_path}")

    prefix_session = ort.InferenceSession(str(prefix_model_path))
    eval_cached_session = ort.InferenceSession(str(eval_cached_model_path))
    value_session = ort.InferenceSession(str(value_head_path))

    # Print model inputs/outputs
    print("\n=== Prefix Model ===")
    for inp in prefix_session.get_inputs():
        print(f"  Input: {inp.name} {inp.shape} {inp.type}")
    for out in prefix_session.get_outputs():
        print(f"  Output: {out.name} {out.shape} {out.type}")

    print("\n=== Eval Cached Model ===")
    for inp in eval_cached_session.get_inputs():
        print(f"  Input: {inp.name} {inp.shape} {inp.type}")
    for out in eval_cached_session.get_outputs():
        print(f"  Output: {out.name} {out.shape} {out.type}")

    print("\n=== Value Head Model ===")
    for inp in value_session.get_inputs():
        print(f"  Input: {inp.name} {inp.shape} {inp.type}")
    for out in value_session.get_outputs():
        print(f"  Output: {out.name} {out.shape} {out.type}")

    # Tokenizer
    tokenizer = TGNTokenizer()

    # Test input: position after Black Pass
    test_tgn = "[Board 5x5]\n\n1. Pass"
    # Encode without special tokens, then add START manually
    tokens = tokenizer.encode(test_tgn, max_length=256, add_special_tokens=False, padding=False)

    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()

    # Handle if it's still a nested list
    if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = tokens[0]

    # Add START token (1)
    input_tokens = [1] + tokens

    print(f"\nTest TGN: {test_tgn}")
    print(f"Tokens: {input_tokens}")
    print(f"Sequence length: {len(input_tokens)}")

    # ================================================================
    # Step 1: Compute prefix cache
    # ================================================================
    print("\n" + "="*60)
    print("Step 1: Compute Prefix Cache")
    print("="*60)

    prefix_ids = np.array([input_tokens], dtype=np.int64)
    n = len(input_tokens)

    print(f"Prefix IDs shape: {prefix_ids.shape}")

    # Run prefix model
    prefix_outputs = prefix_session.run(None, {"prefix_ids": prefix_ids})

    print(f"Prefix outputs: {len(prefix_outputs)} tensors")
    for i, out in enumerate(prefix_outputs):
        print(f"  [{i}] shape: {out.shape}")

    # Extract KV cache (alternating key/value tensors)
    num_layers = len(prefix_outputs) // 2
    print(f"Number of layers: {num_layers}")

    # ================================================================
    # Step 2: VALUE token with cached inference
    # ================================================================
    print("\n" + "="*60)
    print("Step 2: VALUE Token with Cached Inference")
    print("="*60)

    # VALUE token ID = 3
    value_token_id = 3
    eval_ids = np.array([[value_token_id]], dtype=np.int64)
    eval_mask = np.array([[[1.0]]], dtype=np.float32)

    print(f"Eval IDs: {eval_ids.tolist()}")
    print(f"Eval mask shape: {eval_mask.shape}")

    # Prepare inputs for eval_cached model
    eval_inputs = {
        "evaluated_ids": eval_ids,
        "evaluated_mask": eval_mask,
    }

    # Add cached KV tensors
    for i in range(num_layers):
        key_name = f"past_key_{i}"
        value_name = f"past_value_{i}"
        eval_inputs[key_name] = prefix_outputs[i * 2]      # key
        eval_inputs[value_name] = prefix_outputs[i * 2 + 1]  # value

    print(f"\nEval inputs:")
    for name, arr in eval_inputs.items():
        if hasattr(arr, 'shape'):
            print(f"  {name}: {arr.shape}")
        else:
            print(f"  {name}: {arr}")

    # Run eval_cached model
    eval_outputs = eval_cached_session.run(None, eval_inputs)

    print(f"\nEval outputs: {len(eval_outputs)} tensors")
    for i, out in enumerate(eval_outputs):
        print(f"  [{i}] shape: {out.shape}")

    # Hidden states should be [batch, seq_len, hidden_dim]
    # For VALUE token only, should be [1, 1, hidden_dim] or [1, 2, hidden_dim] with dummy
    hidden_states = eval_outputs[0]
    print(f"\nHidden states shape: {hidden_states.shape}")

    # If shape is [1, 2, hidden_dim], position 1 is VALUE
    # If shape is [1, 1, hidden_dim], position 0 is VALUE
    if hidden_states.shape[1] == 2:
        value_hidden = hidden_states[:, 1, :]  # Position 1 is VALUE
        print("Using position 1 (after dummy)")
    else:
        value_hidden = hidden_states[:, 0, :]  # Position 0 is VALUE
        print("Using position 0")

    print(f"VALUE hidden shape: {value_hidden.shape}")
    print(f"VALUE hidden first 5: {value_hidden[0, :5].tolist()}")
    print(f"VALUE hidden norm: {np.linalg.norm(value_hidden):.6f}")

    # ================================================================
    # Step 3: Run value head
    # ================================================================
    print("\n" + "="*60)
    print("Step 3: Run Value Head")
    print("="*60)

    value_output = value_session.run(None, {"hidden_states": value_hidden})
    value = value_output[0]

    print(f"Value output shape: {value.shape}")
    print(f"Value: {value[0]:.6f}")

    # ================================================================
    # Compare with direct evaluation model
    # ================================================================
    eval_model_path = model_dir.parent / "GPT2CausalLM_ep0042_evaluation.onnx"
    if eval_model_path.exists():
        print("\n" + "="*60)
        print("Direct Evaluation Model Comparison")
        print("="*60)

        eval_direct_session = ort.InferenceSession(str(eval_model_path))

        # Print inputs
        print("Evaluation model inputs:")
        for inp in eval_direct_session.get_inputs():
            print(f"  {inp.name} {inp.shape} {inp.type}")

        # Prepare input: pad to 256
        input_padded = input_tokens + [0] * (256 - len(input_tokens))
        input_ids_direct = np.array([input_padded], dtype=np.int64)

        print(f"Input shape: {input_ids_direct.shape}")
        print(f"First 25 tokens: {input_ids_direct[0, :25].tolist()}")

        # Run direct evaluation
        direct_output = eval_direct_session.run(None, {"input_ids": input_ids_direct})
        value_direct = direct_output[0]

        print(f"\nDirect value output shape: {value_direct.shape}")
        print(f"Direct value: {value_direct[0]:.6f}")

        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Cached value:  {value[0]:.6f}")
        print(f"Direct value:  {value_direct[0]:.6f}")
        print(f"Difference:    {abs(value[0] - value_direct[0]):.6f}")
    else:
        print(f"\nEvaluation model not found: {eval_model_path}")


if __name__ == "__main__":
    test_onnx_value_inference()
