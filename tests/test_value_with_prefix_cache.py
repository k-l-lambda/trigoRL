"""
Test value inference with prefix cache

Compare C++ value inference result with PyTorch ONNX prefix cache mode.
"""

import torch
import numpy as np
import onnxruntime as ort

from trigor.data.tokenizer import TGNTokenizer


def main():
    print("=" * 80)
    print("Value Inference with Prefix Cache Test")
    print("=" * 80)
    print()

    model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached"

    # Test prefix - same as C++ test
    tgn_prefix = "[Board 5x5]\n\n1. a0 "

    print(f"Test prefix: {repr(tgn_prefix)}")
    print()

    # Tokenize
    tokenizer = TGNTokenizer()
    prefix_tensor = tokenizer.encode(tgn_prefix, add_special_tokens=False, padding=False)
    prefix_tokens = [1] + prefix_tensor.tolist()  # Add START token

    print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
    print()

    # Load ONNX models
    print("Loading ONNX models...")
    prefix_model = ort.InferenceSession(
        f"{model_dir}/base_model_prefix.onnx",
        providers=['CPUExecutionProvider']
    )
    eval_cached_model = ort.InferenceSession(
        f"{model_dir}/base_model_eval_cached.onnx",
        providers=['CPUExecutionProvider']
    )
    value_head = ort.InferenceSession(
        f"{model_dir}/value_head.onnx",
        providers=['CPUExecutionProvider']
    )
    print("✓ Models loaded")
    print()

    # STEP 1: Compute prefix cache
    print("[STEP 1] Computing prefix cache...")
    prefix_input = np.array([prefix_tokens], dtype=np.int64)
    cache_outputs = prefix_model.run(None, {'prefix_ids': prefix_input})

    # Extract KV caches
    num_layers = len(cache_outputs) // 2
    kv_cache = {}
    for i in range(num_layers):
        kv_cache[f'past_key_{i}'] = cache_outputs[i*2]
        kv_cache[f'past_value_{i}'] = cache_outputs[i*2+1]

    print(f"✓ Prefix cache computed ({num_layers} layers)")
    print()

    # STEP 2: Evaluate VALUE token with cache
    print("[STEP 2] Evaluating VALUE token with cache...")
    VALUE_TOKEN = 3
    evaluated_ids = [VALUE_TOKEN]
    evaluated_mask = [[1.0]]  # Single token, trivial mask

    eval_input = {
        'evaluated_ids': np.array([evaluated_ids], dtype=np.int64),
        'evaluated_mask': np.array(evaluated_mask, dtype=np.float32).reshape(1, 1, 1),
        **kv_cache
    }

    hidden_outputs = eval_cached_model.run(None, eval_input)
    hidden_states = hidden_outputs[0]  # [batch, 1, hidden_dim]

    print(f"✓ Hidden states computed: {hidden_states.shape}")
    print()

    # STEP 3: Run value head
    print("[STEP 3] Running value head...")
    # Value head expects [batch, hidden_dim], squeeze out seq_len dimension
    hidden_for_value = hidden_states.squeeze(1)  # [batch, hidden_dim]

    value_outputs = value_head.run(None, {'hidden_states': hidden_for_value})
    # Output is already a scalar in shape [1]
    value = float(value_outputs[0][0])

    print(f"✓ Value computed: {value:.6f}")
    print()

    # Compare with C++
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"PyTorch ONNX value: {value:.6f}")
    print(f"C++ value:          -0.139135")
    print()

    diff = abs(value - (-0.139135))
    if diff < 0.001:
        print(f"✓ SUCCESS! PyTorch matches C++ (diff = {diff:.6f})")
    else:
        print(f"✗ MISMATCH! PyTorch differs from C++ (diff = {diff:.6f})")


if __name__ == "__main__":
    main()
