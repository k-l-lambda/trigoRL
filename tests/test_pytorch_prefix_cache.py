"""
Test PyTorch prefix cache mode (matching C++ implementation)

This test uses the ONNX exported prefix cache models to verify that
PyTorch prefix cache produces the same results as C++ prefix cache.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

from trigor.data.tokenizer import TGNTokenizer


def main():
    print("=" * 80)
    print("PyTorch Prefix Cache Mode Test (Matching C++)")
    print("=" * 80)
    print()

    # Model paths
    model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached"

    # Test prefix - same as C++
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
    policy_head = ort.InferenceSession(
        f"{model_dir}/policy_head.onnx",
        providers=['CPUExecutionProvider']
    )
    print("✓ Models loaded")
    print()

    # Tree structure from TypeScript (same as C++)
    evaluated_ids = [97, 98, 48, 122, 121, 80, 97, 115]
    mask_flat = [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1]

    num_nodes = len(evaluated_ids)
    mask_2d = np.array(mask_flat, dtype=np.float32).reshape(num_nodes, num_nodes)

    print(f"Tree structure:")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Evaluated IDs: {evaluated_ids}")
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

    # STEP 2: Evaluate with cache
    print("[STEP 2] Evaluating with cache...")
    eval_input = {
        'evaluated_ids': np.array([evaluated_ids], dtype=np.int64),
        'evaluated_mask': mask_2d.reshape(1, num_nodes, num_nodes),
        **kv_cache
    }
    hidden_outputs = eval_cached_model.run(None, eval_input)
    hidden_states = hidden_outputs[0]  # [batch, num_nodes, hidden_dim]

    print(f"✓ Hidden states computed: {hidden_states.shape}")
    print()

    # STEP 3: Run policy head
    print("[STEP 3] Running policy head...")
    policy_outputs = policy_head.run(None, {'hidden_states': hidden_states})
    logits = policy_outputs[0]  # [batch, num_nodes, vocab_size]

    print(f"✓ Policy logits computed: {logits.shape}")
    print()

    # Extract move logits
    print("=" * 80)
    print("Results - PyTorch Prefix Cache Mode")
    print("=" * 80)
    print()

    moves = ["aa", "ab", "a0", "az", "ay", "ba", "bb", "b0", "bz", "by",
             "0b", "00", "0z", "0y", "za", "zb", "z0", "zz", "zy",
             "ya", "yb", "y0", "yz", "yy", "PASS"]

    # Map moves to leaf positions and last tokens
    move_to_leaf = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 7]
    last_tokens = [
        97, 98, 48, 122, 121,  # aa, ab, a0, az, ay
        97, 98, 48, 122, 121,  # ba, bb, b0, bz, by
        98, 48, 122, 121,      # 0b, 00, 0z, 0y
        97, 98, 48, 122, 121,  # za, zb, z0, zz, zy
        97, 98, 48, 122, 121,  # ya, yb, y0, yz, yy
        83                      # PASS -> 'S'
    ]

    print("Move       | Leaf Pos   | Last Token      | Logit       ")
    print("-" * 65)

    for i, move in enumerate(moves):
        leaf_pos = move_to_leaf[i]
        last_token = last_tokens[i]
        logit = logits[0, leaf_pos, last_token]
        print(f"{move:10} | {leaf_pos:10} | {last_token:3} ('{chr(last_token):>1}')      | {logit:12.6f}")

    print()
    print("KEY COMPARISON:")
    print(f"  PyTorch prefix cache logit for \"aa\": {logits[0, 0, 97]:.6f}")
    print(f"  C++ prefix cache logit for \"aa\":     0.323845")
    print(f"  Expected (PyTorch tree mode):         4.147368")
    print()

    diff_cpp = abs(logits[0, 0, 97] - 0.323845)
    diff_tree = abs(logits[0, 0, 97] - 4.147368)

    print(f"  Diff from C++: {diff_cpp:.6f}")
    print(f"  Diff from tree mode: {diff_tree:.6f}")
    print()

    if diff_cpp < 0.01:
        print("✓ PyTorch prefix cache MATCHES C++ (both wrong!)")
    elif diff_tree < 0.01:
        print("✓ PyTorch prefix cache MATCHES tree mode (correct!)")
    else:
        print("✗ PyTorch prefix cache differs from both!")


if __name__ == "__main__":
    main()
