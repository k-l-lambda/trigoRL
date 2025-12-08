"""
Policy and Value Inference Validation - Python Reference

Compare with C++ version to validate correctness
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_validation():
    print("=" * 80)
    print("Policy & Value Inference Validation - Python Reference")
    print("=" * 80)
    print()

    model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached"

    # TGN prefix (empty 5x5 board)
    tgn_prefix = "[Board 5x5]\n\n"
    print("TGN prefix:")
    print(tgn_prefix)

    # Tokenize
    tokenizer = TGNTokenizer()
    tgn_tokens = tokenizer.encode(tgn_prefix, add_special_tokens=False)

    # Add START token
    prefix_tokens = [1] + tgn_tokens.tolist()

    print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens[:20]}")
    print()

    # Test candidates (same as C++)
    test_moves = ["aa", "ab", "a0", "ay", "az", "zz", "PASS"]

    # Build candidate sequences
    candidate_sequences = []
    for move in test_moves:
        move_tokens = tokenizer.encode(move, add_special_tokens=False)
        seq = prefix_tokens + move_tokens.tolist()
        candidate_sequences.append(seq)

    print(f"Test candidates: {' '.join(test_moves)}")
    print()

    # Build tree structure (simple - just concatenate all)
    # For validation, we use simple tree: prefix + all candidate branches
    max_len = max(len(seq) for seq in candidate_sequences)

    # Pad all sequences to same length
    padded_sequences = []
    for seq in candidate_sequences:
        padded = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded)

    evaluated_ids = np.array(padded_sequences, dtype=np.int64)
    print(f"Evaluated IDs shape: {evaluated_ids.shape}")
    print()

    # Load models
    print("Loading ONNX models...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    base_session = ort.InferenceSession(
        f"{model_dir}/base_model.onnx",
        sess_options,
        providers=['CPUExecutionProvider']
    )

    policy_session = ort.InferenceSession(
        f"{model_dir}/policy_head.onnx",
        sess_options,
        providers=['CPUExecutionProvider']
    )

    value_session = ort.InferenceSession(
        f"{model_dir}/value_head.onnx",
        sess_options,
        providers=['CPUExecutionProvider']
    )
    print("✓ Models loaded")
    print()

    # Run base model
    print("[BASE MODEL]")
    base_outputs = base_session.run(None, {'input_ids': evaluated_ids})
    hidden_states = base_outputs[0]
    print(f"Hidden states shape: {hidden_states.shape}")
    print()

    # Run policy head
    print("[POLICY INFERENCE]")
    policy_logits = policy_session.run(None, {'hidden_states': hidden_states})[0]
    print(f"Policy logits shape: {policy_logits.shape}")
    print()

    print("Policy logits:")
    for i, move in enumerate(test_moves):
        # Get last token of this move
        seq = candidate_sequences[i]
        last_token = seq[-1]
        seq_len = len(candidate_sequences[i])

        # Get logit for last token at its position
        logit = policy_logits[i, seq_len - 1, last_token]
        print(f"  {move}: seq_len={seq_len}, last_token={last_token}, logit={logit:.4f}")
    print()

    # Run value head
    print("[VALUE INFERENCE]")
    # Value uses prefix + VALUE token
    value_tokens = prefix_tokens + [3]  # VALUE token ID = 3
    value_ids = np.array([value_tokens], dtype=np.int64)

    # Run through base model
    base_value_outputs = base_session.run(None, {'input_ids': value_ids})
    value_hidden = base_value_outputs[0]

    # Get value prediction
    value_result = value_session.run(None, {'hidden_states': value_hidden})[0]
    value_score = value_result[0, -1, 0]  # Last position, first output

    print(f"Value score: {value_score:.4f}")
    print()

    print("=" * 80)
    print("Python Reference Complete - Compare with C++ output")
    print("=" * 80)


if __name__ == "__main__":
    test_validation()
