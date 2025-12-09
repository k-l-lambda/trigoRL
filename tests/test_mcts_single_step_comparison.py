"""
MCTS Single Step Comparison: Python vs C++ using same prefix-cached models

This test uses the same ONNX models as C++ for fair comparison:
- base_model_prefix.onnx
- base_model_eval_cached.onnx
- policy_head.onnx
"""

import sys
sys.path.insert(0, "/home/camus/work/trigoRL")

import numpy as np
import onnxruntime as ort
from trigor.data.tokenizer import TGNTokenizer


def main():
    print("=" * 80)
    print("MCTS Single Step Comparison (Python using prefix-cached models)")
    print("=" * 80)
    print()

    # Same model path as C++
    model_dir = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_shared_cached"

    print(f"Loading models from: {model_dir}")

    # Load ONNX models
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

    # Game configuration: 5x5 empty board, Move 1 (Black to play)
    # C++ game_to_tgn() for empty board outputs "[Board 5x5]\n\n" (no "1. " yet)
    # Then C++ adds START token at beginning
    tgn_prefix = "[Board 5x5]\n\n"  # 14 tokens with START

    print("Game Configuration:")
    print("  Board: 5×5×1")
    print("  Position: Empty board (Move 1)")
    print("  Current player: Black")
    print()

    # Tokenize prefix
    tokenizer = TGNTokenizer()
    prefix_tensor = tokenizer.encode(tgn_prefix, add_special_tokens=False, padding=False)
    prefix_tokens = [1] + prefix_tensor.tolist()  # Add START token

    print(f"Prefix: {repr(tgn_prefix)}")
    print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
    print()

    # Step 1: Compute prefix cache
    print("[STEP 1] Computing prefix cache...")
    prefix_input = np.array([prefix_tokens], dtype=np.int64)
    cache_outputs = prefix_model.run(None, {'prefix_ids': prefix_input})

    num_layers = len(cache_outputs) // 2
    kv_cache = {}
    for i in range(num_layers):
        kv_cache[f'past_key_{i}'] = cache_outputs[i*2]
        kv_cache[f'past_value_{i}'] = cache_outputs[i*2+1]

    print(f"✓ Prefix cache computed ({num_layers} layers, {len(prefix_tokens)} tokens)")
    print()

    # Step 2: Build tree for all candidate moves (same as C++)
    # For 5x5 board, candidates are 25 positions + Pass
    coords = [
        "aa", "ab", "a0", "ay", "az",
        "ba", "bb", "b0", "by", "bz",
        "0a", "0b", "00", "0y", "0z",
        "ya", "yb", "y0", "yy", "yz",
        "za", "zb", "z0", "zy", "zz"
    ]

    # Build token sequences (excluding last token for tree building)
    token_sequences = []
    for coord in coords:
        tokens = tokenizer.encode(coord, add_special_tokens=False, padding=False).tolist()
        # Exclude last token for tree building
        tree_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        token_sequences.append(tree_tokens)

    # Add Pass
    pass_tokens = tokenizer.encode("Pass", add_special_tokens=False, padding=False).tolist()
    pass_tree = pass_tokens[:-1] if len(pass_tokens) > 1 else pass_tokens
    token_sequences.append(pass_tree)

    # Build prefix tree (same algorithm as C++ and TypeScript)
    def build_prefix_tree(token_arrays):
        """Build prefix tree matching C++/TypeScript implementation"""
        if not token_arrays:
            return [], [], [], 0

        # Group by first token, preserving insertion order
        groups = []  # [(token, [sequences])]
        token_to_idx = {}

        for move_idx, tokens in enumerate(token_arrays):
            if not tokens:
                continue
            first_token = tokens[0]
            if first_token not in token_to_idx:
                token_to_idx[first_token] = len(groups)
                groups.append((first_token, [(move_idx, tokens)]))
            else:
                groups[token_to_idx[first_token]][1].append((move_idx, tokens))

        evaluated_ids = []
        parent = []
        move_to_leaf = [-1] * len(token_arrays)

        def build_recursive(seqs, parent_pos):
            nonlocal evaluated_ids, parent, move_to_leaf

            # Group by first token
            local_groups = []
            local_token_to_idx = {}

            for move_idx, tokens in seqs:
                if not tokens:
                    continue
                first_token = tokens[0]
                if first_token not in local_token_to_idx:
                    local_token_to_idx[first_token] = len(local_groups)
                    local_groups.append((first_token, [(move_idx, tokens)]))
                else:
                    local_groups[local_token_to_idx[first_token]][1].append((move_idx, tokens))

            for token, group in local_groups:
                pos = len(evaluated_ids)
                evaluated_ids.append(token)
                parent.append(parent_pos)

                ends = []
                residues = []
                for move_idx, tokens in group:
                    if len(tokens) == 1:
                        ends.append(move_idx)
                    else:
                        residues.append((move_idx, tokens[1:]))

                for move_idx in ends:
                    move_to_leaf[move_idx] = pos

                if residues:
                    build_recursive(residues, pos)

        initial_seqs = [(i, tokens) for i, tokens in enumerate(token_arrays) if tokens]
        build_recursive(initial_seqs, -1)

        return evaluated_ids, parent, move_to_leaf, len(evaluated_ids)

    evaluated_ids, parent, move_to_leaf, num_nodes = build_prefix_tree(token_sequences)

    print(f"[STEP 2] Tree structure:")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Evaluated IDs: {evaluated_ids}")
    print(f"  Parent: {parent}")
    print()

    # Build ancestor mask
    mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        p = i
        while p != -1:
            mask[i, p] = 1.0
            p = parent[p]

    # Step 3: Evaluate with cache
    print("[STEP 3] Evaluating with cache...")
    eval_input = {
        'evaluated_ids': np.array([evaluated_ids], dtype=np.int64),
        'evaluated_mask': mask.reshape(1, num_nodes, num_nodes),
        **kv_cache
    }
    hidden_outputs = eval_cached_model.run(None, eval_input)
    hidden_states = hidden_outputs[0]  # [batch, num_nodes, hidden_dim]

    print(f"  Hidden states shape: {hidden_states.shape}")

    # Step 4: Run policy head
    print("[STEP 4] Running policy head...")
    policy_outputs = policy_head.run(None, {'hidden_states': hidden_states})
    logits = policy_outputs[0]  # [batch, num_nodes, vocab_size]

    print(f"  Logits shape: {logits.shape}")
    print()

    # Step 5: Compute log scores for each move (matching C++ algorithm)
    print("=" * 80)
    print("Policy Results (Python prefix-cached)")
    print("=" * 80)
    print()

    def softmax_at_position(logits_array, position, vocab_size):
        """Compute softmax for a single position in logits array"""
        position_logits = logits_array[0, position, :]
        max_logit = np.max(position_logits)
        exp_vals = np.exp(position_logits - max_logit)
        return exp_vals / np.sum(exp_vals)

    # For each move, accumulate log probabilities along the path
    move_scores = []
    all_moves = coords + ["Pass"]
    all_tokens = []

    for coord in coords:
        tokens = tokenizer.encode(coord, add_special_tokens=False, padding=False).tolist()
        all_tokens.append(tokens)

    pass_tokens_full = tokenizer.encode("Pass", add_special_tokens=False, padding=False).tolist()
    all_tokens.append(pass_tokens_full)

    vocab_size = logits.shape[2]
    MIN_PROB = 1e-10

    for move_idx, (move, tokens) in enumerate(zip(all_moves, all_tokens)):
        if len(tokens) == 0:
            continue

        leaf_pos = move_to_leaf[move_idx]
        log_prob = 0.0

        if leaf_pos == -1:
            # Empty tree sequence - direct prediction from position 0
            first_token = tokens[0]
            probs = softmax_at_position(logits, 0, vocab_size)
            prob = max(probs[first_token], MIN_PROB)
            log_prob = np.log(prob)
        else:
            # Build path from leaf to root, then reverse
            path_reverse = []
            pos = leaf_pos
            while pos != -1:
                path_reverse.append(pos)
                pos = parent[pos]
            path = path_reverse[::-1]  # Root to leaf

            # 1. Root token (predicted from position 0)
            if path:
                root_pos = path[0]
                root_token = evaluated_ids[root_pos]
                probs = softmax_at_position(logits, 0, vocab_size)
                prob = max(probs[root_token], MIN_PROB)
                log_prob += np.log(prob)

            # 2. Intermediate transitions (parent→child)
            for j in range(1, len(path)):
                parent_pos = path[j - 1]
                child_pos = path[j]
                child_token = evaluated_ids[child_pos]

                # Parent hidden state predicts child token
                # With m+1 output format: use parent_pos + 1 (matching TreeLM)
                logits_index = parent_pos + 1
                if logits_index < logits.shape[1]:
                    probs = softmax_at_position(logits, logits_index, vocab_size)
                    prob = max(probs[child_token], MIN_PROB)
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(MIN_PROB)

            # 3. Last token (predicted from leaf)
            if path:
                leaf = path[-1]
                last_token = tokens[-1]

                # Leaf hidden state predicts last token
                # With m+1 output format: use leaf + 1 (matching TreeLM)
                logits_index = leaf + 1
                if logits_index < logits.shape[1]:
                    probs = softmax_at_position(logits, logits_index, vocab_size)
                    prob = max(probs[last_token], MIN_PROB)
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(MIN_PROB)

        move_scores.append((move, log_prob, tokens))

    # Sort by score descending
    move_scores.sort(key=lambda x: -x[1])

    # Compute priors
    max_score = max(s[1] for s in move_scores)
    exp_scores = [np.exp(s[1] - max_score) for s in move_scores]
    sum_exp = sum(exp_scores)
    priors = [e / sum_exp for e in exp_scores]

    print("Top 5 moves by log score:")
    for i in range(min(5, len(move_scores))):
        move, score, tokens = move_scores[i]
        prior = priors[i]
        print(f"  {i+1}. {move} log_score={score:.6f} prior={prior:.6f}")

    print()
    print("=" * 80)
    print("C++ Results (from test_mcts_single_step with NEW models):")
    print("=" * 80)
    print("  1. az log_score=-7.182709 prior=0.077842")
    print("  2. zz log_score=-7.219956 prior=0.074996")
    print("  3. aa log_score=-7.295769 prior=0.069521")
    print("  4. 0z log_score=-7.443516 prior=0.059972")
    print("  5. za log_score=-7.448757 prior=0.059658")
    print()

    # Check if top move matches
    cpp_top = "az"
    py_top = move_scores[0][0]
    if py_top == cpp_top:
        print(f"✓ Top move MATCHES: {py_top}")
    else:
        print(f"✗ Top move DIFFERS: Python={py_top}, C++={cpp_top}")

    # Check order similarity
    cpp_order = ["az", "zz", "aa", "0z", "za"]
    py_order = [m[0] for m in move_scores[:5]]
    common = set(cpp_order) & set(py_order)
    print(f"  Common in top 5: {len(common)}/5 ({common})")


if __name__ == "__main__":
    main()
