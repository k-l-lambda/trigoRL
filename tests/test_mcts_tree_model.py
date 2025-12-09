"""
MCTS Single Step Comparison using Tree Model (not prefix-cached)

This matches TypeScript's tree-attention model for fair comparison.
"""

import sys
sys.path.insert(0, "/home/camus/work/trigoRL")

import numpy as np
import onnxruntime as ort
from trigor.data.tokenizer import TGNTokenizer


def main():
    print("=" * 80)
    print("MCTS Single Step - Python Tree Model (matching TypeScript)")
    print("=" * 80)
    print()

    # Use tree model (same as TypeScript)
    model_path = "/home/camus/work/trigo/trigo-web/public/onnx/20251204-trigo-value-gpt2-l6-h64-251125-lr500/GPT2CausalLM_ep0019_tree.onnx"

    print(f"Loading tree model: {model_path}")
    tree_model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("✓ Model loaded")
    print()

    # Game configuration: 5x5 empty board
    # TypeScript adds "1. " for Black's turn on empty board
    tgn_prefix = "[Board 5x5]\n\n1. "

    print("Game Configuration:")
    print("  Board: 5×5×1")
    print("  Position: Empty board (Move 1)")
    print()

    # Tokenize prefix
    # Note: Model was trained with START token (add_special_tokens=True)
    # But TypeScript doesn't add START token - this is a TypeScript bug
    # For now, we match TypeScript (no START) to test consistency
    tokenizer = TGNTokenizer()
    prefix_tensor = tokenizer.encode(tgn_prefix, add_special_tokens=False, padding=False)
    prefix_tokens = prefix_tensor.tolist()  # No START token to match TypeScript

    print(f"Prefix: {repr(tgn_prefix)}")
    print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
    print()

    # Build candidate moves for 5x5 board
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
        tree_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        token_sequences.append(tree_tokens)

    # Add Pass
    pass_tokens = tokenizer.encode("Pass", add_special_tokens=False, padding=False).tolist()
    pass_tree = pass_tokens[:-1] if len(pass_tokens) > 1 else pass_tokens
    token_sequences.append(pass_tree)

    # Build prefix tree
    def build_prefix_tree(token_arrays):
        if not token_arrays:
            return [], [], [], 0

        evaluated_ids = []
        parent = []
        move_to_leaf = [-1] * len(token_arrays)

        def build_recursive(seqs, parent_pos):
            nonlocal evaluated_ids, parent, move_to_leaf

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

    print(f"Tree structure:")
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

    # Run tree model
    print("Running tree model inference...")
    prefix_input = np.array([prefix_tokens], dtype=np.int64)
    evaluated_input = np.array([evaluated_ids], dtype=np.int64)
    mask_input = mask.reshape(1, num_nodes, num_nodes)

    logits = tree_model.run(None, {
        'prefix_ids': prefix_input,
        'evaluated_ids': evaluated_input,
        'evaluated_mask': mask_input
    })[0]

    print(f"  Logits shape: {logits.shape}")
    print(f"  (num_nodes={num_nodes}, logits_len={logits.shape[1]})")
    print()

    # Score moves using TypeScript algorithm
    def softmax_at_position(logits_array, position, vocab_size):
        position_logits = logits_array[0, position, :]
        max_logit = np.max(position_logits)
        exp_vals = np.exp(position_logits - max_logit)
        return exp_vals / np.sum(exp_vals)

    all_moves = coords + ["Pass"]
    all_tokens = []
    for coord in coords:
        tokens = tokenizer.encode(coord, add_special_tokens=False, padding=False).tolist()
        all_tokens.append(tokens)
    pass_tokens_full = tokenizer.encode("Pass", add_special_tokens=False, padding=False).tolist()
    all_tokens.append(pass_tokens_full)

    vocab_size = logits.shape[2]
    num_evaluated = logits.shape[1] - 1  # logits has m+1 positions
    MIN_PROB = 1e-10

    move_scores = []
    for move_idx, (move, tokens) in enumerate(zip(all_moves, all_tokens)):
        if len(tokens) == 0:
            continue

        leaf_pos = move_to_leaf[move_idx]
        log_prob = 0.0

        if leaf_pos == -1:
            first_token = tokens[0]
            probs = softmax_at_position(logits, 0, vocab_size)
            prob = max(probs[first_token], MIN_PROB)
            log_prob = np.log(prob)
        else:
            path_reverse = []
            pos = leaf_pos
            while pos != -1:
                path_reverse.append(pos)
                pos = parent[pos]
            path = path_reverse[::-1]

            # 1. Root token (from logits[0])
            if path:
                root_token = evaluated_ids[path[0]]
                probs = softmax_at_position(logits, 0, vocab_size)
                prob = max(probs[root_token], MIN_PROB)
                log_prob += np.log(prob)

            # 2. Intermediate transitions (parent→child)
            for j in range(1, len(path)):
                parent_pos = path[j - 1]
                child_token = evaluated_ids[path[j]]
                logits_index = parent_pos + 1  # TypeScript uses +1
                if logits_index <= num_evaluated:
                    probs = softmax_at_position(logits, logits_index, vocab_size)
                    prob = max(probs[child_token], MIN_PROB)
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(MIN_PROB)

            # 3. Last token (from leaf output)
            if path:
                leaf = path[-1]
                last_token = tokens[-1]
                logits_index = leaf + 1  # TypeScript uses +1
                if logits_index <= num_evaluated:
                    probs = softmax_at_position(logits, logits_index, vocab_size)
                    prob = max(probs[last_token], MIN_PROB)
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(MIN_PROB)

        move_scores.append((move, log_prob, tokens))

    move_scores.sort(key=lambda x: -x[1])

    # Compute priors
    max_score = max(s[1] for s in move_scores)
    exp_scores = [np.exp(s[1] - max_score) for s in move_scores]
    sum_exp = sum(exp_scores)
    priors = [e / sum_exp for e in exp_scores]

    print("=" * 80)
    print("Policy Results (Python Tree Model)")
    print("=" * 80)
    print()
    print("Top 5 moves by log score:")
    for i in range(min(5, len(move_scores))):
        move, score, tokens = move_scores[i]
        prior = priors[i]
        print(f"  {i+1}. {move} log_score={score:.6f} prior={prior:.6f}")

    print()
    print("=" * 80)
    print("TypeScript Results (from testMCTSSingleStep.ts):")
    print("=" * 80)
    print("  1. zb log_score=-7.234940 prior=0.063014")
    print("  2. zz log_score=-7.253980 prior=0.061826")
    print("  3. zy log_score=-7.285677 prior=0.059897")
    print("  4. za log_score=-7.292008 prior=0.059519")
    print("  5. ab log_score=-7.374583 prior=0.054802")
    print()

    # Check match
    ts_top5 = ["zb", "zz", "zy", "za", "ab"]
    py_top5 = [m[0] for m in move_scores[:5]]

    if py_top5 == ts_top5:
        print("✓ Top 5 moves MATCH TypeScript exactly!")
    else:
        common = set(ts_top5) & set(py_top5)
        print(f"Top 5 comparison:")
        print(f"  Python: {py_top5}")
        print(f"  TypeScript: {ts_top5}")
        print(f"  Common: {len(common)}/5 ({common})")


if __name__ == "__main__":
    main()
