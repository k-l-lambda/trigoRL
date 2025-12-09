"""
Test tree model inference using PyTorch directly (no ONNX)

Load checkpoint and run inference with the same architecture as tree.onnx,
then compare with C++ prefix cache results.
"""

import torch
import numpy as np
import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigor.data.tokenizer import TGNTokenizer
from trigor.models.registry import make_model
from trigor.models.gpt2CausalLM import GPT2CausalLM


def string_to_tokens(s: str) -> List[int]:
	"""Convert string to byte tokens (ASCII encoding)"""
	return [ord(c) for c in s]


def build_prefix_tree(token_arrays: List[List[int]]) -> Dict[str, Any]:
	"""
	Build prefix tree from token arrays using recursive merging.
	Same logic as tree model export.
	"""

	class Node:
		def __init__(self, token, pos, parent):
			self.token = token
			self.pos = pos
			self.parent = parent
			self.children = []
			self.move_ends = []

	next_pos = [0]

	def build(seqs: List[Dict], parent: int | None) -> List[Node]:
		groups = {}
		for s in seqs:
			if len(s['tokens']) == 0:
				continue
			t = s['tokens'][0]
			if t not in groups:
				groups[t] = []
			groups[t].append(s)

		level_nodes = []

		for token, group in groups.items():
			pos = next_pos[0]
			next_pos[0] += 1

			node = Node(token, pos, parent)

			ends = []
			residues = []

			for g in group:
				if len(g['tokens']) == 1:
					ends.append(g['move_index'])
				else:
					residues.append({
						'move_index': g['move_index'],
						'tokens': g['tokens'][1:]
					})

			node.move_ends = ends

			if residues:
				node.children = build(residues, pos)

			level_nodes.append(node)

		return level_nodes

	seqs = [{'move_index': i, 'tokens': t} for i, t in enumerate(token_arrays)]
	roots = build(seqs, None)
	total = next_pos[0]

	evaluated_ids = [0] * total
	parent = [None] * total
	move_to_leaf_pos = [-1] * len(token_arrays)

	def dfs(n: Node):
		evaluated_ids[n.pos] = n.token
		parent[n.pos] = n.parent

		for m in n.move_ends:
			move_to_leaf_pos[m] = n.pos
		for c in n.children:
			dfs(c)

	for r in roots:
		dfs(r)

	# Build ancestor mask
	mask = [0] * (total * total)
	for i in range(total):
		p = i
		while p is not None:
			mask[i * total + p] = 1
			p = parent[p]

	return {
		'evaluated_ids': evaluated_ids,
		'mask': mask,
		'move_to_leaf_pos': move_to_leaf_pos,
		'parent': parent,
		'total': total
	}


def test_pytorch_tree_inference():
	"""Test inference using PyTorch model directly"""

	print("=" * 80)
	print("PyTorch Tree Model Inference Test")
	print("=" * 80)
	print()

	# Load checkpoint
	checkpoint_path = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/checkpoints/ep0019_val_loss_2.3693.chkpt"

	print(f"Loading checkpoint: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	print(f"Checkpoint keys: {list(checkpoint.keys())}")
	print()

	# Get hyperparameters
	state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
	if state_dict is None:
		raise ValueError(f"No state_dict found in checkpoint. Available keys: {list(checkpoint.keys())}")

	config = checkpoint.get('config') or checkpoint.get('hyper_parameters')

	if config:
		print("Configuration:")
		for key in ['model_type', 'n_layer', 'n_head', 'n_embd', 'vocab_size']:
			if key in config:
				print(f"  {key}: {config[key]}")
		print()
		hparams = config
	else:
		print("No configuration found in checkpoint")
		hparams = None

	# Create model
	print("Creating model...")

	# For GPT2CausalLM, we can use the checkpoint's hyperparameters directly
	# The Lightning module wraps the base model
	if 'model.transformer.wte.weight' in state_dict or 'transformer.wte.weight' in state_dict:
		# This is a GPT2 model
		# Extract model config from state dict shapes
		if 'model.transformer.wte.weight' in state_dict:
			vocab_size = state_dict['model.transformer.wte.weight'].shape[0]
			n_embd = state_dict['model.transformer.wte.weight'].shape[1]
			n_layer = sum(1 for k in state_dict.keys() if 'model.transformer.h.' in k and '.attn.c_attn.weight' in k)
			n_head = hparams.get('n_head', 8) if hparams else 8
		else:
			vocab_size = state_dict['transformer.wte.weight'].shape[0]
			n_embd = state_dict['transformer.wte.weight'].shape[1]
			n_layer = sum(1 for k in state_dict.keys() if 'transformer.h.' in k and '.attn.c_attn.weight' in k)
			n_head = hparams.get('n_head', 8) if hparams else 8

		print(f"  Detected config from state dict:")
		print(f"    vocab_size: {vocab_size}")
		print(f"    n_embd: {n_embd}")
		print(f"    n_layer: {n_layer}")
		print(f"    n_head: {n_head}")

		# Get max_seq_len from positional embedding shape
		if 'model.transformer.wpe.weight' in state_dict:
			max_seq_len = state_dict['model.transformer.wpe.weight'].shape[0]
		elif 'transformer.wpe.weight' in state_dict:
			max_seq_len = state_dict['transformer.wpe.weight'].shape[0]
		else:
			max_seq_len = 256  # default

		print(f"    max_seq_len: {max_seq_len}")

		model = GPT2CausalLM.from_config({
			'vocab_size': vocab_size,
			'hidden_size': n_embd,
			'num_layers': n_layer,
			'num_heads': n_head,
			'max_seq_len': max_seq_len,
		})
	else:
		raise ValueError("Cannot determine model architecture from state dict")

	# Remove 'model.' prefix if present (Lightning saves with this prefix)
	new_state_dict = {}
	for key, value in state_dict.items():
		if key.startswith('model.'):
			new_state_dict[key[6:]] = value
		else:
			new_state_dict[key] = value

	model.load_state_dict(new_state_dict, strict=False)
	model.eval()
	print("Model loaded successfully")
	print()

	# Test prefix
	tgn_prefix = "[Board 5x5]\n\n1. a0 "

	print(f"Test prefix: {repr(tgn_prefix)}")
	print()

	# Tokenize prefix
	tokenizer = TGNTokenizer()
	prefix_tensor = tokenizer.encode(tgn_prefix, add_special_tokens=False, padding=False)
	prefix_tokens = prefix_tensor.tolist()
	prefix_tokens = [1] + prefix_tokens  # Add START token

	print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
	print()

	# Generate all valid moves
	board_size = 5
	mid = board_size // 2

	def encode_ab0yz(x, y):
		def encode_axis(val):
			if val == mid:
				return '0'
			elif val < mid:
				return chr(ord('a') + val)
			else:
				return chr(ord('z') - (val - mid - 1))
		return encode_axis(x) + encode_axis(y)

	moves = []
	for x in range(board_size):
		for y in range(board_size):
			if x == 2 and y == 0:  # Skip a0
				continue
			moves.append(encode_ab0yz(x, y))
	moves.append("Pass")

	print(f"Valid moves ({len(moves)}): {moves[:5]}...")
	print()

	# Tokenize moves (exclude last token)
	move_token_arrays = []
	move_last_tokens = []
	for move in moves:
		full_tokens = string_to_tokens(move)
		tokens = full_tokens[:-1]
		last_token = full_tokens[-1]
		move_token_arrays.append(tokens)
		move_last_tokens.append(last_token)

	print(f"Example move tokens (excluding last):")
	for i in range(min(5, len(moves))):
		print(f"  {moves[i]}: tree={move_token_arrays[i]}, last={move_last_tokens[i]} ('{chr(move_last_tokens[i])}')")
	print()

	# Build prefix tree
	tree = build_prefix_tree(move_token_arrays)
	evaluated_ids = tree['evaluated_ids']
	mask = tree['mask']
	move_to_leaf_pos = tree['move_to_leaf_pos']
	total = tree['total']

	print(f"Tree structure:")
	print(f"  Total nodes: {total}")
	print(f"  Evaluated IDs: {evaluated_ids}")
	print(f"  Example moves -> leaf positions: {[(moves[i], move_to_leaf_pos[i]) for i in range(min(5, len(moves)))]}")
	print()

	# Prepare PyTorch inputs
	# For tree model: we need to create full sequence with prefix + evaluated tokens
	# and use attention mask to control which tokens can attend to which

	# Create input: [prefix_tokens, evaluated_tokens]
	# Shape: [batch_size, prefix_len + num_nodes]
	prefix_len = len(prefix_tokens)
	total_len = prefix_len + total

	input_ids = prefix_tokens + evaluated_ids
	input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)

	# Create attention mask: [batch_size, total_len, total_len]
	# Prefix tokens can attend to all prefix tokens
	# Each evaluated token can attend to: all prefix + its ancestors in tree
	attention_mask = torch.zeros(1, total_len, total_len, dtype=torch.float32)

	# Prefix attends to prefix
	attention_mask[0, :prefix_len, :prefix_len] = 1.0

	# Evaluated tokens attend to prefix
	attention_mask[0, prefix_len:, :prefix_len] = 1.0

	# Evaluated tokens attend to their ancestors
	mask_2d = np.array(mask, dtype=np.float32).reshape(total, total)
	attention_mask[0, prefix_len:, prefix_len:] = torch.from_numpy(mask_2d)

	print(f"PyTorch input shapes:")
	print(f"  input_ids: {input_ids_tensor.shape}")
	print(f"  attention_mask: {attention_mask.shape}")
	print()

	# Run inference
	print("Running PyTorch inference...")
	with torch.no_grad():
		outputs = model(input_ids_tensor, attention_mask=attention_mask)
		logits = outputs.logits  # [batch_size, seq_len, vocab_size]

	print(f"Output logits shape: {logits.shape}")
	print()

	# Extract logits for each move
	# The logit for each move is at the position of its leaf node + prefix_len
	# We want the logit for the last token (which was excluded from tree)

	logits_2d = logits[0]  # [seq_len, vocab_size]

	print("=" * 80)
	print("Results - PyTorch Tree Model")
	print("=" * 80)
	print()

	print(f"{'Move':<10} | {'Leaf Pos':<10} | {'Last Token':<15} | {'Logit':<12}")
	print("-" * 65)

	results = []
	for move_idx, move_str in enumerate(moves):
		leaf_pos = move_to_leaf_pos[move_idx]
		last_token = move_last_tokens[move_idx]

		# Get logit at position: prefix_len + leaf_pos
		# This gives us the logit AFTER processing the leaf node
		# We want the logit for the last token
		output_pos = prefix_len + leaf_pos
		logit = logits_2d[output_pos, last_token].item()

		results.append({
			'move': move_str,
			'logit': float(logit),
			'leaf_pos': leaf_pos,
			'last_token': last_token
		})

		print(f"{move_str:<10} | {leaf_pos:<10} | {last_token:<3} ('{chr(last_token)}')      | {logit:>12.6f}")

	print()

	# Print top 10
	print("Top 10 by logit:")
	results_sorted = sorted(results, key=lambda r: r['logit'], reverse=True)
	print(f"{'Rank':<6} | {'Move':<10} | {'Logit':<12}")
	print("-" * 40)
	for rank, r in enumerate(results_sorted[:10], 1):
		print(f"{rank:<6} | {r['move']:<10} | {r['logit']:>12.6f}")

	print()

	# Compare with C++ prefix cache
	print("=" * 80)
	print("Comparison with C++ Prefix Cache")
	print("=" * 80)
	print()

	cpp_results = {
		'aa': 0.323845, 'ab': 0.169717, 'a0': 0.130439, 'ay': 0.124742, 'az': 0.169002,
		'ba': 0.322231, 'bb': 0.168808, 'b0': 0.135104, 'by': 0.125019, 'bz': 0.165614,
		'0b': 0.168893, '00': 0.131113, '0y': 0.123505, '0z': 0.165950,
		'ya': 0.325722, 'yb': 0.171871, 'y0': 0.132299, 'yy': 0.128370, 'yz': 0.169590,
		'za': 0.318889, 'zb': 0.168740, 'z0': 0.132069, 'zy': 0.124033, 'zz': 0.166124,
		'Pass': -1.259445
	}

	print(f"{'Move':<10} | {'PyTorch':<14} | {'C++':<14} | {'Diff':<12}")
	print("-" * 60)

	max_diff = 0.0
	for r in results:
		move = r['move']
		pt_logit = r['logit']
		cpp_logit = cpp_results.get(move, float('nan'))
		diff = abs(pt_logit - cpp_logit)
		max_diff = max(max_diff, diff)

		print(f"{move:<10} | {pt_logit:>14.6f} | {cpp_logit:>14.6f} | {diff:>12.6f}")

	print()
	print(f"Maximum difference: {max_diff:.6f}")
	print()

	if max_diff < 0.01:
		print("✓ PyTorch matches C++ prefix cache!")
		print()
		print("This would mean the architectures are equivalent.")
	else:
		print("✗ PyTorch tree model differs from C++ prefix cache")
		print()

		# Check rankings
		pt_top5 = [r['move'] for r in results_sorted[:5]]
		cpp_sorted = sorted(cpp_results.items(), key=lambda x: x[1], reverse=True)
		cpp_top5 = [move for move, _ in cpp_sorted[:5]]

		overlap = set(pt_top5) & set(cpp_top5)

		print(f"Rankings comparison:")
		print(f"  PyTorch top 5: {pt_top5}")
		print(f"  C++ top 5: {cpp_top5}")
		print(f"  Overlap: {overlap} ({len(overlap)}/5)")
		print()

		if len(overlap) >= 4:
			print("High overlap in top 5 - likely just different output scales")
		else:
			print("Low overlap - fundamentally different predictions")
			print()
			print("This suggests a BUG in one of the implementations:")
			print("1. PyTorch tree implementation (this test)")
			print("2. C++ prefix cache implementation")
			print()
			print("Need to verify PyTorch implementation is correct first.")

	return results


if __name__ == "__main__":
	test_pytorch_tree_inference()
