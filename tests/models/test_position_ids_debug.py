"""
Debug: Verify that position_ids are actually being used by the model
"""

import torch
from trigor.models.gpt2CausalLM import GPT2CausalLM
from omegaconf import OmegaConf


def test_position_ids_effect():
	"""
	Test if position_ids actually affect the output.

	Compare:
	  - [a, b] with position_ids [0, 1]
	  - [a, b] with position_ids [5, 6]

	These should produce DIFFERENT results if position_ids are working.
	"""
	torch.manual_seed(42)
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,
	})
	model = GPT2CausalLM.from_config(config)
	model.eval()

	a, b = 40, 50
	input_ids = torch.tensor([[a, b]], dtype=torch.long)

	# Test 1: position_ids [0, 1]
	position_ids_1 = torch.tensor([[0, 1]], dtype=torch.long)

	# Test 2: position_ids [5, 6]
	position_ids_2 = torch.tensor([[5, 6]], dtype=torch.long)

	with torch.no_grad():
		output_1 = model(input_ids, position_ids=position_ids_1)
		output_2 = model(input_ids, position_ids=position_ids_2)

	logits_1 = output_1.logits if hasattr(output_1, 'logits') else output_1
	logits_2 = output_2.logits if hasattr(output_2, 'logits') else output_2

	diff = torch.abs(logits_1 - logits_2).max().item()

	print("=" * 80)
	print("Position IDs Effect Test")
	print("=" * 80)
	print()
	print("Input: [a, b] (same tokens)")
	print(f"  Test 1: position_ids = [0, 1]")
	print(f"  Test 2: position_ids = [5, 6]")
	print()
	print(f"Max absolute difference: {diff:.6e}")
	print()

	if diff > 1e-5:
		print("✓ Position IDs ARE being used (outputs differ)")
	else:
		print("❌ Position IDs are NOT being used (outputs identical)")
	print()


def test_token_identity():
	"""
	Test if different tokens produce different outputs.

	Compare:
	  - [a, a] with position_ids [0, 1]
	  - [a, b] with position_ids [0, 1]

	These should produce DIFFERENT results.
	"""
	torch.manual_seed(42)
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,
	})
	model = GPT2CausalLM.from_config(config)
	model.eval()

	a, b = 40, 50
	position_ids = torch.tensor([[0, 1]], dtype=torch.long)

	# Test 1: [a, a]
	input_ids_1 = torch.tensor([[a, a]], dtype=torch.long)

	# Test 2: [a, b]
	input_ids_2 = torch.tensor([[a, b]], dtype=torch.long)

	with torch.no_grad():
		output_1 = model(input_ids_1, position_ids=position_ids)
		output_2 = model(input_ids_2, position_ids=position_ids)

	logits_1 = output_1.logits if hasattr(output_1, 'logits') else output_1
	logits_2 = output_2.logits if hasattr(output_2, 'logits') else output_2

	diff = torch.abs(logits_1 - logits_2).max().item()

	print("=" * 80)
	print("Token Identity Test")
	print("=" * 80)
	print()
	print("Position IDs: [0, 1] (same)")
	print(f"  Test 1: input_ids = [a, a]")
	print(f"  Test 2: input_ids = [a, b]")
	print()
	print(f"Max absolute difference: {diff:.6e}")
	print()

	if diff > 1e-5:
		print("✓ Token embeddings ARE working (outputs differ)")
	else:
		print("❌ Token embeddings are NOT working (outputs identical)")
	print()


if __name__ == '__main__':
	test_position_ids_effect()
	test_token_identity()
